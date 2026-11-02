"""Point-in-time export snapshots: stability under concurrent writes/deletes.

These tests pin the exact failure mode offline backups hit: rows inserted or
soft-deleted while pages are being walked. A pinned snapshot must keep serving
the set that was live at pin time, and its keyset cursor must never skip or
repeat a row.
"""

import time

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.mixins import (
    ExportSnapshotExpired,
    ExportSnapshotNotFound,
)
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest_asyncio.fixture
async def storage(temp_db_path):
    db = SqliteVecMemoryStorage(f"{temp_db_path}/export_snapshot.db")
    await db.initialize()
    return db


async def _seed(storage, n, prefix="pinned", tags=None, store="default"):
    hashes = []
    for i in range(n):
        content = f"{prefix} memory {i} - {time.time_ns()}"
        content_hash = generate_content_hash(content)
        ok, _ = await storage.store(
            Memory(content=content, content_hash=content_hash, tags=list(tags or [])),
            store=store,
        )
        assert ok
        hashes.append(content_hash)
    return hashes


async def _walk(storage, snapshot_id, limit):
    """Walk a snapshot to the end using its keyset cursor."""
    seen = []
    cursor = 0
    while True:
        page = await storage.get_export_snapshot_page(
            snapshot_id, after_id=cursor, limit=limit
        )
        seen.extend(memory.content_hash for memory in page["memories"])
        if not page["has_more"]:
            assert page["next_cursor"] is None
            return seen, page["snapshot"]
        cursor = page["next_cursor"]


@pytest.mark.asyncio
async def test_snapshot_excludes_rows_inserted_after_pin(storage):
    pinned = await _seed(storage, 5)
    snapshot = await storage.create_export_snapshot()
    assert snapshot["total_rows"] == 5
    assert snapshot["next_cursor"] == 0

    # Background agent keeps writing after the pin.
    latecomers = await _seed(storage, 3, prefix="late")

    seen, _ = await _walk(storage, snapshot["snapshot_id"], limit=2)
    assert sorted(seen) == sorted(pinned)
    assert not (set(seen) & set(latecomers))


@pytest.mark.asyncio
async def test_snapshot_keeps_rows_soft_deleted_after_pin(storage):
    pinned = await _seed(storage, 4)
    snapshot = await storage.create_export_snapshot()

    # Delete two rows mid-backup. They vanish from the live listing ...
    for content_hash in pinned[:2]:
        ok, _ = await storage.delete(content_hash)
        assert ok
    live = await storage.get_all_memories(store="default")
    assert {m.content_hash for m in live} == set(pinned[2:])

    # ... but must still be present in the pinned backup.
    seen, _ = await _walk(storage, snapshot["snapshot_id"], limit=2)
    assert sorted(seen) == sorted(pinned)


@pytest.mark.asyncio
async def test_keyset_walk_never_skips_or_repeats_across_page_sizes(storage):
    pinned = await _seed(storage, 11)
    snapshot = await storage.create_export_snapshot()

    for limit in (1, 2, 3, 7, 50):
        seen, _ = await _walk(storage, snapshot["snapshot_id"], limit=limit)
        assert len(seen) == len(pinned)
        assert len(seen) == len(set(seen)), f"duplicate rows with limit={limit}"
        assert sorted(seen) == sorted(pinned)


@pytest.mark.asyncio
async def test_same_cursor_is_deterministic(storage):
    await _seed(storage, 6)
    snapshot = await storage.create_export_snapshot()
    sid = snapshot["snapshot_id"]

    first = await storage.get_export_snapshot_page(sid, after_id=0, limit=3)
    cursor = first["next_cursor"]
    assert first["has_more"] is True

    # A concurrent insert between calls cannot move a fixed cursor.
    await _seed(storage, 2, prefix="during")
    second = await storage.get_export_snapshot_page(sid, after_id=cursor, limit=3)

    # Re-reading the earlier cursor returns byte-identical content ordering.
    reread = await storage.get_export_snapshot_page(sid, after_id=0, limit=3)
    assert [m.content_hash for m in reread["memories"]] == [
        m.content_hash for m in first["memories"]
    ]
    # Pages never overlap: the join point is strictly after the prior last id.
    first_ids = {m.content_hash for m in first["memories"]}
    second_ids = {m.content_hash for m in second["memories"]}
    assert not (first_ids & second_ids)


@pytest.mark.asyncio
async def test_expired_snapshot_raises_and_is_reaped(storage):
    await _seed(storage, 2)
    snapshot = await storage.create_export_snapshot(ttl_seconds=1)
    sid = snapshot["snapshot_id"]

    time.sleep(1.1)
    with pytest.raises(ExportSnapshotExpired) as exc_info:
        await storage.get_export_snapshot_page(sid, after_id=0, limit=10)
    assert exc_info.value.snapshot_id == sid

    # The failed read reaps the pin; the next read reports it as unknown so a
    # client cannot mistake an expired pin for a typo'd id.
    with pytest.raises(ExportSnapshotNotFound):
        await storage.get_export_snapshot_page(sid, after_id=0, limit=10)


@pytest.mark.asyncio
async def test_unknown_and_released_snapshots_are_not_found(storage):
    with pytest.raises(ExportSnapshotNotFound):
        await storage.get_export_snapshot_page("deadbeef", after_id=0, limit=10)

    snapshot = await storage.create_export_snapshot()
    sid = snapshot["snapshot_id"]
    assert await storage.release_export_snapshot(sid) is True
    assert await storage.release_export_snapshot(sid) is False
    with pytest.raises(ExportSnapshotNotFound):
        await storage.get_export_snapshot_page(sid, after_id=0, limit=10)


@pytest.mark.asyncio
async def test_snapshot_filters_by_tag_and_store(storage):
    keep = await _seed(storage, 3, prefix="keep", tags=["backup", "work"])
    await _seed(storage, 2, prefix="skip", tags=["other"])
    other_store = await _seed(storage, 2, prefix="docs", tags=["backup"], store="docs")

    # The default pin is confined to the default store just like get_all_memories.
    default_tagged = await storage.create_export_snapshot(tags=["backup"])
    assert default_tagged["total_rows"] == 3

    # Cross-store tag filtering requires the explicit all-store pin (store=None).
    by_tag = await storage.create_export_snapshot(store=None, tags=["backup"])
    assert by_tag["total_rows"] == 5  # 3 default + 2 docs
    seen, _ = await _walk(storage, by_tag["snapshot_id"], limit=4)
    assert sorted(seen) == sorted(keep + other_store)

    scoped = await storage.create_export_snapshot(store="docs")
    assert scoped["total_rows"] == 2
    seen, _ = await _walk(storage, scoped["snapshot_id"], limit=4)
    assert sorted(seen) == sorted(other_store)

    all_stores = await storage.create_export_snapshot(store=None)
    assert all_stores["total_rows"] == 7

    with pytest.raises(ValueError):
        await storage.create_export_snapshot(tag_match="nope")


@pytest.mark.asyncio
async def test_page_arguments_validated(storage):
    snapshot = await storage.create_export_snapshot()
    sid = snapshot["snapshot_id"]

    with pytest.raises(ValueError):
        await storage.get_export_snapshot_page(sid, after_id=-1, limit=10)
    with pytest.raises(ValueError):
        await storage.get_export_snapshot_page(sid, after_id=0, limit=0)


@pytest.mark.asyncio
async def test_metadata_reports_pin_and_expiry_window(storage):
    before = time.time()
    await _seed(storage, 1)
    snapshot = await storage.create_export_snapshot(ttl_seconds=120)
    after = time.time()

    assert before <= snapshot["pinned_at"] <= after
    assert snapshot["expires_at"] == pytest.approx(snapshot["pinned_at"] + 120)
    assert snapshot["pinned_at_iso"].endswith("Z")
    assert snapshot["expires_at_iso"].endswith("Z")

    meta = await storage.get_export_snapshot_metadata(snapshot["snapshot_id"])
    assert meta["total_rows"] == 1
    assert meta["ttl_seconds"] == 120
