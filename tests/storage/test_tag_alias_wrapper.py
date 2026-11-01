# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end tests for the tag alias storage wrapper on tmp SQLite-vec DBs.

Covers the team scenario: variants (project prefix, bare name, capitalization)
registered under one canonical tag; new writes normalize; legacy rows are
grouped virtually for queries and statistics; undo restores split behavior;
the mapping survives a restart / backend swap because it lives outside the DB.
"""

import hashlib

import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.alias_wrapper import AliasedStorage, wrap_with_tag_aliases
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.tag_aliases import TagAliasRegistry


def _mem(content: str, tags) -> Memory:
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.strip().lower().encode()).hexdigest(),
        tags=tags,
    )


@pytest_asyncio.fixture
async def stack(tmp_path):
    db_path = tmp_path / "test_tag_aliases.db"
    inner = SqliteVecMemoryStorage(str(db_path))
    await inner.initialize()
    registry = TagAliasRegistry(tmp_path / "tag_aliases.json")
    storage = AliasedStorage(inner, registry)
    try:
        yield storage, inner, registry
    finally:
        await inner.close()


async def _seed_atlas_group(storage, inner):
    """Two legacy rows written with raw variants + one new write via wrapper."""
    legacy_a, legacy_b = _mem("legacy prefixed spelling", ["proj:atlas"]), \
        _mem("legacy bare project name", ["atlasproj"])
    await inner.store(legacy_a)
    await inner.store(legacy_b)

    new_row = _mem("new write with shouty spelling", ["ATLASPROJ", "backend"])
    original_tags = list(new_row.tags)
    await storage.store(new_row)
    return original_tags


async def _hashes(memories):
    return {m.content_hash for m in memories}


# ── writes ───────────────────────────────────────────────────────────────────

async def test_new_write_is_stored_under_canonical_tag(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])

    incoming = _mem("alias on the write path", ["Proj:Atlas"])
    incoming_tags = list(incoming.tags)
    await storage.store(incoming)

    stored = await inner.get_by_hash(incoming.content_hash)
    assert stored.tags == ["atlas"]
    # The caller's object is not mutated.
    assert incoming.tags == incoming_tags


async def test_wrapper_is_idempotent(stack):
    storage, inner, registry = stack
    assert wrap_with_tag_aliases(storage) is storage


# ── queries: old + new rows behave as one tag ────────────────────────────────

async def test_search_by_tag_groups_legacy_and_new_rows(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    for queried in ("atlas", "proj:atlas", "atlasproj"):
        rows = await storage.search_by_tag([queried])
        assert len(rows) == 3, queried
        # View canonicalization: no variant spelling leaks to callers.
        assert all("atlas" in (m.tags or []) for m in rows)
        assert all("proj:atlas" not in (m.tags or []) for m in rows)

    # Raw physical rows still carry their original variant spelling.
    raw_rows = await inner.search_by_tag(["proj:atlas"])
    assert raw_rows and all("atlas" not in (m.tags or []) for m in raw_rows)


async def test_and_and_or_grouping_across_two_canonical_tags(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    registry.register("borealis", ["proj:borealis"])
    await _seed_atlas_group(storage, inner)
    await inner.store(_mem("both projects note", ["atlasproj", "Proj:Borealis"]))

    or_rows = await storage.search_by_tags(["atlas", "borealis"], operation="OR")
    # 3 atlas-group rows plus the cross-group "both" row.
    assert len(or_rows) == 4

    and_rows = await storage.search_by_tags(["ATLAS", "proj:borealis"], operation="AND")
    assert len(and_rows) == 1
    assert and_rows[0].content.startswith("both projects")

    # Single-tag AND behaves like the variant-group OR: the cross-group row
    # carries atlasproj too, so it belongs to the atlas group.
    single = await storage.search_by_tags(["atlasproj"], operation="AND")
    assert len(single) == 4


async def test_get_all_memories_any_and_all_semantics(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    registry.register("borealis", ["proj:borealis"])
    await _seed_atlas_group(storage, inner)
    await inner.store(_mem("both again", ["proj:atlas", "proj:borealis"]))

    any_rows = await storage.get_all_memories(tags=["atlasproj"], tag_match="any")
    assert len(any_rows) == 4
    all_rows = await storage.get_all_memories(tags=["atlas", "borealis"], tag_match="all")
    assert len(all_rows) == 1

    # Fast path is untouched when no queried tag has aliases.
    other = await storage.get_all_memories(tags=["backend"], tag_match="any")
    assert len(other) == 1


async def test_search_memories_tag_only_returns_merged_dict_view(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    payload = await storage.search_memories(query=None, tags=["atlas"], limit=10)
    assert payload["total"] == 3
    for memory in payload["memories"]:
        assert "atlas" in memory["tags"]
        assert "proj:atlas" not in memory["tags"]


async def test_retrieve_tag_filter_expands_variants(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    results = await storage.retrieve("project", n_results=20, tags=["proj:atlas"])
    # Whatever the vector ranking returns must both match the group and expose
    # the canonical view; completeness is covered by search_by_tag tests.
    assert results
    assert all("atlas" in (r.memory.tags or []) for r in results)


# ── statistics ───────────────────────────────────────────────────────────────

async def test_tag_listings_and_counts_merge_under_canonical(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    counts = {row["tag"]: row["count"] for row in await storage.get_all_tags_with_counts()}
    assert counts.get("atlas") == 3
    assert "proj:atlas" not in counts and "atlasproj" not in counts

    assert await storage.count_memories_by_tag(["atlasproj"]) == 3
    assert await storage.count_all_memories(tags=["atlas"]) == 3


# ── deletes match the query set ──────────────────────────────────────────────

async def test_delete_by_tag_removes_every_variant(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    count, _message = await storage.delete_by_tag("atlas")
    assert count == 3
    assert await storage.count_all_memories(tags=["atlas"]) == 0


# ── undo ─────────────────────────────────────────────────────────────────────

async def test_unregister_restores_split_behavior(stack):
    storage, inner, registry = stack
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    await _seed_atlas_group(storage, inner)

    registry.unregister(canonical="atlas")

    # Old rows are scattered again under their literal spellings...
    assert len(await storage.search_by_tag(["atlas"])) == 1
    assert len(await storage.search_by_tag(["proj:atlas"])) == 1
    # ...and stats no longer merge.
    counts = {row["tag"]: row["count"] for row in await storage.get_all_tags_with_counts()}
    assert "atlas" in counts and "proj:atlas" in counts

    # New writes pass through untouched.
    fresh = _mem("after undo", ["atlasproj"])
    await storage.store(fresh)
    stored = await inner.get_by_hash(fresh.content_hash)
    assert stored.tags == ["atlasproj"]


# ── persistence across restart and backend swap ──────────────────────────────

async def test_mapping_survives_restart(tmp_path):
    db_path = tmp_path / "persist.db"
    mapping_path = tmp_path / "tag_aliases.json"

    inner = SqliteVecMemoryStorage(str(db_path))
    await inner.initialize()
    registry = TagAliasRegistry(mapping_path)
    registry.register("atlas", ["proj:atlas"])
    await AliasedStorage(inner, registry).store(_mem("first run", ["proj:atlas"]))
    await inner.close()

    # Simulate a service restart: brand-new backend + registry instances.
    inner2 = SqliteVecMemoryStorage(str(db_path))
    await inner2.initialize()
    try:
        storage2 = AliasedStorage(inner2, TagAliasRegistry(mapping_path))
        rows = await storage2.search_by_tag(["atlas"])
        assert len(rows) == 1
        assert rows[0].tags == ["atlas"]
    finally:
        await inner2.close()


async def test_mapping_applies_to_a_different_backend(tmp_path):
    """The registry file is backend-independent: a second empty backend that
    points at the same mapping immediately normalizes and virtually merges."""
    mapping_path = tmp_path / "tag_aliases.json"
    registry_a = TagAliasRegistry(mapping_path)
    registry_a.register("atlas", ["proj:atlas"])

    db_b = tmp_path / "other_backend.db"
    inner_b = SqliteVecMemoryStorage(str(db_b))
    await inner_b.initialize()
    try:
        storage_b = AliasedStorage(inner_b, TagAliasRegistry(mapping_path))
        await storage_b.store(_mem("written into another backend", ["proj:atlas"]))
        stored = await inner_b.get_by_hash(
            hashlib.sha256("written into another backend".encode()).hexdigest()
        )
        assert stored.tags == ["atlas"]
    finally:
        await inner_b.close()
