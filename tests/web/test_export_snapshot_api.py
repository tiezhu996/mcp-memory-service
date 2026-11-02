"""HTTP contract for point-in-time export snapshots.

Covers the backup workflow end to end: pin → page with a cursor while the
database keeps changing → release, plus the explicit restart signals (410
expired / 404 unknown) and scope/501 handling.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.api.export import router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import (
    AuthenticationResult,
    get_current_user,
)


def _user(scope="read"):
    return AuthenticationResult(
        authenticated=True,
        client_id="backup-client",
        scope=scope,
        auth_method="none",
    )


def _make_app(storage, scope="read"):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_current_user] = lambda: _user(scope)
    return app


def _seed(storage, n, prefix="row", store="default"):
    async def _go():
        hashes = []
        for i in range(n):
            content = f"{prefix} {i} {time.time_ns()}"
            content_hash = generate_content_hash(content)
            ok, _ = await storage.store(
                Memory(content=content, content_hash=content_hash), store=store
            )
            assert ok
            hashes.append(content_hash)
        return hashes

    return asyncio.run(_go())


@pytest.fixture
def storage(temp_db_path):
    db = SqliteVecMemoryStorage(f"{temp_db_path}/export_api.db")
    asyncio.run(db.initialize())
    yield db


def test_create_then_walk_pages_is_stable_under_concurrent_writes(storage):
    pinned = _seed(storage, 5)
    client = TestClient(_make_app(storage))

    resp = client.post("/export/snapshots", json={})
    assert resp.status_code == 200, resp.text
    created = resp.json()
    sid = created["snapshot_id"]
    assert created["total_rows"] == 5
    assert created["next_cursor"] == 0
    assert created["pinned_at_iso"].endswith("Z")
    assert created["expires_at"] > created["pinned_at"]

    # Background agent writes after the pin; pages must not pick these up.
    late = _seed(storage, 3, prefix="late")

    seen = []
    cursor = 0
    while True:
        page = client.get(
            f"/export/snapshots/{sid}/pages", params={"cursor": cursor, "limit": 2}
        )
        assert page.status_code == 200, page.text
        body = page.json()
        assert body["snapshot"]["snapshot_id"] == sid
        assert body["snapshot"]["total_rows"] == 5
        seen.extend(m["content_hash"] for m in body["memories"])
        if not body["has_more"]:
            assert body["next_cursor"] is None
            break
        cursor = body["next_cursor"]

    assert sorted(seen) == sorted(pinned)
    assert not (set(seen) & set(late))


def test_unknown_snapshot_returns_404_with_restart_instruction(storage):
    client = TestClient(_make_app(storage))
    resp = client.get("/export/snapshots/nope/pages", params={"cursor": 0})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["error"] == "export_snapshot_not_found"
    assert "POST /api/export/snapshots" in detail["error_description"]


def test_expired_snapshot_returns_410_and_tells_client_to_restart(storage, monkeypatch):
    _seed(storage, 2)
    monkeypatch.setenv("MCP_EXPORT_SNAPSHOT_TTL_SECONDS", "1")
    client = TestClient(_make_app(storage))

    sid = client.post("/export/snapshots", json={}).json()["snapshot_id"]
    time.sleep(1.1)

    resp = client.get(f"/export/snapshots/{sid}/pages", params={"cursor": 0})
    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert detail["error"] == "export_snapshot_expired"
    assert detail["snapshot_id"] == sid
    assert "Start a new export" in detail["error_description"]
    assert detail["restart"] == "POST /api/export/snapshots"

    # The expired pin was reaped: further reads report it as unknown.
    assert client.get(f"/export/snapshots/{sid}/pages").status_code == 404


def test_store_scoped_credential_cannot_pin_default_store(storage):
    _seed(storage, 1)
    client = TestClient(_make_app(storage, scope="read:other"))
    resp = client.post("/export/snapshots", json={"store": "default"})
    assert resp.status_code == 403


def test_release_is_idempotent(storage):
    _seed(storage, 1)
    client = TestClient(_make_app(storage))
    sid = client.post("/export/snapshots", json={}).json()["snapshot_id"]

    first = client.delete(f"/export/snapshots/{sid}")
    assert first.status_code == 200
    assert first.json() == {"snapshot_id": sid, "released": True}

    second = client.delete(f"/export/snapshots/{sid}")
    assert second.status_code == 200
    assert second.json() == {"snapshot_id": sid, "released": False}

    # Pages are gone immediately after release.
    assert client.get(f"/export/snapshots/{sid}/pages").status_code == 404


def test_backend_without_snapshot_support_returns_501():
    mock_storage = MagicMock()
    mock_storage.create_export_snapshot = AsyncMock(side_effect=NotImplementedError)
    client = TestClient(_make_app(mock_storage))

    resp = client.post("/export/snapshots", json={})
    assert resp.status_code == 501


def test_create_rejects_bad_tag_match(storage):
    client = TestClient(_make_app(storage))
    # pydantic Literal rejects it at the request model boundary.
    resp = client.post("/export/snapshots", json={"tag_match": "nope"})
    assert resp.status_code == 422
