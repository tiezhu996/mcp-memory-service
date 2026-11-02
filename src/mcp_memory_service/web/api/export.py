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

"""
Point-in-time export endpoints for offline backups.

An offline backup cannot use ordinary offset pages: background agents keep
writing while the backup walks the database, so later pages shift (new rows
show up, deleted rows vanish). The flow here is:

1. ``POST /api/export/snapshots`` pins one instant and returns an opaque
   ``snapshot_id`` plus the pinned row count.
2. ``GET /api/export/snapshots/{id}/pages?cursor=...`` returns stable keyset
   pages. Feed the previous page's ``next_cursor`` back in; a fixed cursor
   always returns the same rows, so a resumed backup neither skips nor repeats.
3. The pin expires (default one hour, ``MCP_EXPORT_SNAPSHOT_TTL_SECONDS``).
   Reading an expired pin returns **410 Gone** — the client must restart the
   export with a new snapshot instead of continuing silently.
4. ``DELETE`` releases a pin early.
"""

import logging
from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...storage.mixins import ExportSnapshotNotFound, ExportSnapshotExpired
from ..dependencies import get_storage
from ..oauth.middleware import get_current_user, AuthenticationResult
from .memories import MemoryResponse, memory_to_response
from .store_scope import resolve_store

router = APIRouter()
logger = logging.getLogger(__name__)


class ExportSnapshotCreateRequest(BaseModel):
    """Filters to apply when pinning the export set."""

    store: str = Field(
        "default",
        description="Store partition to pin (default: 'default'). Use 'all' for every store.",
    )
    tags: Optional[List[str]] = Field(None, description="Only pin memories matching these tags.")
    tag_match: Literal["any", "all"] = Field(
        "any", description="Match ANY tag (OR, default) or ALL tags (AND)."
    )
    memory_type: Optional[str] = Field(None, description="Only pin memories of this type.")


class ExportSnapshotMeta(BaseModel):
    """Pinned-point metadata echoed on create and on every page response."""

    snapshot_id: str
    pinned_at: float
    pinned_at_iso: str
    expires_at: float
    expires_at_iso: str
    ttl_seconds: int
    store: Optional[str]
    memory_type: Optional[str]
    tags: Optional[List[str]]
    tag_match: str
    total_rows: int


class ExportSnapshotResponse(ExportSnapshotMeta):
    """Result of pinning a new export snapshot; paging starts at cursor 0."""

    next_cursor: int
    has_more: bool


class ExportSnapshotPageResponse(BaseModel):
    """One stable page of a pinned export, plus the cursor to resume from."""

    snapshot: ExportSnapshotMeta
    memories: List[MemoryResponse]
    next_cursor: Optional[int] = None
    has_more: bool


class ExportSnapshotReleaseResponse(BaseModel):
    snapshot_id: str
    released: bool


def _restart_hint(snapshot_id: str) -> str:
    return (
        f"Export snapshot '{snapshot_id}' is no longer available. "
        "Start a new export with POST /api/export/snapshots and re-read from cursor 0."
    )


def _snapshot_iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshot_error(snapshot_id: str, exc: Exception) -> HTTPException:
    """Map storage-side snapshot failures onto explicit client statuses."""
    if isinstance(exc, ExportSnapshotExpired):
        logger.info(
            "Export snapshot %s expired at %s; client must restart the export",
            snapshot_id,
            exc.expires_at,
        )
        # 410 (not 404): the pin existed, but its fixed point has expired. The
        # message tells the client exactly what to do — restart, do not resume.
        return HTTPException(
            status_code=410,
            detail={
                "error": "export_snapshot_expired",
                "error_description": (
                    f"Export snapshot '{snapshot_id}' expired at {exc.expires_at} "
                    f"({_snapshot_iso(exc.expires_at)}); pinned pages are no longer "
                    "available. Start a new export with POST /api/export/snapshots "
                    "and re-read from cursor 0."
                ),
                "snapshot_id": snapshot_id,
                "pinned_at": exc.pinned_at,
                "expires_at": exc.expires_at,
                "restart": "POST /api/export/snapshots",
            },
        )
    # ExportSnapshotNotFound (or anything treated as it): never echo whether an
    # id merely never existed versus belonged to another caller.
    return HTTPException(
        status_code=404,
        detail={
            "error": "export_snapshot_not_found",
            "error_description": _restart_hint(snapshot_id),
        },
    )


def _meta_model(meta: dict) -> ExportSnapshotMeta:
    return ExportSnapshotMeta(**meta)


@router.post("/export/snapshots", response_model=ExportSnapshotResponse, tags=["export"])
async def create_export_snapshot(
    request: ExportSnapshotCreateRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(get_current_user),
):
    """Pin the current live memory set so a paginated backup reads one fixed point.

    The pinned set excludes rows written afterwards and keeps rows soft-deleted
    afterwards, so concurrent agent activity can no longer change the pages.
    """
    # The store arrives in the body, so scope is enforced in-handler just like
    # POST /memories rather than via the query-string store dependency.
    scope = resolve_store(request.store)
    user.require_store_scope("read", scope)

    try:
        snapshot = await storage.create_export_snapshot(
            store=scope,
            memory_type=request.memory_type,
            tags=request.tags,
            tag_match=request.tag_match,
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Point-in-time export is not supported by the configured storage backend.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportSnapshotResponse(**snapshot)


@router.get(
    "/export/snapshots/{snapshot_id}/pages",
    response_model=ExportSnapshotPageResponse,
    tags=["export"],
)
async def get_export_snapshot_page(
    snapshot_id: str,
    cursor: int = Query(0, ge=0, description="Resume cursor; pass next_cursor from the previous page (0 to start)."),
    limit: int = Query(1000, ge=1, le=5000, description="Maximum memories on the page."),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(get_current_user),
):
    """Read one page of a pinned export.

    Pages are deterministic for a given cursor: resuming at ``next_cursor``
    cannot skip or repeat rows as long as the snapshot stays alive.
    """
    try:
        meta = await storage.get_export_snapshot_metadata(snapshot_id)
    except (ExportSnapshotNotFound, ExportSnapshotExpired) as e:
        raise _snapshot_error(snapshot_id, e)

    # Scope is frozen on the pin itself — check it before returning any content.
    user.require_store_scope("read", meta.get("store"))

    try:
        page = await storage.get_export_snapshot_page(
            snapshot_id, after_id=cursor, limit=limit
        )
    except (ExportSnapshotNotFound, ExportSnapshotExpired) as e:
        raise _snapshot_error(snapshot_id, e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportSnapshotPageResponse(
        snapshot=_meta_model(page["snapshot"]),
        memories=[memory_to_response(memory) for memory in page["memories"]],
        next_cursor=page["next_cursor"],
        has_more=page["has_more"],
    )


@router.delete(
    "/export/snapshots/{snapshot_id}",
    response_model=ExportSnapshotReleaseResponse,
    tags=["export"],
)
async def release_export_snapshot(
    snapshot_id: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(get_current_user),
):
    """Release a pin before it expires. Idempotent: unknown ids return ``released=false``."""
    try:
        meta = await storage.get_export_snapshot_metadata(snapshot_id)
        user.require_store_scope("read", meta.get("store"))
    except ExportSnapshotExpired:
        # Expired pins are reaped lazily; the release still reports success.
        pass
    except ExportSnapshotNotFound:
        return ExportSnapshotReleaseResponse(snapshot_id=snapshot_id, released=False)

    try:
        released = await storage.release_export_snapshot(snapshot_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Point-in-time export is not supported by the configured storage backend.",
        )
    return ExportSnapshotReleaseResponse(snapshot_id=snapshot_id, released=released)
