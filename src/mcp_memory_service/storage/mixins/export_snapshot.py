"""ExportSnapshotMixin: point-in-time, resumable pagination for offline export.

Background agents keep writing (and soft-deleting) while an offline backup walks
the database. Offset pagination over the live ``memories`` table then shifts
between pages: a row inserted mid-export shows up on a later page, and a row
deleted mid-export vanishes from the backup.

A *snapshot export* fixes both problems:

1. ``create_export_snapshot()`` pins a point in time by copying the ``id`` of
   every row that is live at that moment into ``export_snapshot_rows``. The
   membership set is frozen — later inserts are not members, later soft-deletes
   leave the physical row in place, so every page reflects exactly the pinned
   set.
2. Pages use keyset pagination (``memory_id > after_id``) over the frozen
   membership instead of OFFSET, so resuming at the last returned cursor can
   neither skip nor repeat a row.
3. Snapshots carry an expiry. Reading an expired snapshot raises
   ``ExportSnapshotExpired`` (and reaps it); the caller is expected to start a
   new export rather than silently continue against a half-discarded pin.
"""

import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...compat import _sanitize_log_value
from .retrieve import _escape_like

logger = logging.getLogger(__name__)

# Pins are short-lived by design: they exist to make one backup read stable, not
# to archive history. Override with MCP_EXPORT_SNAPSHOT_TTL_SECONDS for slower
# links. Expired rows are pruned opportunistically whenever a new snapshot is
# created (and when an expired snapshot is read).
_DEFAULT_SNAPSHOT_TTL_SECONDS = 3600

# Match the cap the hashes listing endpoint enforces.
_MAX_PAGE_LIMIT = 5000


def _iso(ts: float) -> str:
    """Format a Unix timestamp the same way Memory timestamps are rendered."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshot_ttl_seconds() -> int:
    """Resolve the snapshot lifetime, floored at one second."""
    raw = os.getenv("MCP_EXPORT_SNAPSHOT_TTL_SECONDS")
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid MCP_EXPORT_SNAPSHOT_TTL_SECONDS=%s",
                _sanitize_log_value(raw),
            )
    return _DEFAULT_SNAPSHOT_TTL_SECONDS


class ExportSnapshotError(Exception):
    """Base class for snapshot export failures callers are expected to handle."""


class ExportSnapshotNotFound(ExportSnapshotError):
    """The snapshot id is unknown — never created, already released, or reaped.

    The only valid response is to start a new export from page one.
    """

    def __init__(self, snapshot_id: str):
        self.snapshot_id = snapshot_id
        super().__init__(f"export snapshot {snapshot_id!r} not found")


class ExportSnapshotExpired(ExportSnapshotError):
    """The snapshot passed its expiry and has been removed.

    The pinned read can no longer be continued; a fresh snapshot is required.
    """

    def __init__(self, snapshot_id: str, pinned_at: float, expires_at: float):
        self.snapshot_id = snapshot_id
        self.pinned_at = pinned_at
        self.expires_at = expires_at
        super().__init__(f"export snapshot {snapshot_id!r} expired at {expires_at}")


class ExportSnapshotMixin:
    """Point-in-time, keyset-paginated export for the SQLite backend."""

    def _ensure_export_snapshot_tables(self) -> None:
        """Create the snapshot registry and frozen membership table (idempotent)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS export_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                pinned_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                store TEXT,
                memory_type TEXT,
                tags TEXT,
                tag_match TEXT NOT NULL DEFAULT 'any',
                total_rows INTEGER NOT NULL
            )
            """
        )
        # The (snapshot_id, memory_id) PRIMARY KEY is also the keyset index for
        # paging: SQLite stores WITHOUT ROWID tables ordered by the key prefix.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS export_snapshot_rows (
                snapshot_id TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                PRIMARY KEY (snapshot_id, memory_id)
            ) WITHOUT ROWID
            """
        )

    def _prune_expired_snapshots(self, now: float) -> None:
        """Remove snapshots (and their membership rows) past their expiry."""
        expired_ids = [
            row[0]
            for row in self.conn.execute(
                "SELECT snapshot_id FROM export_snapshots WHERE expires_at <= ?",
                (now,),
            ).fetchall()
        ]
        for snapshot_id in expired_ids:
            self.conn.execute(
                "DELETE FROM export_snapshot_rows WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            self.conn.execute(
                "DELETE FROM export_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )

    def _get_active_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Return registry metadata for a live snapshot, reaping/raising otherwise."""
        row = self.conn.execute(
            """
            SELECT snapshot_id, pinned_at, expires_at, store, memory_type,
                   tags, tag_match, total_rows
            FROM export_snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise ExportSnapshotNotFound(snapshot_id)

        pinned_at, expires_at = row[1], row[2]
        if time.time() >= expires_at:
            # Reap eagerly so an expired pin stops occupying space immediately.
            self.conn.execute(
                "DELETE FROM export_snapshot_rows WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            self.conn.execute(
                "DELETE FROM export_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            self.conn.commit()
            raise ExportSnapshotExpired(snapshot_id, pinned_at, expires_at)

        tags = json.loads(row[5]) if row[5] else None
        return {
            "snapshot_id": row[0],
            "pinned_at": pinned_at,
            "pinned_at_iso": _iso(pinned_at),
            "expires_at": expires_at,
            "expires_at_iso": _iso(expires_at),
            "ttl_seconds": max(0, int(expires_at - pinned_at)),
            "store": row[3],
            "memory_type": row[4],
            "tags": tags,
            "tag_match": row[6],
            "total_rows": row[7],
        }

    async def create_export_snapshot(
        self,
        store: Optional[str] = "default",
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Pin the current live set so later pages read a stable point in time.

        Args:
            store: Store partition to pin. ``None`` pins every store, matching
                the ``store='all'`` sentinel of the HTTP surface.
            memory_type: Optional memory-type filter, same semantics as
                ``get_all_memories``.
            tags: Optional tag filter; rows must match any (or all, per
                ``tag_match``) of these tags.
            tag_match: ``"any"`` (OR, default) or ``"all"`` (AND).
            ttl_seconds: Optional lifetime override (tests / slow links); the
                configured ``MCP_EXPORT_SNAPSHOT_TTL_SECONDS`` applies otherwise.

        Returns:
            Snapshot metadata including the opaque ``snapshot_id`` and
            ``total_rows`` captured by the pin. Start paging with ``cursor=0``.
        """
        if tag_match not in ("any", "all"):
            raise ValueError("tag_match must be 'any' or 'all'")
        if ttl_seconds is not None and int(ttl_seconds) < 1:
            raise ValueError("ttl_seconds must be >= 1")

        await self.initialize()

        pinned_at = time.time()
        ttl = int(ttl_seconds) if ttl_seconds is not None else _snapshot_ttl_seconds()
        expires_at = pinned_at + ttl
        snapshot_id = secrets.token_hex(16)
        stripped_tags = [tag.strip() for tag in tags] if tags else None

        def _create():
            self._ensure_export_snapshot_tables()
            now = time.time()
            self._prune_expired_snapshots(now)

            where = ["deleted_at IS NULL"]
            params: List[Any] = []
            if store is not None:
                where.append("store = ?")
                params.append(store)
            if memory_type is not None:
                where.append("memory_type = ?")
                params.append(memory_type)
            if stripped_tags:
                joiner = " AND " if tag_match == "all" else " OR "
                tag_clause = joiner.join(
                    "(',' || tags || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags
                )
                where.append(f"({tag_clause})")
                params.extend(f"%,{_escape_like(tag)},%" for tag in stripped_tags)

            self.conn.execute(
                """
                INSERT INTO export_snapshots
                    (snapshot_id, pinned_at, expires_at, store, memory_type,
                     tags, tag_match, total_rows)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    snapshot_id,
                    pinned_at,
                    expires_at,
                    store,
                    memory_type,
                    json.dumps(stripped_tags) if stripped_tags else None,
                    tag_match,
                ),
            )

            # Freeze membership in a single statement: only ids of rows live at
            # pin time. AUTOINCREMENT ids are monotonic, so every later insert
            # sits outside this set and every later soft-delete leaves the row
            # physically joinable.
            self.conn.execute(
                f"""
                INSERT INTO export_snapshot_rows (snapshot_id, memory_id)
                SELECT ?, id FROM memories
                WHERE {' AND '.join(where)}
                """,
                (snapshot_id, *params),
            )
            total_rows = self.conn.execute(
                "SELECT COUNT(*) FROM export_snapshot_rows WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()[0]
            self.conn.execute(
                "UPDATE export_snapshots SET total_rows = ? WHERE snapshot_id = ?",
                (total_rows, snapshot_id),
            )
            self.conn.commit()
            return total_rows

        try:
            total_rows = await self._execute_with_retry(_create)
        except Exception as e:
            logger.error(
                "Failed to create export snapshot: %s", _sanitize_log_value(e)
            )
            raise

        logger.info(
            "Export snapshot %s pinned %d rows (store=%s, ttl=%ds)",
            _sanitize_log_value(snapshot_id),
            total_rows,
            _sanitize_log_value("all" if store is None else store),
            ttl,
        )
        return {
            "snapshot_id": snapshot_id,
            "pinned_at": pinned_at,
            "pinned_at_iso": _iso(pinned_at),
            "expires_at": expires_at,
            "expires_at_iso": _iso(expires_at),
            "ttl_seconds": ttl,
            "store": store,
            "memory_type": memory_type,
            "tags": stripped_tags,
            "tag_match": tag_match,
            "total_rows": total_rows,
            "next_cursor": 0,
            "has_more": total_rows > 0,
        }

    async def get_export_snapshot_metadata(self, snapshot_id: str) -> Dict[str, Any]:
        """Return metadata for a live snapshot; 404/410-equivalent otherwise."""
        await self.initialize()

        def _meta():
            self._ensure_export_snapshot_tables()
            return self._get_active_snapshot(snapshot_id)

        return await self._execute_with_retry(_meta)

    async def get_export_snapshot_page(
        self,
        snapshot_id: str,
        after_id: int = 0,
        limit: int = 1000,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Read one stable page of a pinned snapshot.

        Args:
            snapshot_id: Id returned by :meth:`create_export_snapshot`.
            after_id: Resume cursor — pass ``next_cursor`` from the previous
                page, or 0 to start. Keyset paging makes a given cursor return
                the same row every time: no row is skipped or repeated while
                walking the pin.
            limit: Maximum rows on the page (1..5000).
            include_embeddings: Hydrate embedding vectors on returned memories.

        Raises:
            ExportSnapshotNotFound: Unknown/released/reaped snapshot id.
            ExportSnapshotExpired: Pin expired (and was just reaped); the
                caller must start a new export.
        """
        if not isinstance(after_id, int) or after_id < 0:
            raise ValueError("after_id must be a non-negative integer")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be >= 1")
        limit = min(limit, _MAX_PAGE_LIMIT)

        await self.initialize()

        def _page():
            self._ensure_export_snapshot_tables()
            meta = self._get_active_snapshot(snapshot_id)

            embedding_col = ", e.content_embedding" if include_embeddings else ", NULL"
            embedding_join = (
                "LEFT JOIN memory_embeddings e ON m.id = e.rowid"
                if include_embeddings
                else ""
            )
            sql = f"""
                SELECT m.content_hash, m.content, m.tags, m.memory_type,
                       m.metadata, m.created_at, m.updated_at,
                       m.created_at_iso, m.updated_at_iso{embedding_col},
                       s.memory_id
                FROM export_snapshot_rows s
                JOIN memories m ON m.id = s.memory_id
                {embedding_join}
                WHERE s.snapshot_id = ? AND s.memory_id > ?
                ORDER BY s.memory_id ASC
                LIMIT ?
            """
            rows = self.conn.execute(sql, (snapshot_id, after_id, limit)).fetchall()
            return meta, rows

        try:
            meta, rows = await self._execute_with_retry(_page)
        except (ExportSnapshotNotFound, ExportSnapshotExpired):
            raise
        except Exception as e:
            logger.error(
                "Export snapshot %s page failed: %s",
                _sanitize_log_value(snapshot_id),
                _sanitize_log_value(e),
            )
            raise

        memories = []
        for row in rows:
            # _row_to_memory reads the first nine columns plus an optional
            # embedding blob at index nine; the cursor id trails at index ten.
            memory = self._row_to_memory(row[:10])
            if memory is not None:
                memories.append((row[10], memory))

        has_more = len(rows) == limit
        next_cursor = memories[-1][0] if has_more else None
        return {
            "snapshot": meta,
            "memories": [memory for _, memory in memories],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    async def release_export_snapshot(self, snapshot_id: str) -> bool:
        """Drop a snapshot and its membership rows before expiry.

        Returns ``True`` when something was removed, ``False`` for an unknown id
        (so an early release is idempotent).
        """
        await self.initialize()

        def _release():
            self._ensure_export_snapshot_tables()
            existed = self.conn.execute(
                "SELECT 1 FROM export_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone() is not None
            self.conn.execute(
                "DELETE FROM export_snapshot_rows WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            self.conn.execute(
                "DELETE FROM export_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            self.conn.commit()
            return existed

        return await self._execute_with_retry(_release)
