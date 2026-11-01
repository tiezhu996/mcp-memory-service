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

"""Transparent storage wrapper that applies registered tag aliases.

``AliasedStorage`` wraps any concrete :class:`MemoryStorage` backend so that
every user-facing path — MCP tools, REST API and CLI alike — sees one
canonical tag vocabulary:

* **writes** store tags under their canonical form;
* **queries** expand to every registered variant so legacy rows (never
  physically rewritten) are still found, with correct ANY/ALL semantics;
* **returned rows** expose the canonical vocabulary as a view (physical rows
  are not mutated), which also unifies every Counter-based statistic;
* **tag listings/counts** merge variant rows onto the canonical tag;
* **deletes** match the same set a query would return.

Delta-sync apply materializes remote events through direct SQL
(``storage/sync/apply.py``) and never crosses this wrapper, so remote tag
spellings are preserved verbatim.

With an empty registry every method is a straight pass-through: no copies, no
reordering, no behavioral change.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import MemoryStorage, recreate_kwargs
from .tag_aliases import TagAliasRegistry, get_registry

logger = logging.getLogger(__name__)


def _dedupe_memories(memories: List[Any]) -> List[Any]:
    """First occurrence wins by content_hash; rows without a hash all survive."""
    seen: set = set()
    result: List[Any] = []
    for memory in memories:
        key = getattr(memory, "content_hash", None)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        result.append(memory)
    return result


class AliasedStorage(MemoryStorage):
    """Backend-agnostic tag-alias layer around a concrete storage backend."""

    def __init__(self, inner: MemoryStorage, registry: Optional[TagAliasRegistry] = None):
        # Bypass __setattr__/delegation: these two are the wrapper's own state.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_registry", registry or get_registry())

    # ── plumbing ─────────────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        # Only called for attributes not found on the wrapper itself.
        return getattr(object.__getattribute__(self, "_inner"), name)

    @property
    def inner_storage(self) -> MemoryStorage:
        return self._inner

    @property
    def tag_alias_registry(self) -> TagAliasRegistry:
        return self._registry

    @property
    def max_content_length(self):
        return self._inner.max_content_length

    @property
    def supports_chunking(self) -> bool:
        return self._inner.supports_chunking

    # Capability flags and the store-scoping flag are declared as defaults on
    # MemoryStorage, so inherited lookups would shadow the inner backend's
    # overrides (e.g. Cloudflare sets supports_delete_operations=True). Always
    # report the concrete backend's answer.
    @property
    def supports_capacity_monitoring(self) -> bool:
        return getattr(self._inner, "supports_capacity_monitoring", False)

    @property
    def supports_recreate_refusal(self) -> bool:
        return getattr(self._inner, "supports_recreate_refusal", False)

    @property
    def requires_metadata_normalization(self) -> bool:
        return getattr(self._inner, "requires_metadata_normalization", False)

    @property
    def supports_delete_operations(self) -> bool:
        return getattr(self._inner, "supports_delete_operations", False)

    @property
    def get_by_hash_is_store_scoped(self) -> bool:
        return getattr(self._inner, "get_by_hash_is_store_scoped", False)

    async def initialize(self) -> None:
        return await self._inner.initialize()

    # ── view helpers ───────────────────────────────────────────────────────

    def _canon_memory(self, memory: Any) -> Any:
        """Return a view of a Memory row with canonical tags.

        Shallow copy only; the physical row and other fields are untouched.
        With no registrations the original object comes back unchanged.
        """
        if memory is None or self._registry.is_empty:
            return memory
        tags = getattr(memory, "tags", None) or []
        canonical = self._registry.canonicalize(tags)
        view = copy.copy(memory)
        view.tags = canonical
        return view

    def _canon_memories(self, memories: Sequence[Any]) -> List[Any]:
        if self._registry.is_empty:
            return list(memories)
        return _dedupe_memories([self._canon_memory(m) for m in memories])

    def _canon_query_result(self, result: Any) -> Any:
        if result is None or self._registry.is_empty:
            return result
        view = copy.copy(result)
        view.memory = self._canon_memory(result.memory)
        return view

    def _canon_query_results(self, results: Sequence[Any]) -> List[Any]:
        if self._registry.is_empty:
            return list(results)
        return [self._canon_query_result(r) for r in results]

    def _canon_dict_memories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Canonicalize ``tags`` on dict-shaped search_memories() results."""
        if not self._registry.is_empty and isinstance(payload, dict):
            for memory in payload.get("memories") or []:
                if isinstance(memory, dict):
                    memory["tags"] = self._registry.canonicalize(memory.get("tags") or [])
        return payload

    def _variants_flat(self, tags: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not tags:
            return list(tags) if tags else []
        return self._registry.expand_flat(list(tags))

    def _aliases_involved(self, tags: Optional[Sequence[str]]) -> bool:
        """Whether expanding ``tags`` adds any variant beyond the query itself."""
        if not tags or self._registry.is_empty:
            return False
        return any(len(group) > 1 for group in self._registry.query_groups(list(tags)))

    # ── writes ──────────────────────────────────────────────────────────────

    async def store(
        self,
        memory: Any,
        skip_semantic_dedup: bool = False,
        store: Optional[str] = "default",
        recreate: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        if not self._registry.is_empty:
            memory = self._canon_memory(memory)
        return await self._inner.store(
            memory,
            skip_semantic_dedup=skip_semantic_dedup,
            store=store,
            **recreate_kwargs(self._inner, recreate),
        )

    async def store_batch(self, memories: List[Any], recreate: Optional[bool] = None) -> List[Tuple[bool, str]]:
        if not self._registry.is_empty:
            memories = [self._canon_memory(m) for m in memories]
        return await self._inner.store_batch(memories, **recreate_kwargs(self._inner, recreate))

    async def update_memory_metadata(
        self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True
    ) -> Tuple[bool, str]:
        if not self._registry.is_empty and updates and "tags" in updates and updates["tags"] is not None:
            updates = dict(updates)
            updates["tags"] = self._registry.canonicalize(updates["tags"])
        return await self._inner.update_memory_metadata(
            content_hash, updates, preserve_timestamps=preserve_timestamps
        )

    async def update_memory(self, memory: Any) -> bool:
        if not self._registry.is_empty:
            memory = self._canon_memory(memory)
        return await self._inner.update_memory(memory)

    async def update_memory_versioned(self, *args, **kwargs):
        if not self._registry.is_empty:
            if "new_tags" in kwargs and kwargs["new_tags"] is not None:
                kwargs["new_tags"] = self._registry.canonicalize(kwargs["new_tags"])
            elif len(args) >= 3 and args[2] is not None:
                args = (*args[:2], self._registry.canonicalize(args[2]), *args[3:])
        return await self._inner.update_memory_versioned(*args, **kwargs)

    # ── semantic retrieval ─────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        include_superseded: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        store: Optional[str] = None,
    ) -> List[Any]:
        if tags and not self._registry.is_empty:
            tags = self._registry.expand_flat(tags)
        results = await self._inner.retrieve(
            query,
            n_results=n_results,
            tags=tags,
            min_confidence=min_confidence,
            include_superseded=include_superseded,
            start_time=start_time,
            end_time=end_time,
            store=store,
        )
        return self._canon_query_results(results)

    async def recall(
        self,
        query: Optional[str] = None,
        n_results: int = 5,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        store: Optional[str] = None,
    ) -> List[Any]:
        results = await self._inner.recall(
            query=query,
            n_results=n_results,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            store=store,
        )
        return self._canon_query_results(results)

    # ── tag search ──────────────────────────────────────────────────────────

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Any]:
        variants = self._variants_flat(tags) if not self._registry.is_empty else list(tags or [])
        rows = await self._inner.search_by_tag(variants, time_start=time_start)
        return self._canon_memories(rows)

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        include_pending: bool = False,
    ) -> List[Any]:
        normalized_op = operation.strip().upper() if isinstance(operation, str) else "AND"
        if normalized_op not in {"AND", "OR"}:
            normalized_op = "AND"

        if self._registry.is_empty or not tags:
            rows = await self._inner.search_by_tags(
                tags,
                operation=normalized_op,
                time_start=time_start,
                time_end=time_end,
                include_pending=include_pending,
            )
            return list(rows)

        groups = self._registry.query_groups(list(tags))
        if normalized_op == "OR" or len(groups) <= 1:
            # OR: union of every variant. AND over a single tag: every variant
            # of that one group is interchangeable.
            variants: List[str] = []
            for group in groups:
                for variant in group:
                    if variant not in variants:
                        variants.append(variant)
            rows = await self._inner.search_by_tags(
                variants,
                operation="OR",
                time_start=time_start,
                time_end=time_end,
                include_pending=include_pending,
            )
            return self._canon_memories(rows)

        # AND over several groups: intersect per-group OR result sets.
        per_group_rows: List[List[Any]] = []
        for group in groups:
            rows = await self._inner.search_by_tags(
                group,
                operation="OR",
                time_start=time_start,
                time_end=time_end,
                include_pending=include_pending,
            )
            per_group_rows.append(rows)

        if not per_group_rows or any(not rows for rows in per_group_rows):
            return []
        intersection_hashes = set.intersection(*(
            {getattr(m, "content_hash", None) for m in rows}
            for rows in per_group_rows
        ))
        merged: List[Any] = []
        seen: set = set()
        for rows in per_group_rows:
            for memory in rows:
                key = getattr(memory, "content_hash", None)
                if key in seen or key not in intersection_hashes:
                    continue
                seen.add(key)
                merged.append(memory)
        return self._canon_memories(merged)

    async def search_by_tag_chronological(self, tags: List[str], limit: int = None, offset: int = 0) -> List[Any]:
        variants = self._variants_flat(tags) if not self._registry.is_empty else list(tags or [])
        rows = await self._inner.search_by_tag_chronological(variants, limit=limit, offset=offset)
        return self._canon_memories(rows)

    async def search_memories(self, **kwargs) -> Dict[str, Any]:
        tags = kwargs.get("tags")
        aliased = bool(tags) and self._aliases_involved(tags)
        if aliased:
            original_tags = list(tags)
            original_match = kwargs.get("tag_match", "any")
            variants = self._registry.expand_flat(original_tags)
            kwargs = dict(kwargs)
            kwargs["tags"] = variants
            # The inner backend filters against raw stored tags: ask for the
            # ANY-superset and re-apply canonical ANY/ALL semantics here.
            kwargs["tag_match"] = "any"
            original_limit = kwargs.get("limit", 10)
            overfetch = min(10, max(1, len(variants) // max(1, len(original_tags))))
            if overfetch > 1 and isinstance(original_limit, int) and original_limit > 0:
                kwargs["limit"] = original_limit * overfetch
            payload = await self._inner.search_memories(**kwargs)
            if isinstance(payload, dict) and payload.get("memories"):
                if original_match == "all":
                    payload["memories"] = [
                        memory for memory in payload["memories"]
                        if self._registry.match_all(memory.get("tags") or [], original_tags)
                    ]
                else:
                    payload["memories"] = [
                        memory for memory in payload["memories"]
                        if self._registry.match_any(memory.get("tags") or [], original_tags)
                    ]
                if isinstance(original_limit, int):
                    payload["memories"] = payload["memories"][:original_limit]
                payload["total"] = len(payload["memories"])
            return self._canon_dict_memories(payload)

        payload = await self._inner.search_memories(**kwargs)
        return self._canon_dict_memories(payload)

    # ── listing / counting ──────────────────────────────────────────────────

    async def get_all_memories(
        self,
        limit: int = None,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        stale_days: Optional[int] = None,
        include_embeddings: bool = False,
        store: Optional[str] = "default",
        agent_id: Optional[str] = None,
        exclude_pending: bool = False,
    ) -> List[Any]:
        if not tags or not self._aliases_involved(tags):
            rows = await self._inner.get_all_memories(
                limit=limit,
                offset=offset,
                memory_type=memory_type,
                tags=tags,
                tag_match=tag_match,
                stale_days=stale_days,
                include_embeddings=include_embeddings,
                store=store,
                agent_id=agent_id,
                exclude_pending=exclude_pending,
            )
            return self._canon_memories(rows)

        variants = self._registry.expand_flat(list(tags))
        if tag_match == "all":
            # Fetch the ANY superset without pagination, then keep rows that
            # cover every queried canonical group. Alias+ALL is the rare case;
            # the tag-matched set bounds the work.
            superset = await self._inner.get_all_memories(
                limit=None,
                offset=0,
                memory_type=memory_type,
                tags=variants,
                tag_match="any",
                stale_days=stale_days,
                include_embeddings=include_embeddings,
                store=store,
                agent_id=agent_id,
                exclude_pending=exclude_pending,
            )
            rows = [
                memory for memory in superset
                if self._registry.match_all(getattr(memory, "tags", None) or [], list(tags))
            ]
            if offset:
                rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]
            return self._canon_memories(rows)

        rows = await self._inner.get_all_memories(
            limit=limit,
            offset=offset,
            memory_type=memory_type,
            tags=variants,
            tag_match="any",
            stale_days=stale_days,
            include_embeddings=include_embeddings,
            store=store,
            agent_id=agent_id,
            exclude_pending=exclude_pending,
        )
        return self._canon_memories(rows)

    async def count_all_memories(
        self,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        stale_days: Optional[int] = None,
        store: Optional[str] = "default",
        agent_id: Optional[str] = None,
    ) -> int:
        if not tags or not self._aliases_involved(tags):
            return await self._inner.count_all_memories(
                memory_type=memory_type,
                tags=tags,
                tag_match=tag_match,
                stale_days=stale_days,
                store=store,
                agent_id=agent_id,
            )
        if tag_match == "all":
            rows = await self.get_all_memories(
                memory_type=memory_type,
                tags=tags,
                tag_match="all",
                stale_days=stale_days,
                store=store,
                agent_id=agent_id,
            )
            return len(rows)
        return await self._inner.count_all_memories(
            memory_type=memory_type,
            tags=self._registry.expand_flat(list(tags)),
            tag_match="any",
            stale_days=stale_days,
            store=store,
            agent_id=agent_id,
        )

    async def count_memories_by_tag(self, tags: List[str]) -> int:
        if not tags or self._registry.is_empty:
            return await self._inner.count_memories_by_tag(tags)
        return await self._inner.count_memories_by_tag(self._registry.expand_flat(list(tags)))

    async def get_all_tags(self) -> List[str]:
        tags = await self._inner.get_all_tags()
        if self._registry.is_empty:
            return tags
        return self._registry.canonicalize(tags)

    async def get_all_tags_with_counts(self, store: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = await self._inner.get_all_tags_with_counts(store=store)
        if self._registry.is_empty:
            return rows
        return self._registry.merge_tag_rows(rows)

    # ── direct lookups / listings (view canonicalization only) ──────────────

    async def get_by_hash(self, content_hash: str, store: Optional[str] = None) -> Optional[Any]:
        memory = await self._inner.get_by_hash(content_hash, store=store)
        return self._canon_memory(memory)

    async def get_by_exact_content(self, content: str, store: Optional[str] = None) -> List[Any]:
        rows = await self._inner.get_by_exact_content(content, store=store)
        return self._canon_memories(rows)

    async def get_recent_memories(self, n: int = 10) -> List[Any]:
        rows = await self._inner.get_recent_memories(n)
        return self._canon_memories(rows)

    async def get_largest_memories(self, n: int = 10) -> List[Any]:
        rows = await self._inner.get_largest_memories(n)
        return self._canon_memories(rows)

    async def get_memories_by_time_range(
        self,
        start_time: float,
        end_time: float,
        include_embeddings: bool = False,
        store: Optional[str] = None,
    ) -> List[Any]:
        rows = await self._inner.get_memories_by_time_range(
            start_time,
            end_time,
            include_embeddings=include_embeddings,
            store=store,
        )
        return self._canon_memories(rows)

    # ── deletes (same matching set as queries) ─────────────────────────────

    async def delete(self, content_hash: str, store: Optional[str] = None) -> Tuple[bool, str]:
        return await self._inner.delete(content_hash, store=store)

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        return await self._inner.cleanup_duplicates()

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        if self._registry.is_empty:
            return await self._inner.delete_by_tag(tag)
        variants = self._registry.expand_flat([tag])
        if len(variants) <= 1:
            return await self._inner.delete_by_tag(variants[0] if variants else tag)
        result = await self._inner.delete_by_tags(variants)
        # Backends return either (count, message) or (count, message, hashes).
        return result[0], result[1]

    async def delete_by_tags(self, tags: List[str], store: Optional[str] = None):
        if tags and not self._registry.is_empty:
            tags = self._registry.expand_flat(list(tags))
        return await self._inner.delete_by_tags(tags, store=store)

    async def delete_memories(
        self,
        content_hash: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        before: Optional[str] = None,
        after: Optional[str] = None,
        dry_run: bool = False,
        store: Optional[str] = "default",
    ) -> Dict[str, Any]:
        if content_hash or not tags or not self._aliases_involved(tags):
            return await self._inner.delete_memories(
                content_hash=content_hash,
                tags=tags,
                tag_match=tag_match,
                before=before,
                after=after,
                dry_run=dry_run,
                store=store,
            )

        # Alias-tagged path: resolve the exact memory set through the wrapped
        # read side (canonical ANY/ALL), apply date filters, delete by hash.
        candidates = await self.get_all_memories(
            tags=tags,
            tag_match=tag_match,
            store=store,
        )
        filtered = self._filter_by_dates(candidates, before, after)

        hashes = [getattr(m, "content_hash", None) for m in filtered]
        hashes = [h for h in hashes if h]
        if dry_run:
            return {
                "success": True,
                "deleted_count": len(hashes),
                "deleted_hashes": hashes,
                "dry_run": True,
                "message": f"Would delete {len(hashes)} memories",
            }

        deleted_hashes: List[str] = []
        errors: List[str] = []
        for content_hash_value in hashes:
            try:
                success, message = await self.delete(content_hash_value, store=store)
                if success:
                    deleted_hashes.append(content_hash_value)
                else:
                    errors.append(message)
            except Exception as exc:  # noqa: BLE001 - reported in the result envelope
                errors.append(str(exc))

        if errors:
            return {
                "success": bool(deleted_hashes),
                "deleted_count": len(deleted_hashes),
                "deleted_hashes": deleted_hashes,
                "dry_run": False,
                "message": f"Deleted {len(deleted_hashes)} memories with {len(errors)} failures",
                "error": "; ".join(errors[:3]),
            }
        return {
            "success": True,
            "deleted_count": len(deleted_hashes),
            "deleted_hashes": deleted_hashes,
            "dry_run": False,
            "message": f"Successfully deleted {len(deleted_hashes)} memories",
        }

    @staticmethod
    def _filter_by_dates(memories: Sequence[Any], before: Optional[str], after: Optional[str]) -> List[Any]:
        if not before and not after:
            return list(memories)
        before_date = datetime.fromisoformat(before).date() if before else None
        after_date = datetime.fromisoformat(after).date() if after else None
        result: List[Any] = []
        for memory in memories:
            iso_value = getattr(memory, "created_at_iso", None)
            if not iso_value:
                continue
            try:
                memory_date = datetime.fromisoformat(str(iso_value).replace("Z", "+00:00")).date()
            except ValueError:
                continue
            if before_date and memory_date >= before_date:
                continue
            if after_date and memory_date <= after_date:
                continue
            result.append(memory)
        return result


def _install_delegators() -> None:
    """Forward every base-contract method the wrapper does not intercept.

    MemoryStorage ships concrete default implementations (get_stats,
    health_probe, conflict APIs, time-range deletes, graph/access data, ...).
    Without explicit forwarding the wrapper would inherit those defaults and
    silently shadow the concrete backend's overrides, so delegate every public
    member of the contract that is not intentionally reimplemented above.
    Tag-bearing methods reimplemented above keep the alias semantics; delegated
    methods that call ``self.retrieve``/``self.search_*`` still pass through
    those intercepts.
    """
    import inspect

    for name, member in vars(MemoryStorage).items():
        if name.startswith("_") or name in AliasedStorage.__dict__:
            continue
        if isinstance(member, property):
            # Properties are handled explicitly (capability flags etc.).
            continue
        if not inspect.isfunction(member):
            continue

        if inspect.iscoroutinefunction(member):
            def _make_async(method_name: str):
                async def _delegator(self, *args, **kwargs):
                    return await getattr(self._inner, method_name)(*args, **kwargs)
                _delegator.__name__ = method_name
                return _delegator
            setattr(AliasedStorage, name, _make_async(name))
        else:
            def _make_sync(method_name: str):
                def _delegator(self, *args, **kwargs):
                    return getattr(self._inner, method_name)(*args, **kwargs)
                _delegator.__name__ = method_name
                return _delegator
            setattr(AliasedStorage, name, _make_sync(name))


_install_delegators()


def wrap_with_tag_aliases(
    storage: MemoryStorage,
    registry: Optional[TagAliasRegistry] = None,
) -> MemoryStorage:
    """Wrap a backend with the alias layer; idempotent (never double-wrap)."""
    if isinstance(storage, AliasedStorage):
        return storage
    return AliasedStorage(storage, registry=registry)
