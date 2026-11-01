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

"""Tag alias registry — user-registered canonical-tag mappings.

Teams accumulate several spellings for the same tag (a project prefix, a bare
project name, different capitalization). The alias registry lets an operator
register those spellings as variants of one **canonical** tag. The mapping is
intentionally backend-independent: it is a small JSON file under the service
data directory (``BASE_DIR``), so it survives restarts AND storage-backend
switches (sqlite-vec -> Cloudflare -> Milvus -> hybrid).

The registry itself holds only the mapping. It is consumed by
:class:`mcp_memory_service.storage.alias_wrapper.AliasedStorage`, which applies
it to writes (new tags are stored under the canonical form), reads (queries
expand to every registered variant; old, never-rewritten rows are still found),
and statistics (variant counts merge under the canonical tag).

JSON shape::

    {
      "version": 1,
      "groups": [
        {"canonical": "atlas", "aliases": ["proj:atlas", "atlasproj"],
         "created_at": "...", "updated_at": "..."}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

REGISTRY_ENV_VAR = "MCP_TAG_ALIASES_PATH"
REGISTRY_FILENAME = "tag_aliases.json"
REGISTRY_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(tags: Union[str, Sequence[str], None]) -> List[str]:
    """Normalize tag spellings the same way the write path does.

    Imported lazily: ``services.memory_service`` imports the storage layer, and
    importing it at module load time would create a cycle.
    """
    from ..services.memory_service import normalize_tags

    return normalize_tags(tags)


@dataclass
class TagAliasGroup:
    """One canonical tag and every registered variant spelling."""

    canonical: str
    aliases: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "aliases": list(self.aliases),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def default_registry_path() -> Path:
    """Where the registry lives unless ``MCP_TAG_ALIASES_PATH`` says otherwise."""
    override = os.getenv(REGISTRY_ENV_VAR)
    if override and override.strip():
        return Path(override.strip()).expanduser()
    from ..config import BASE_DIR

    return Path(BASE_DIR) / REGISTRY_FILENAME


class TagAliasRegistry:
    """Canonical-tag mappings persisted as a backend-independent JSON file.

    All comparison keys are normalized with the same rules as stored tags
    (stripped, lowercased, deduplicated), so ``"Atlas"`` and ``"atlas"`` are
    the same registration key.

    Concurrency: in-process operations are serialized by a lock; cross-process
    edits (the MCP server and the HTTP server can run as separate processes)
    are picked up via mtime checks on every call.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path) if path is not None else default_registry_path()
        self._lock = threading.RLock()
        self._groups: List[TagAliasGroup] = []
        # alias -> canonical
        self._alias_map: Dict[str, str] = {}
        self._mtime: Optional[float] = None
        self._loaded = False
        # True when the on-disk file could not be parsed: we keep serving an
        # empty mapping but refuse to silently overwrite the user's bad file.
        self._corrupt_on_disk = False

    # ── persistence ────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def _reload_if_changed(self) -> None:
        try:
            stat_result = os.stat(self._path)
        except FileNotFoundError:
            if self._loaded and self._mtime is not None:
                # File was deleted out from under us (e.g. operator undo).
                self._set_state([], mtime=None)
            elif not self._loaded:
                self._set_state([], mtime=None)
            return
        except OSError as exc:
            logger.warning("Could not stat tag alias file %s: %s", self._path, exc)
            return
        if not self._loaded or self._mtime != stat_result.st_mtime_ns:
            self._load(stat_result.st_mtime_ns)

    def _load(self, mtime_ns: Optional[int]) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            logger.error(
                "Tag alias file %s is unreadable (%s); acting on an empty "
                "registry until it is repaired. The file is left untouched.",
                self._path, exc,
            )
            self._set_state([], mtime=mtime_ns, corrupt=True)
            return

        groups: List[TagAliasGroup] = []
        if isinstance(data, dict) and isinstance(data.get("groups"), list):
            for entry in data["groups"]:
                if not isinstance(entry, dict):
                    continue
                canonical = _clean(entry.get("canonical"))
                if len(canonical) != 1:
                    continue
                aliases = _clean(entry.get("aliases") or [])
                aliases = [a for a in aliases if a != canonical[0]]
                groups.append(TagAliasGroup(
                    canonical=canonical[0],
                    aliases=list(dict.fromkeys(aliases)),
                    created_at=entry.get("created_at"),
                    updated_at=entry.get("updated_at"),
                ))
        elif data:
            logger.error(
                "Tag alias file %s has an unrecognized shape; ignoring its "
                "contents and leaving the file untouched.", self._path,
            )
            self._set_state([], mtime=mtime_ns, corrupt=True)
            return

        self._set_state(self._dedupe_loaded_groups(groups), mtime=mtime_ns)

    @staticmethod
    def _dedupe_loaded_groups(groups: List[TagAliasGroup]) -> List[TagAliasGroup]:
        """Resolve duplicate/overlapping rows from a hand-edited file.

        First occurrence of a canonical wins; aliases claimed by two groups
        stay with the first group.
        """
        merged: List[TagAliasGroup] = []
        claimed: Dict[str, str] = {}
        for group in groups:
            existing = next((g for g in merged if g.canonical == group.canonical), None)
            if existing is None:
                existing = TagAliasGroup(
                    canonical=group.canonical,
                    created_at=group.created_at or _utc_now_iso(),
                    updated_at=group.updated_at,
                )
                merged.append(existing)
            for alias in group.aliases:
                if alias in claimed and claimed[alias] != existing.canonical:
                    logger.warning(
                        "Tag alias %r is registered under multiple canonical "
                        "tags in %s; keeping %r", alias, REGISTRY_FILENAME,
                        claimed[alias],
                    )
                    continue
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
                claimed[alias] = existing.canonical
        return merged

    def _set_state(
        self,
        groups: List[TagAliasGroup],
        mtime: Optional[int],
        corrupt: bool = False,
    ) -> None:
        self._groups = groups
        self._mtime = mtime
        self._corrupt_on_disk = corrupt
        self._rebuild_index()
        self._loaded = True

    def _rebuild_index(self) -> None:
        index: Dict[str, str] = {}
        for group in self._groups:
            for alias in group.aliases:
                index[alias] = group.canonical
        self._alias_map = index

    def _save(self) -> None:
        payload = {
            "version": REGISTRY_VERSION,
            "groups": [group.to_dict() for group in self._groups],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Never destroy an unparseable file on first write: park it beside the
        # new one so an operator can recover hand-edited data.
        if self._corrupt_on_disk and self._path.exists():
            backup = self._path.with_name(
                f"{self._path.name}.corrupt-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
            )
            try:
                os.replace(self._path, backup)
                logger.warning("Unparseable tag alias file backed up to %s", backup)
            except OSError as exc:
                raise OSError(
                    f"Could not back up unparseable tag alias file {self._path}: {exc}"
                ) from exc

        tmp_path = self._path.with_name(f".{self._path.name}.tmp-{os.getpid()}")
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            self._mtime = os.stat(self._path).st_mtime_ns
        except OSError:
            self._mtime = None
        self._corrupt_on_disk = False

    # ─ queries (read side) ─────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        with self._lock:
            self._reload_if_changed()
            return not self._alias_map

    def list_groups(self) -> List[Dict[str, Any]]:
        """Return registered groups as plain dicts (canonical first, then aliases)."""
        with self._lock:
            self._reload_if_changed()
            return [group.to_dict() for group in self._groups]

    def resolve(self, tag: str) -> str:
        """Canonical form of one tag; the tag itself when it is not an alias."""
        cleaned = _clean([tag])
        if not cleaned:
            return tag
        key = cleaned[0]
        with self._lock:
            self._reload_if_changed()
            return self._alias_map.get(key, key)

    def _group_for(self, canonical: str) -> Optional[TagAliasGroup]:
        return next((g for g in self._groups if g.canonical == canonical), None)

    def canonicalize(self, tags: Iterable[str]) -> List[str]:
        """Map a stored/returned tag list onto canonical forms (dedup, order kept)."""
        with self._lock:
            self._reload_if_changed()
            result: List[str] = []
            for tag in _clean(tags):
                canonical = self._alias_map.get(tag, tag)
                if canonical not in result:
                    result.append(canonical)
            return result

    def query_groups(self, tags: Sequence[str]) -> List[List[str]]:
        """Expand each queried tag into its variant group, query order, deduped.

        Every group lists the canonical tag first, then the registered aliases.
        ANY matching uses the flattened union; ALL matching intersects groups.
        """
        with self._lock:
            self._reload_if_changed()
            groups: List[List[str]] = []
            seen: set[str] = set()
            for tag in _clean(list(tags)):
                canonical = self._alias_map.get(tag, tag)
                if canonical in seen:
                    continue
                seen.add(canonical)
                group = self._group_for(canonical)
                variants = [canonical] + (list(group.aliases) if group else [])
                groups.append(variants)
            return groups

    def expand_flat(self, tags: Sequence[str]) -> List[str]:
        """Flat union of every variant for the queried tags (ANY semantics)."""
        flat: List[str] = []
        for variants in self.query_groups(tags):
            for variant in variants:
                if variant not in flat:
                    flat.append(variant)
        return flat

    def match_any(
        self, stored_tags: Sequence[str], query_tags: Sequence[str]
    ) -> bool:
        """Whether stored tags share a canonical group with any queried tag."""
        with self._lock:
            self._reload_if_changed()
            wanted = {self._alias_map.get(t, t) for t in _clean(list(query_tags))}
            if not wanted:
                return False
            have = {self._alias_map.get(t, t) for t in _clean(list(stored_tags))}
            return bool(wanted & have)

    def match_all(
        self, stored_tags: Sequence[str], query_tags: Sequence[str]
    ) -> bool:
        """Whether stored tags cover every queried tag's canonical group."""
        with self._lock:
            self._reload_if_changed()
            wanted = [
                self._alias_map.get(t, t) for t in _clean(list(query_tags))
            ]
            if not wanted:
                return False
            have = {self._alias_map.get(t, t) for t in _clean(list(stored_tags))}
            return all(canonical in have for canonical in dict.fromkeys(wanted))

    def merge_tag_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge ``[{"tag": str, "count": int}, ...]`` under canonical tags.

        Counts of every variant sum onto the canonical row; aliases are not
        listed separately. Rows without a usable tag/count are dropped.
        """
        with self._lock:
            self._reload_if_changed()
            totals: Dict[str, int] = {}
            order: List[str] = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                cleaned = _clean([row.get("tag")])
                if not cleaned:
                    continue
                canonical = self._alias_map.get(cleaned[0], cleaned[0])
                try:
                    count = int(row.get("count", 0))
                except (TypeError, ValueError):
                    continue
                if canonical not in totals:
                    totals[canonical] = 0
                    order.append(canonical)
                totals[canonical] += count
            return [
                {"tag": canonical, "count": totals[canonical]}
                for canonical in sorted(order, key=lambda name: (-totals[name], name))
            ]

    # ─ mutations (write side) ──────────────────────────────────────────────

    def register(
        self,
        canonical: Union[str, Sequence[str]],
        aliases: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        """Register variant spellings under one canonical tag.

        Idempotent: re-registering adds only new aliases. Registering an
        existing canonical tag as an alias **merges** the two groups (its
        aliases move under the new canonical tag). Raises ValueError on empty
        input or when an alias is already claimed by a different canonical tag.
        """
        with self._lock:
            self._reload_if_changed()

            canonical_clean = _clean([canonical] if isinstance(canonical, str) else list(canonical))
            if len(canonical_clean) != 1:
                raise ValueError("A single non-empty canonical tag is required")
            canonical_tag = canonical_clean[0]

            alias_tags = list(dict.fromkeys(_clean(aliases)))
            # The canonical tag is never also its own alias.
            alias_tags = [a for a in alias_tags if a != canonical_tag]
            if not alias_tags:
                raise ValueError("At least one non-empty alias is required")

            if canonical_tag in self._alias_map:
                owner = self._alias_map[canonical_tag]
                raise ValueError(
                    f"'{canonical_tag}' is already an alias of '{owner}'. "
                    f"Unregister it there before making it a canonical tag."
                )

            # Inputs naming an existing canonical tag merge that whole group
            # into this one. Resolve that set first so aliases carried by the
            # absorbed group are not mistaken for cross-group conflicts.
            merge_targets = {
                alias for alias in alias_tags
                if self._group_for(alias) is not None and alias != canonical_tag
            }
            merged_groups = [self._group_for(name) for name in sorted(merge_targets)]

            # A name already claimed as another group's alias cannot silently
            # move — unless that group is being absorbed in this same call.
            def _claimed_elsewhere(alias_name: str) -> bool:
                owner = self._alias_map.get(alias_name)
                if owner is None or owner == canonical_tag:
                    return False
                return owner not in merge_targets

            conflicts = sorted({
                alias for alias in alias_tags if _claimed_elsewhere(alias)
            })
            if conflicts:
                owners = ", ".join(
                    f"{name} -> {self._alias_map[name]}" for name in conflicts
                )
                raise ValueError(
                    f"Alias(es) already registered under another canonical tag: {owners}. "
                    "Unregister them there first."
                )

            now = _utc_now_iso()
            target = self._group_for(canonical_tag)
            if target is None:
                target = TagAliasGroup(
                    canonical=canonical_tag, aliases=[], created_at=now, updated_at=now
                )
                self._groups.append(target)

            # Union the explicit aliases with every member of absorbed groups.
            combined: List[str] = list(alias_tags)
            absorbed: List[str] = []
            for source in merged_groups:
                absorbed.append(source.canonical)
                for member in [source.canonical, *source.aliases]:
                    if member != canonical_tag and member not in combined:
                        combined.append(member)
                self._groups.remove(source)

            added: List[str] = []
            for alias in combined:
                if alias == canonical_tag:
                    continue
                if alias not in target.aliases:
                    target.aliases.append(alias)
                    added.append(alias)

            target.updated_at = now
            self._rebuild_index()
            self._save()
            logger.info(
                "Tag alias registration: canonical=%r added=%r merged_groups=%r",
                canonical_tag, added, absorbed,
            )
            return {
                "canonical": canonical_tag,
                "aliases": list(target.aliases),
                "added": added,
                "merged_canonical_tags": absorbed,
                "updated_at": now,
            }

    def unregister(
        self,
        canonical: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Undo a registration.

        - ``alias`` only: remove that one variant (its group disappears when no
          alias remains).
        - ``canonical``: remove the whole group (canonical tag + all aliases).
        Exactly one of the two must be given.
        """
        if bool(canonical) == bool(alias):
            raise ValueError("Provide exactly one of canonical= or alias=")

        with self._lock:
            self._reload_if_changed()
            now = _utc_now_iso()

            if canonical is not None:
                cleaned = _clean([canonical])
                if len(cleaned) != 1:
                    raise ValueError("A non-empty canonical tag is required")
                canonical_tag = cleaned[0]
                group = self._group_for(canonical_tag)
                if group is None and canonical_tag in self._alias_map:
                    # Given the alias spelling instead: still unambiguous undo.
                    group = self._group_for(self._alias_map[canonical_tag])
                    canonical_tag = group.canonical if group else canonical_tag
                if group is None:
                    raise ValueError(f"No registration found for canonical tag '{canonical}'")
                removed_aliases = list(group.aliases)
                self._groups.remove(group)
                self._rebuild_index()
                self._save()
                logger.info("Tag alias group removed: canonical=%r", canonical_tag)
                return {
                    "removed": "group",
                    "canonical": canonical_tag,
                    "aliases": removed_aliases,
                    "updated_at": now,
                }

            cleaned = _clean([alias])
            if len(cleaned) != 1:
                raise ValueError("A non-empty alias is required")
            alias_tag = cleaned[0]
            canonical_tag = self._alias_map.get(alias_tag)
            if canonical_tag is None:
                raise ValueError(f"'{alias}' is not a registered alias")
            group = self._group_for(canonical_tag)
            if group is None or alias_tag not in group.aliases:  # pragma: no cover
                raise ValueError(f"'{alias}' is not a registered alias")
            group.aliases.remove(alias_tag)
            removed_group = False
            if not group.aliases:
                self._groups.remove(group)
                removed_group = True
            else:
                group.updated_at = now
            self._rebuild_index()
            self._save()
            logger.info(
                "Tag alias removed: alias=%r canonical=%r group_removed=%r",
                alias_tag, canonical_tag, removed_group,
            )
            return {
                "removed": "group" if removed_group else "alias",
                "canonical": canonical_tag,
                "alias": alias_tag,
                "aliases": [] if removed_group else list(group.aliases),
                "updated_at": now,
            }


# ── process-wide singleton ───────────────────────────────────────────────────

_REGISTRIES: Dict[str, TagAliasRegistry] = {}
_REGISTRIES_LOCK = threading.Lock()


def get_registry(path: Optional[Union[str, Path]] = None) -> TagAliasRegistry:
    """Shared registry for a file path (defaults to the configured file)."""
    resolved = Path(path) if path is not None else default_registry_path()
    key = str(resolved)
    with _REGISTRIES_LOCK:
        registry = _REGISTRIES.get(key)
        if registry is None:
            registry = TagAliasRegistry(resolved)
            _REGISTRIES[key] = registry
        return registry


def reset_registry_cache() -> None:
    """Forget cached singletons (tests)."""
    with _REGISTRIES_LOCK:
        _REGISTRIES.clear()
