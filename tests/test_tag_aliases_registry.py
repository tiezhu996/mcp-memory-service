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

"""Unit tests for the backend-independent tag alias registry."""

import json

import pytest

from mcp_memory_service.storage.tag_aliases import (
    TagAliasRegistry,
    get_registry,
    reset_registry_cache,
)


@pytest.fixture
def registry(tmp_path):
    return TagAliasRegistry(tmp_path / "tag_aliases.json")


# ── register / resolve ───────────────────────────────────────────────────────

def test_register_normalizes_spelling_and_resolves(registry):
    registry.register("Atlas", ["Proj:Atlas", " atlasproj ", "ATLASPROJ"])

    # Names are trimmed/lowercased; duplicate spellings collapse.
    assert registry.resolve("proj:atlas") == "atlas"
    assert registry.resolve("ATLASPROJ") == "atlas"
    # Unmapped tags resolve to themselves.
    assert registry.resolve("other") == "other"

    groups = {g["canonical"]: g["aliases"] for g in registry.list_groups()}
    assert groups == {"atlas": ["proj:atlas", "atlasproj"]}


def test_register_is_idempotent_and_additive(registry):
    registry.register("atlas", ["proj:atlas"])
    result = registry.register("atlas", ["proj:atlas", "atlasproj"])

    assert result["added"] == ["atlasproj"]
    assert registry.list_groups()[0]["aliases"] == ["proj:atlas", "atlasproj"]


def test_canonical_cannot_be_a_foreign_alias(registry):
    registry.register("atlas", ["proj:atlas"])
    with pytest.raises(ValueError, match="alias of 'atlas'"):
        registry.register("proj:atlas", ["something"])


def test_alias_claimed_by_other_group_is_rejected(registry):
    registry.register("atlas", ["proj:atlas"])
    with pytest.raises(ValueError, match="already registered"):
        registry.register("borealis", ["proj:atlas"])


def test_alias_equal_to_canonical_is_dropped(registry):
    result = registry.register("atlas", ["atlas", "proj:atlas"])
    assert result["aliases"] == ["proj:atlas"]


def test_register_rejects_empty_inputs(registry):
    with pytest.raises(ValueError):
        registry.register("", ["a"])
    with pytest.raises(ValueError):
        registry.register("atlas", [])
    with pytest.raises(ValueError):
        registry.register("atlas", ["", " "])


def test_registering_existing_canonical_as_alias_merges_groups(registry):
    registry.register("oldname", ["old-a", "old-b"])
    result = registry.register("atlas", ["oldname", "new-a"])

    assert result["merged_canonical_tags"] == ["oldname"]
    groups = {g["canonical"]: g["aliases"] for g in registry.list_groups()}
    assert groups == {"atlas": ["oldname", "new-a", "old-a", "old-b"]}
    # Every former spelling now resolves to the surviving canonical tag.
    assert registry.resolve("old-a") == "atlas"
    assert registry.resolve("oldname") == "atlas"


# ── query expansion / matching ───────────────────────────────────────────────

def test_query_groups_and_flat_expansion(registry):
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    registry.register("borealis", ["proj:borealis"])

    groups = registry.query_groups(["atlasproj", "atlas", "borealis"])
    # Deduped on canonical tag, canonical spelling first in each group.
    assert groups == [
        ["atlas", "proj:atlas", "atlasproj"],
        ["borealis", "proj:borealis"],
    ]
    assert registry.expand_flat(["atlasproj"]) == ["atlas", "proj:atlas", "atlasproj"]


def test_match_any_and_all_against_legacy_spellings(registry):
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    registry.register("borealis", ["proj:borealis"])

    # Old rows keep their raw, possibly mixed-case tags: matching canonicalizes
    # both sides and is case-insensitive.
    legacy_row_tags = ["Proj:Atlas", "backend"]
    assert registry.match_any(legacy_row_tags, ["atlas"])
    assert registry.match_any(legacy_row_tags, ["atlasproj"])
    assert not registry.match_any(legacy_row_tags, ["borealis"])

    both_rows = ["atlasproj", "Proj:Borealis"]
    assert registry.match_all(both_rows, ["atlas", "borealis"])
    assert not registry.match_all(both_rows, ["atlas", "missing"])


def test_canonicalize_dedupes_preserving_order(registry):
    registry.register("atlas", ["proj:atlas"])
    assert registry.canonicalize(["Proj:Atlas", "atlas", "x", "X"]) == ["atlas", "x"]


def test_merge_tag_rows_sums_variants(registry):
    registry.register("atlas", ["proj:atlas", "atlasproj"])
    rows = [
        {"tag": "proj:atlas", "count": 3},
        {"tag": "atlas", "count": 2},
        {"tag": "atlasproj", "count": 1},
        {"tag": "backend", "count": 9},
    ]
    merged = {row["tag"]: row["count"] for row in registry.merge_tag_rows(rows)}
    assert merged == {"backend": 9, "atlas": 6}


# ── undo ─────────────────────────────────────────────────────────────────────

def test_unregister_single_alias_then_whole_group(registry):
    registry.register("atlas", ["proj:atlas", "atlasproj"])

    result = registry.unregister(alias="proj:atlas")
    assert result["removed"] == "alias"
    assert registry.resolve("proj:atlas") == "proj:atlas"
    assert registry.resolve("atlasproj") == "atlas"

    result = registry.unregister(canonical="atlas")
    assert result["removed"] == "group"
    assert registry.resolve("atlasproj") == "atlasproj"
    assert registry.list_groups() == []


def test_removing_last_alias_drops_group(registry):
    registry.register("atlas", ["proj:atlas"])
    result = registry.unregister(alias="proj:atlas")
    assert result["removed"] == "group"
    assert registry.list_groups() == []


def test_unregister_validates_arguments(registry):
    registry.register("atlas", ["proj:atlas"])
    with pytest.raises(ValueError):
        registry.unregister()
    with pytest.raises(ValueError):
        registry.unregister(canonical="atlas", alias="proj:atlas")
    with pytest.raises(ValueError):
        registry.unregister(alias="unknown")
    with pytest.raises(ValueError):
        registry.unregister(canonical="unknown")


# ── persistence ──────────────────────────────────────────────────────────────

def test_mapping_persists_to_file_and_reloads(tmp_path):
    path = tmp_path / "tag_aliases.json"
    TagAliasRegistry(path).register("atlas", ["proj:atlas"])

    reloaded = TagAliasRegistry(path)
    assert reloaded.resolve("proj:atlas") == "atlas"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["groups"][0]["canonical"] == "atlas"


def test_registry_hot_reloads_after_external_change(tmp_path):
    path = tmp_path / "tag_aliases.json"
    first = TagAliasRegistry(path)
    first.register("atlas", ["proj:atlas"])

    second = TagAliasRegistry(path)
    second.register("borealis", ["proj:borealis"])

    # First instance must observe the other process's edit via mtime.
    assert first.resolve("proj:borealis") == "borealis"


def test_corrupt_file_is_served_as_empty_and_not_overwritten(tmp_path):
    path = tmp_path / "tag_aliases.json"
    path.write_text("{not json", encoding="utf-8")

    registry = TagAliasRegistry(path)
    assert registry.list_groups() == []
    assert registry.resolve("anything") == "anything"

    # A later valid write parks the corrupt file instead of destroying it.
    registry.register("atlas", ["proj:atlas"])
    assert registry.resolve("proj:atlas") == "atlas"
    assert list(tmp_path.glob("tag_aliases.json.corrupt-*"))


def test_registry_file_removal_is_observed(tmp_path):
    path = tmp_path / "tag_aliases.json"
    registry = TagAliasRegistry(path)
    registry.register("atlas", ["proj:atlas"])
    path.unlink()
    # mtime-based reload notices the deletion.
    assert registry.resolve("proj:atlas") == "proj:atlas"


def test_get_registry_singleton_and_cache_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TAG_ALIASES_PATH", str(tmp_path / "singleton.json"))
    reset_registry_cache()
    try:
        first = get_registry()
        second = get_registry()
        assert first is second
    finally:
        reset_registry_cache()
    # After reset a fresh instance is built (and picks up edits on disk).
    assert get_registry() is not first
    reset_registry_cache()
