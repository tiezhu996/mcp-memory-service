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

"""Tests for the tag_alias_* MCP tool handlers."""

import json

import pytest

from mcp_memory_service.server.handlers import tags as tag_handlers
from mcp_memory_service.storage.tag_aliases import reset_registry_cache


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TAG_ALIASES_PATH", str(tmp_path / "mcp_tag_aliases.json"))
    reset_registry_cache()
    yield
    reset_registry_cache()


def _text(result):
    return result[0].text


async def test_register_list_remove_tools(tmp_path):
    result = await tag_handlers.handle_tag_alias_register(None, {
        "canonical": " Atlas ",
        "aliases": ["Proj:Atlas", "atlasproj", "atlasproj"],
    })
    assert "atlas" in _text(result)

    listed = await tag_handlers.handle_tag_alias_list(None, {})
    payload = json.loads(_text(listed))
    group = payload["groups"][0]
    assert group["canonical"] == "atlas"
    assert group["aliases"] == ["proj:atlas", "atlasproj"]
    assert group["created_at"] and group["updated_at"]

    removed = await tag_handlers.handle_tag_alias_remove(None, {"canonical": "atlas"})
    assert "Removed alias group" in _text(removed)

    payload = json.loads(_text(await tag_handlers.handle_tag_alias_list(None, {})))
    assert payload["groups"] == []


async def test_register_accepts_comma_separated_string():
    result = await tag_handlers.handle_tag_alias_register(None, {
        "canonical": "atlas",
        "aliases": "proj:atlas, atlasproj",
    })
    assert "atlasproj" in _text(result)


async def test_register_rejects_bad_arguments():
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_register(None, {"canonical": "", "aliases": ["x"]})
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_register(None, {"canonical": "atlas", "aliases": []})
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_register(None, {"canonical": "atlas"})
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_register(None, {"canonical": "atlas", "aliases": 123})


async def test_remove_requires_one_selector():
    await tag_handlers.handle_tag_alias_register(None, {
        "canonical": "atlas", "aliases": ["proj:atlas"]
    })
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_remove(None, {})
    with pytest.raises(ValueError):
        await tag_handlers.handle_tag_alias_remove(None, {"canonical": "atlas", "alias": "proj:atlas"})
