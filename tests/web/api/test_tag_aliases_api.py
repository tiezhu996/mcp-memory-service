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

"""REST tests for the tag alias administration endpoints."""

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.storage.tag_aliases import reset_registry_cache


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TAG_ALIASES_PATH", str(tmp_path / "api_tag_aliases.json"))
    reset_registry_cache()

    monkeypatch.setenv("MCP_API_KEY", "")
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "false")
    monkeypatch.setenv("MCP_ALLOW_ANONYMOUS_ACCESS", "true")

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        AuthenticationResult,
        get_current_user,
        require_read_access,
        require_write_access,
    )

    async def _user():
        return AuthenticationResult(
            authenticated=True, client_id="test", scope="read write", auth_method="test"
        )

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[require_read_access] = _user
    app.dependency_overrides[require_write_access] = _user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    reset_registry_cache()


def test_register_list_remove_flow(client):
    response = client.post(
        "/api/tags/aliases",
        json={"canonical": "atlas", "aliases": ["Proj:Atlas", "atlasproj"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["canonical"] == "atlas"
    assert set(body["aliases"]) == {"proj:atlas", "atlasproj"}

    listing = client.get("/api/tags/aliases").json()
    assert listing["groups"][0]["canonical"] == "atlas"

    response = client.request(
        "DELETE",
        "/api/tags/aliases",
        json={"canonical": "atlas"},
    )
    assert response.status_code == 200
    assert response.json()["removed"] == "group"
    assert client.get("/api/tags/aliases").json()["groups"] == []


def test_register_conflict_reports_409(client):
    client.post("/api/tags/aliases", json={"canonical": "atlas", "aliases": ["proj:atlas"]})
    response = client.post(
        "/api/tags/aliases",
        json={"canonical": "borealis", "aliases": ["proj:atlas"]},
    )
    assert response.status_code == 409


def test_remove_unknown_reports_404(client):
    response = client.request(
        "DELETE",
        "/api/tags/aliases",
        json={"alias": "ghost"},
    )
    assert response.status_code == 404


def test_remove_requires_exactly_one_selector(client):
    response = client.request(
        "DELETE",
        "/api/tags/aliases",
        json={"canonical": "a", "alias": "b"},
    )
    assert response.status_code == 422
