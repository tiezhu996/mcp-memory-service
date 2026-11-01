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
Tests that the CLI can actually build the hybrid backend.

`get_storage()` had no hybrid branch, so `memory status --storage-backend hybrid`
raised `ValueError: Unsupported storage backend: hybrid` (issue #233). Both the
`click.Choice` on three commands and the function's own docstring offered
`hybrid`, and hybrid is what CLAUDE.md recommends for production — so the
promise was made in three places and kept in none.

HybridMemoryStorage is patched throughout: a real one starts a background sync
task, which has no business running inside the unit suite.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.cli import utils

COMPLETE_CF = {
    "CLOUDFLARE_API_TOKEN": "token",
    "CLOUDFLARE_ACCOUNT_ID": "account",
    "CLOUDFLARE_VECTORIZE_INDEX": "index",
    "CLOUDFLARE_D1_DATABASE_ID": "d1",
}


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Point the config constants get_storage imports at harmless values."""
    import mcp_memory_service.config as config  # inline import: at module level this would run load_dotenv before the fixture can isolate anything

    defaults = {
        "SQLITE_VEC_PATH": str(tmp_path / "test.db"),
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "CLOUDFLARE_API_TOKEN": None,
        "CLOUDFLARE_ACCOUNT_ID": None,
        "CLOUDFLARE_VECTORIZE_INDEX": None,
        "CLOUDFLARE_D1_DATABASE_ID": None,
        "CLOUDFLARE_R2_BUCKET": None,
        "CLOUDFLARE_EMBEDDING_MODEL": "@cf/baai/bge-base-en-v1.5",
        "CLOUDFLARE_LARGE_CONTENT_THRESHOLD": 1024,
        "CLOUDFLARE_MAX_RETRIES": 3,
        "CLOUDFLARE_BASE_DELAY": 1,
        "HYBRID_SYNC_INTERVAL": None,
        "HYBRID_BATCH_SIZE": None,
    }
    for name, value in defaults.items():
        monkeypatch.setattr(config, name, value, raising=False)
    return config


@pytest.fixture
def hybrid_cls():
    instance = MagicMock()
    instance.initialize = AsyncMock()
    with patch("mcp_memory_service.storage.hybrid.HybridMemoryStorage") as cls:
        cls.return_value = instance
        yield cls


class TestHybridBranchExists:

    @pytest.mark.asyncio
    async def test_hybrid_no_longer_raises(self, cfg, hybrid_cls):
        """The regression itself: this used to raise ValueError. The backend is
        returned behind the tag-alias wrapper, which delegates initialize()."""
        from mcp_memory_service.storage.alias_wrapper import AliasedStorage

        storage = await utils.get_storage("hybrid")
        assert isinstance(storage, AliasedStorage)
        assert storage.inner_storage is hybrid_cls.return_value
        hybrid_cls.return_value.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_backend_still_raises(self, cfg):
        """Adding a branch must not turn the guard into a catch-all."""
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            await utils.get_storage("no-such-backend")


class TestConstructionMatchesTheServer:
    """The server's eager-init path is the reference. A CLI that builds hybrid
    differently would report on a different store than the one running."""

    @pytest.mark.asyncio
    async def test_passes_the_same_kwargs(self, cfg, hybrid_cls, tmp_path):
        await utils.get_storage("hybrid")
        kwargs = hybrid_cls.call_args.kwargs
        assert kwargs["sqlite_db_path"] == str(tmp_path / "test.db")
        assert kwargs["embedding_model"] == "all-MiniLM-L6-v2"
        assert set(kwargs) == {
            "sqlite_db_path", "embedding_model", "cloudflare_config",
            "sync_interval", "batch_size",
        }

    @pytest.mark.asyncio
    async def test_none_config_values_fall_back(self, cfg, hybrid_cls):
        """HYBRID_SYNC_INTERVAL and HYBRID_BATCH_SIZE are None whenever hybrid
        is not the configured backend, which is exactly the case when it is
        named on the command line instead."""
        await utils.get_storage("hybrid")
        kwargs = hybrid_cls.call_args.kwargs
        assert kwargs["sync_interval"] == 300
        assert kwargs["batch_size"] == 50


class TestIncompleteCloudflareConfig:
    """The branch that makes this usable for the documented setup: with
    MCP_HYBRID_SYNC_OWNER=http the CLI side has no Cloudflare credentials on
    purpose, so demanding them would leave the command broken."""

    @pytest.mark.asyncio
    async def test_missing_credentials_degrade_to_sqlite_only(self, cfg, hybrid_cls):
        await utils.get_storage("hybrid")
        assert hybrid_cls.call_args.kwargs["cloudflare_config"] is None

    @pytest.mark.asyncio
    async def test_partial_credentials_also_degrade(self, cfg, hybrid_cls, monkeypatch):
        """Three of the four required values is still not a usable config."""
        for name, value in list(COMPLETE_CF.items())[:3]:
            monkeypatch.setattr(cfg, name, value)
        await utils.get_storage("hybrid")
        assert hybrid_cls.call_args.kwargs["cloudflare_config"] is None

    @pytest.mark.asyncio
    async def test_complete_credentials_are_passed_through(self, cfg, hybrid_cls, monkeypatch):
        for name, value in COMPLETE_CF.items():
            monkeypatch.setattr(cfg, name, value)
        await utils.get_storage("hybrid")
        cf = hybrid_cls.call_args.kwargs["cloudflare_config"]
        assert cf is not None
        assert cf["api_token"] == "token"
        assert cf["d1_database_id"] == "d1"
