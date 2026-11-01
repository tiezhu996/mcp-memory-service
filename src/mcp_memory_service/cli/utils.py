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
CLI utilities for MCP Memory Service.
"""

import logging
import os
from typing import Optional

from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)


def _wrap_tag_aliases(storage: MemoryStorage) -> MemoryStorage:
    """Apply the same tag alias layer the servers use (idempotent)."""
    from ..storage.alias_wrapper import wrap_with_tag_aliases
    return wrap_with_tag_aliases(storage)


async def get_storage(backend: Optional[str] = None, strict_dimension_check: bool = True) -> MemoryStorage:
    """
    Get storage backend for CLI operations.

    Args:
        backend: Storage backend name ('sqlite_vec', 'cloudflare', 'hybrid', or 'milvus')
        strict_dimension_check: Passed to the sqlite_vec backend's initialize().
            When False, an existing-database embedding-dimension mismatch is logged
            as a warning instead of raising, so read-only diagnostics (`memory status`)
            can open and report on a mismatched database (#143).

    Returns:
        Initialized storage backend
    """
    # Determine backend
    if backend is None:
        backend = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec').lower()

    backend = backend.lower()

    if backend in ('sqlite_vec', 'sqlite-vec'):
        from ..storage.sqlite_vec import SqliteVecMemoryStorage
        from ..config import EMBEDDING_MODEL_NAME, SQLITE_VEC_PATH
        storage = SqliteVecMemoryStorage(
            SQLITE_VEC_PATH,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        await storage.initialize(strict_dimension_check=strict_dimension_check)
        return _wrap_tag_aliases(storage)
    elif backend == 'cloudflare':
        from ..storage.cloudflare import CloudflareStorage
        from ..config import (
            CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
            CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
            CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
            CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
            CLOUDFLARE_BASE_DELAY
        )
        storage = CloudflareStorage(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY
        )
        await storage.initialize()
        return _wrap_tag_aliases(storage)
    elif backend == 'hybrid':
        return _wrap_tag_aliases(await _build_hybrid_storage())
    elif backend == 'milvus':
        from ..storage.milvus import MilvusMemoryStorage
        from ..config import (
            MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION_NAME, EMBEDDING_MODEL_NAME
        )
        storage = MilvusMemoryStorage(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            collection_name=MILVUS_COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        await storage.initialize()
        return _wrap_tag_aliases(storage)
    else:
        raise ValueError(f"Unsupported storage backend: {backend}")


def _hybrid_cloudflare_config() -> Optional[dict]:
    """Assemble the Cloudflare half of the hybrid config, or None.

    None means SQLite-only mode, and that is a supported outcome rather than a
    failure: CLAUDE.md tells hybrid deployments to set
    MCP_HYBRID_SYNC_OWNER=http, so the CLI side deliberately has no Cloudflare
    credentials. Demanding them would leave the command broken for exactly the
    setup being recommended.
    """
    from ..config import (
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY
    )

    if not all([CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
                CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID]):
        logger.warning(
            "Incomplete Cloudflare configuration; hybrid storage will run "
            "in SQLite-only mode."
        )
        return None

    return {
        'api_token': CLOUDFLARE_API_TOKEN,
        'account_id': CLOUDFLARE_ACCOUNT_ID,
        'vectorize_index': CLOUDFLARE_VECTORIZE_INDEX,
        'd1_database_id': CLOUDFLARE_D1_DATABASE_ID,
        'r2_bucket': CLOUDFLARE_R2_BUCKET,
        'embedding_model': CLOUDFLARE_EMBEDDING_MODEL,
        'large_content_threshold': CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
        'max_retries': CLOUDFLARE_MAX_RETRIES,
        'base_delay': CLOUDFLARE_BASE_DELAY
    }


async def _build_hybrid_storage() -> MemoryStorage:
    """Build the hybrid backend the way server_impl.py's eager-init path does.

    Kept identical to that path on purpose: a CLI that assembled hybrid
    differently would report on a different store than the one running. This is
    now the third copy of this construction (server_impl twice, here once); a
    shared factory belongs with the launcher consolidation in #281.

    HybridMemoryStorage.initialize() takes no arguments, so
    get_storage()'s strict_dimension_check does not reach the wrapped
    SQLite-vec store on this path.
    """
    from ..storage.hybrid import HybridMemoryStorage
    from ..config import (
        SQLITE_VEC_PATH, EMBEDDING_MODEL_NAME,
        HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE
    )

    # HYBRID_SYNC_INTERVAL and HYBRID_BATCH_SIZE are only populated when hybrid
    # is the *configured* backend, so they are None whenever it is named on the
    # command line against a different environment. Same fallbacks as
    # server_impl.py uses for that case.
    storage = HybridMemoryStorage(
        sqlite_db_path=SQLITE_VEC_PATH,
        embedding_model=EMBEDDING_MODEL_NAME,
        cloudflare_config=_hybrid_cloudflare_config(),
        sync_interval=HYBRID_SYNC_INTERVAL or 300,
        batch_size=HYBRID_BATCH_SIZE or 50
    )
    await storage.initialize()
    return storage
