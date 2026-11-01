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

"""REST endpoints for tag alias (canonical tag) administration.

The mapping is backend-independent (a JSON file under the service data
directory); these endpoints only need the registry, not a storage backend.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...storage.tag_aliases import get_registry
from ..oauth.middleware import (
    AuthenticationResult,
    require_read_access,
    require_write_access,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TagAliasGroup(BaseModel):
    canonical: str
    aliases: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TagAliasListResponse(BaseModel):
    registry_path: str
    groups: List[TagAliasGroup]


class TagAliasRegisterRequest(BaseModel):
    canonical: str = Field(..., min_length=1, description="Primary tag")
    aliases: List[str] = Field(..., min_length=1, description="Variant spellings")


class TagAliasRegisterResponse(BaseModel):
    canonical: str
    aliases: List[str]
    added: List[str]
    merged_canonical_tags: List[str] = []
    updated_at: Optional[str] = None


class TagAliasRemoveRequest(BaseModel):
    canonical: Optional[str] = Field(None, description="Remove the whole group")
    alias: Optional[str] = Field(None, description="Remove one variant only")


class TagAliasRemoveResponse(BaseModel):
    removed: str
    canonical: str
    alias: Optional[str] = None
    aliases: List[str] = []
    updated_at: Optional[str] = None


@router.get(
    "/tags/aliases",
    response_model=TagAliasListResponse,
    tags=["tags"],
)
async def list_tag_aliases(
    user: AuthenticationResult = Depends(require_read_access),
):
    """List all registered canonical-tag/alias groups."""
    registry = get_registry()
    return TagAliasListResponse(
        registry_path=str(registry.path),
        groups=[TagAliasGroup(**group) for group in registry.list_groups()],
    )


@router.post(
    "/tags/aliases",
    response_model=TagAliasRegisterResponse,
    tags=["tags"],
)
async def register_tag_alias(
    request: TagAliasRegisterRequest,
    user: AuthenticationResult = Depends(require_write_access),
):
    """Register variant spellings under one canonical tag."""
    try:
        result = get_registry().register(request.canonical, request.aliases)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        logger.error("Failed writing tag alias registry: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not persist tag aliases: {exc}")
    return TagAliasRegisterResponse(**result)


@router.delete(
    "/tags/aliases",
    response_model=TagAliasRemoveResponse,
    tags=["tags"],
)
async def remove_tag_alias(
    request: TagAliasRemoveRequest,
    user: AuthenticationResult = Depends(require_write_access),
):
    """Undo a registration: remove one alias ({"alias": ...}) or a whole group ({"canonical": ...})."""
    if bool(request.canonical) == bool(request.alias):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'canonical' or 'alias'",
        )
    try:
        result = get_registry().unregister(canonical=request.canonical, alias=request.alias)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        logger.error("Failed writing tag alias registry: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not persist tag aliases: {exc}")
    return TagAliasRemoveResponse(**result)
