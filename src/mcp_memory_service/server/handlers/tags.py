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

"""MCP handlers for tag alias administration.

These tools manage the backend-independent canonical-tag mapping
(``storage/tag_aliases.py``). They work even when the storage backend itself
is unavailable: the registry is a small JSON file, not backend state.
"""

import json
import logging
import traceback
from typing import List

from mcp import types

from ...storage.tag_aliases import get_registry

logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


async def handle_tag_alias_list(server, arguments: dict) -> List[types.TextContent]:
    """Return every registered alias group as pretty-printed JSON."""
    try:
        registry = get_registry()
        payload = {
            "registry_path": str(registry.path),
            "groups": registry.list_groups(),
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))]
    except Exception as e:
        logger.error("tag_alias_list failed: %s\n%s", _sanitize_log_value(e), traceback.format_exc())
        raise


async def handle_tag_alias_register(server, arguments: dict) -> List[types.TextContent]:
    """Register variant spellings under one canonical tag."""
    canonical = arguments.get("canonical")
    aliases = arguments.get("aliases")

    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError("canonical must be a single non-empty tag string")
    if isinstance(aliases, str):
        alias_list: list = [part.strip() for part in aliases.split(",") if part.strip()]
    elif isinstance(aliases, list) and all(isinstance(a, str) for a in aliases):
        alias_list = aliases
    else:
        raise ValueError("aliases must be an array of tag strings (or a comma-separated string)")
    if not alias_list:
        raise ValueError("at least one alias is required")

    try:
        result = get_registry().register(canonical, alias_list)
        return [types.TextContent(
            type="text",
            text=(
                f"Registered aliases for canonical tag '{result['canonical']}': "
                f"{', '.join(result['aliases']) or '(no aliases)'}"
                + (f" (merged groups: {', '.join(result['merged_canonical_tags'])})"
                   if result.get("merged_canonical_tags") else "")
            ),
        )]
    except ValueError:
        # Validation message is user-facing and safe; raise so isError=True.
        raise
    except Exception as e:
        logger.error("tag_alias_register failed: %s\n%s", _sanitize_log_value(e), traceback.format_exc())
        raise


async def handle_tag_alias_remove(server, arguments: dict) -> List[types.TextContent]:
    """Undo a registration: remove one alias or a whole canonical group."""
    canonical = arguments.get("canonical")
    alias = arguments.get("alias")

    if bool(canonical) == bool(alias):
        raise ValueError("provide exactly one of canonical or alias")

    try:
        result = get_registry().unregister(canonical=canonical, alias=alias)
        if result["removed"] == "group":
            message = (
                f"Removed alias group '{result['canonical']}' "
                f"({len(result['aliases'])} alias(es))"
            )
        else:
            remaining = result.get("aliases") or []
            message = (
                f"Removed alias '{result['alias']}' from '{result['canonical']}'; "
                f"{len(remaining)} alias(es) remain"
            )
        return [types.TextContent(type="text", text=message)]
    except ValueError:
        raise
    except Exception as e:
        logger.error("tag_alias_remove failed: %s\n%s", _sanitize_log_value(e), traceback.format_exc())
        raise
