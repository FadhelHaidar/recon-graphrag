"""Shared effective inputs for entity description prompts and freshness checks."""

from __future__ import annotations

import hashlib
import json

from recon_graphrag.graphdb.entity_resolution_context import (
    INTERNAL_CONTEXT_FIELDS,
    STANDARD_CONTEXT_FIELDS,
)

_NON_CONTEXT_FIELDS = INTERNAL_CONTEXT_FIELDS | set(STANDARD_CONTEXT_FIELDS) | {
    "uid", "descriptions", "observation_count", "type", "weight",
}


def summary_context_properties(item: dict) -> dict:
    """Exclude SDK bookkeeping and empty domain values from summary context."""
    return {
        key: value for key, value in (item.get("props") or {}).items()
        if key not in _NON_CONTEXT_FIELDS
        and not key.startswith(("embedding_", "description_"))
        and value not in (None, "", [], {})
    }


def entity_summary_fingerprint(entity: dict) -> str:
    """Hash the same normalized entity inputs that the summary prompt consumes."""
    payload = {
        "id": entity.get("entity_id") or entity.get("id"),
        "name": str(entity.get("name") or entity.get("entity_id") or ""),
        "type": str(entity.get("type") or "Entity"),
        "descriptions": sorted(
            str(text).strip() for text in (entity.get("descriptions") or []) if str(text).strip()
        ),
        "properties": {key: str(value) for key, value in summary_context_properties(entity).items()},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
