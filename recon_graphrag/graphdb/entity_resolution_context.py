"""Shared helpers for property-aware entity resolution review."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONTEXT_VALUE_LIMIT = 500
CONTEXT_SEQUENCE_LIMIT = 8

STANDARD_CONTEXT_FIELDS = (
    "name",
    "title",
    "description",
    "canonical_key",
    "human_readable_id",
    "aliases",
)

INTERNAL_CONTEXT_FIELDS = {
    "embedding",
    "graph_name",
    "id",
    "created",
    "updated",
    "text",
    "text_hash",
    "source_chunk_ids",
}


def build_entity_profiles(
    entities_by_node_id: dict[Any, Any],
    node_ids: Sequence[Any],
    *,
    context_properties: dict[str, list[str]] | list[str] | None = None,
    context_mode: str = "safe_defaults",
) -> list[dict[str, Any]]:
    """Build compact entity profiles for LLM review prompts."""
    return [
        build_entity_profile(
            entities_by_node_id[node_id],
            context_properties=context_properties,
            context_mode=context_mode,
        )
        for node_id in node_ids
        if node_id in entities_by_node_id
    ]


def build_entity_profile(
    entity: Any,
    *,
    context_properties: dict[str, list[str]] | list[str] | None = None,
    context_mode: str = "safe_defaults",
) -> dict[str, Any]:
    properties = dict(getattr(entity, "properties", {}) or {})
    label = getattr(entity, "domain_label", "__Entity__")
    profile = {
        "label": label,
        "name": _clean_value(
            properties.get("name") or getattr(entity, "resolve_value", "")
        ),
        "title": _clean_value(properties.get("title")),
        "description": _clean_value(properties.get("description")),
        "canonical_key": _clean_value(properties.get("canonical_key")),
        "human_readable_id": _clean_value(properties.get("human_readable_id")),
        "aliases": _clean_value(properties.get("aliases")),
    }
    profile = {
        key: value for key, value in profile.items() if value not in (None, "", [])
    }

    selected = _select_context_properties(
        properties,
        label=label,
        context_properties=context_properties,
        context_mode=context_mode,
    )
    if selected:
        profile["properties"] = selected
    return profile


def conflict_for_group(
    group: Sequence[Any],
    conflict_properties: dict[str, list[str]] | list[str] | None,
) -> list[dict[str, Any]]:
    """Return non-empty property conflicts for an entity candidate group."""
    if not conflict_properties or len(group) < 2:
        return []

    conflicts: list[dict[str, Any]] = []
    for index, left in enumerate(group):
        for right in group[index + 1 :]:
            properties = _configured_properties(
                getattr(left, "domain_label", "__Entity__"),
                conflict_properties,
            )
            properties += [
                prop
                for prop in _configured_properties(
                    getattr(right, "domain_label", "__Entity__"),
                    conflict_properties,
                )
                if prop not in properties
            ]
            for prop in properties:
                left_value = _first_value(
                    (getattr(left, "properties", {}) or {}).get(prop)
                )
                right_value = _first_value(
                    (getattr(right, "properties", {}) or {}).get(prop)
                )
                if _is_empty(left_value) or _is_empty(right_value):
                    continue
                if _normalize_conflict_value(left_value) != _normalize_conflict_value(
                    right_value
                ):
                    conflicts.append(
                        {
                            "property": prop,
                            "values": [
                                _clean_value(left_value),
                                _clean_value(right_value),
                            ],
                            "names": [
                                getattr(left, "resolve_value", ""),
                                getattr(right, "resolve_value", ""),
                            ],
                        }
                    )
    return conflicts


def blocked_review_group(group: Sequence[Any], conflicts: list[dict[str, Any]]) -> dict:
    return {
        "domain_label": (
            getattr(group[0], "domain_label", "__Entity__")
            if group
            else "__Entity__"
        ),
        "names": [getattr(entity, "resolve_value", "") for entity in group],
        "node_ids": [getattr(entity, "node_id", None) for entity in group],
        "reason": "property_conflict",
        "conflicts": conflicts,
        "scores": {"fuzzy": None, "embedding": None, "llm": None},
        "decision": "blocked",
        "llm_review": {
            "same_entity": False,
            "confidence": 1.0,
            "reason": "Configured conflict properties differ.",
            "merge_allowed": False,
        },
    }


def _select_context_properties(
    properties: dict[str, Any],
    *,
    label: str,
    context_properties: dict[str, list[str]] | list[str] | None,
    context_mode: str,
) -> dict[str, Any]:
    if context_mode not in ("safe_defaults", "all", "config_only"):
        raise ValueError(f"Unknown entity resolution context mode: {context_mode}")

    configured = _configured_properties(label, context_properties)
    if configured:
        names = configured
    elif context_mode == "config_only":
        names = []
    else:
        names = [
            key
            for key in properties
            if key not in INTERNAL_CONTEXT_FIELDS and key not in STANDARD_CONTEXT_FIELDS
        ]

    selected = {}
    for name in names:
        if name in INTERNAL_CONTEXT_FIELDS and context_mode != "all":
            continue
        if name not in properties:
            continue
        value = _clean_value(properties.get(name))
        if value not in (None, "", [], {}):
            selected[name] = value
    return selected


def _configured_properties(
    label: str,
    config: dict[str, list[str]] | list[str] | None,
) -> list[str]:
    if config is None:
        return []
    if isinstance(config, list):
        return [str(item) for item in config]
    if isinstance(config, dict):
        values = []
        for key in ("*", "__default__", label):
            raw = config.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw)
        return list(dict.fromkeys(values))
    return []


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:CONTEXT_VALUE_LIMIT]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in list(value.items())[:CONTEXT_SEQUENCE_LIMIT]:
            if str(key) in INTERNAL_CONTEXT_FIELDS:
                continue
            cleaned[str(key)] = _clean_value(item)
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clean_value(item) for item in list(value)[:CONTEXT_SEQUENCE_LIMIT]]
    return str(value)[:CONTEXT_VALUE_LIMIT]


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _normalize_conflict_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return str(value).casefold()
