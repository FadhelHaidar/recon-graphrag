"""Canonical text embedded for an entity, plus its fingerprint.

The writer records the fingerprint of the text it expects to be embedded and
the embedder records the fingerprint of the text it actually embedded, so a
vector can be detected as stale after a name or description change. Both sides
must build the text identically, which is why it lives here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def value_to_text(value: Any) -> str:
    """Render a property value the way entity embedding text expects."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {value_to_text(item)}"
            for key, item in value.items()
            if item is not None
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(text for item in value if (text := value_to_text(item)))
    return str(value)


def entity_embedding_text(
    *, labels: list[str] | None, name: Any, description: Any
) -> str:
    """Build the exact string embedded for one entity."""
    label = [lbl for lbl in (labels or []) if lbl != "__Entity__"]
    label = label[0] if label else "Entity"
    parts = [f"{label}: {value_to_text(name)}"]
    desc = value_to_text(description)
    if desc:
        parts.append(desc)
    return " - ".join(parts)


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entity_embedding_fingerprint(
    *, labels: list[str] | None, name: Any, description: Any
) -> str:
    return text_fingerprint(
        entity_embedding_text(labels=labels, name=name, description=description)
    )


def embedder_fingerprint(embedder: Any) -> str:
    """Best-effort identity of the embedding provider and model.

    Unwraps one wrapper level so the fingerprint follows the model rather than
    the ``ModelParamsEmbedder``/``TruncatingEmbedder`` shell. Providers that
    expose neither a model nor a name are identified by their class alone, so
    pointing a different provider class at the same graph still refreshes
    vectors.
    """
    inner = getattr(embedder, "_embedder", embedder)
    model = getattr(inner, "model", None)
    identity = {
        "class": f"{type(inner).__module__}.{type(inner).__qualname__}",
        "model": _model_name(model),
        "params": getattr(embedder, "_model_params", None)
        or getattr(inner, "_model_params", None),
        "dim": getattr(embedder, "target_embedding_dim", None)
        or getattr(inner, "target_embedding_dim", None),
    }
    raw = json.dumps(identity, sort_keys=True, default=str)
    return text_fingerprint(raw)


def _model_name(model: Any) -> str | None:
    if model is None or isinstance(model, str):
        return model
    for attribute in ("model_name_or_path", "name_or_path", "model_name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return f"{type(model).__module__}.{type(model).__qualname__}"


__all__ = [
    "embedder_fingerprint",
    "entity_embedding_fingerprint",
    "entity_embedding_text",
    "text_fingerprint",
    "value_to_text",
]
