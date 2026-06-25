"""Shared provider compatibility helpers.

Lightweight utilities for parsing OpenAI-compatible responses and normalizing
errors. Used by both LLM and embedding factory modules.
"""

from __future__ import annotations

from typing import Any


class OpenAICompatibleProviderError(RuntimeError):
    """Provider error returned inside an OpenAI-compatible response payload."""

    @classmethod
    def from_error(
        cls,
        operation: str,
        error: Any,
        response: Any,
    ) -> "OpenAICompatibleProviderError":
        message = _error_value(error, "message") or repr(error)
        code = _error_value(error, "code")
        return cls(
            f"OpenAI-compatible {operation} provider returned error"
            f"{f' ({code})' if code is not None else ''}: {message}. "
            f"Response: {_safe_response_summary(response)}"
        )


def _response_error(response: Any) -> Any:
    payload = _response_payload(response)
    if isinstance(payload, dict) and payload.get("error"):
        return payload["error"]
    return getattr(response, "error", None)


def _response_payload(response: Any) -> Any:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return None
    return None


def _error_value(error: Any, key: str) -> Any:
    if isinstance(error, dict):
        return error.get(key)
    return getattr(error, key, None)


def _safe_response_summary(response: Any) -> str:
    if response is None:
        return "None"
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return repr(model_dump())
        except Exception:
            pass
    return repr(response)
