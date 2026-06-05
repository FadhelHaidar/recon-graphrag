"""Local LLM interface types.

These protocols keep Recon-GraphRAG independent from provider-specific
wrapper classes while preserving the simple ``ainvoke(...).content`` contract
used throughout the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class LLMUsage:
    """Token usage returned by a model provider, when available."""

    request_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """Response returned by an LLM invocation."""

    content: str
    usage: Optional[LLMUsage] = None


@runtime_checkable
class BaseLLM(Protocol):
    """Protocol for LLM providers accepted by Recon-GraphRAG."""

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Synchronously generate a response."""
        ...

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Asynchronously generate a response."""
        ...


__all__ = ["BaseLLM", "LLMResponse", "LLMUsage"]
