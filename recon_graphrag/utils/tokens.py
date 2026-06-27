"""Token counting and packing utilities.

These primitives are intentionally dependency-light. The default
``ApproximateTokenCounter`` uses a configurable character-to-token ratio so that
budgeting works without an exact tokenizer. For provider-level accuracy, the
``TiktokenTokenCounter`` adapter is available and ``tiktoken`` is included as a
required project dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting and truncation."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return the longest prefix of ``text`` that fits in ``max_tokens``."""
        ...


class ApproximateTokenCounter:
    """Lightweight token estimator.

    Uses ``ceil(len(text) / ratio)`` as the token count. This is a coarse
    estimate suitable for budgeting when no exact tokenizer is available.
    Callers requiring provider-level accuracy should pass an exact counter.
    """

    DEFAULT_RATIO = 4.0

    def __init__(self, ratio: float | None = None):
        ratio = ratio if ratio is not None else self.DEFAULT_RATIO
        if ratio <= 0:
            raise ValueError("ratio must be > 0")
        self._ratio = ratio

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._ratio)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if max_tokens == 0:
            return ""
        if not text:
            return ""
        max_chars = int(max_tokens * self._ratio)
        return text[:max_chars]


@dataclass(frozen=True)
class PackItem:
    """An already ordered item to include in a token budget."""

    id: str
    text: str
    priority: float = 0.0


@dataclass(frozen=True)
class PackResult:
    """Result of greedy token packing."""

    included: list[PackItem]
    excluded: list[PackItem]
    used_tokens: int
    max_tokens: int
    truncated_item_ids: list[str]


def pack_items(
    items: list[PackItem],
    max_tokens: int,
    counter: TokenCounter | None = None,
    *,
    truncate_oversized: bool = False,
) -> PackResult:
    """Greedily pack already ordered items into ``max_tokens``.

    The caller owns ordering policy. This helper only applies stable greedy
    inclusion and optionally truncates a single oversized item to fit remaining
    budget.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    counter = counter or ApproximateTokenCounter()
    included: list[PackItem] = []
    excluded: list[PackItem] = []
    truncated_item_ids: list[str] = []
    used_tokens = 0

    for item in items:
        item_tokens = counter.count(item.text)
        if used_tokens + item_tokens <= max_tokens:
            included.append(item)
            used_tokens += item_tokens
            continue

        remaining = max_tokens - used_tokens
        if truncate_oversized and remaining > 0:
            truncated_text = counter.truncate(item.text, remaining)
            if truncated_text:
                truncated = PackItem(
                    id=item.id,
                    text=truncated_text,
                    priority=item.priority,
                )
                included.append(truncated)
                used_tokens += counter.count(truncated_text)
                truncated_item_ids.append(item.id)
                continue

        excluded.append(item)

    return PackResult(
        included=included,
        excluded=excluded,
        used_tokens=used_tokens,
        max_tokens=max_tokens,
        truncated_item_ids=truncated_item_ids,
    )


class TiktokenTokenCounter:
    """Exact token counter backed by ``tiktoken``."""

    def __init__(self, model: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TiktokenTokenCounter requires the 'tiktoken' package. "
                "Install it or use ApproximateTokenCounter."
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(model)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TiktokenTokenCounter could not load the requested tiktoken "
                "encoding. Install/cache the encoding or use "
                "ApproximateTokenCounter."
            ) from exc

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if max_tokens == 0:
            return ""
        if not text:
            return ""
        encoded = self._encoding.encode(text)
        return self._encoding.decode(encoded[:max_tokens])


def create_token_counter(name: str = "approximate", **kwargs) -> TokenCounter:
    """Factory for token counters.

    Supported names:
    - ``"approximate"``: ``ApproximateTokenCounter`` (always available).
      Accepts ``ratio``.
    - ``"tiktoken"``: ``TiktokenTokenCounter`` (uses ``tiktoken``).
      Accepts ``model`` (defaults to ``"cl100k_base"``).
    """
    if name == "approximate":
        return ApproximateTokenCounter(**kwargs)
    if name == "tiktoken":
        return TiktokenTokenCounter(**kwargs)
    raise ValueError(f"Unknown token counter: {name!r}")


def count_tokens(text: str, counter: TokenCounter | None = None) -> int:
    """Convenience helper: count tokens in ``text``."""
    return (counter or ApproximateTokenCounter()).count(text)


def truncate_text(text: str, max_tokens: int, counter: TokenCounter | None = None) -> str:
    """Convenience helper: truncate ``text`` to ``max_tokens``."""
    return (counter or ApproximateTokenCounter()).truncate(text, max_tokens)


__all__ = [
    "TokenCounter",
    "ApproximateTokenCounter",
    "PackItem",
    "PackResult",
    "TiktokenTokenCounter",
    "create_token_counter",
    "count_tokens",
    "pack_items",
    "truncate_text",
]
