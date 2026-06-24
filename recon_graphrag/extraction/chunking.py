"""Text chunking and sliding page windows for graph extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Minimal protocol for token counting (matches utils.tokens.TokenCounter)."""

    def count(self, text: str) -> int: ...
    def truncate(self, text: str, max_tokens: int) -> str: ...


@dataclass
class TextChunk:
    id: str
    text: str
    index: int
    metadata: dict


class TextChunker:
    """Split text into overlapping chunks.

    Supports two units:
    - ``"characters"`` (default): existing character-based sliding window.
    - ``"tokens"``: token-based sliding window using a ``TokenCounter``.
      Requires ``token_counter`` to be provided.

    In both modes each chunk retains ``char_start`` and ``char_end`` in the
    original text. Token mode additionally records ``token_start`` and
    ``token_end``.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        unit: str = "characters",
        token_counter: TokenCounter | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if unit not in ("characters", "tokens"):
            raise ValueError(f"unit must be 'characters' or 'tokens', got {unit!r}")
        if unit == "tokens" and token_counter is None:
            raise ValueError("token_counter is required when unit='tokens'")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit
        self.token_counter = token_counter

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        if self.unit == "tokens":
            return self._chunk_by_tokens(text, document_id, metadata or {})
        return self._chunk_by_characters(text, document_id, metadata or {})

    def _chunk_by_characters(
        self,
        text: str,
        document_id: str,
        metadata: dict,
    ) -> list[TextChunk]:
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for index, start in enumerate(range(0, len(text), step)):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if not chunk_text.strip():
                continue

            chunks.append(
                TextChunk(
                    id=f"{document_id}:chunk:{index}",
                    text=chunk_text,
                    index=index,
                    metadata={
                        **metadata,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )

            if end >= len(text):
                break

        return chunks

    def _chunk_by_tokens(
        self,
        text: str,
        document_id: str,
        metadata: dict,
    ) -> list[TextChunk]:
        counter = self.token_counter
        assert counter is not None  # guarded by __init__ validation

        total_tokens = counter.count(text)
        if total_tokens == 0:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks = []
        char_start = 0
        token_start = 0
        index = 0

        while char_start < len(text) and token_start < total_tokens:
            # Find the text that fits in chunk_size tokens starting from char_start
            remaining_text = text[char_start:]
            chunk_text = counter.truncate(remaining_text, self.chunk_size)
            if not chunk_text or not chunk_text.strip():
                break

            char_end = char_start + len(chunk_text)
            token_end = token_start + counter.count(chunk_text)

            chunks.append(
                TextChunk(
                    id=f"{document_id}:chunk:{index}",
                    text=chunk_text,
                    index=index,
                    metadata={
                        **metadata,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                    },
                )
            )

            if char_end >= len(text):
                break

            # Advance by step tokens
            advance_text = counter.truncate(remaining_text, step)
            char_start += len(advance_text)
            token_start += step
            index += 1

        return chunks


class PageWindowBuilder:
    def __init__(self, window_size: int = 2, window_overlap: int = 1):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if window_overlap < 0:
            raise ValueError("window_overlap must be >= 0")
        if window_overlap >= window_size:
            raise ValueError("window_overlap must be < window_size")

        self.window_size = window_size
        self.window_overlap = window_overlap

    def build_windows(
        self,
        pages: list[str | dict],
        document_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        metadata = metadata or {}
        chunks = []
        step = self.window_size - self.window_overlap

        for index, start in enumerate(range(0, len(pages), step)):
            end = min(start + self.window_size, len(pages))
            window_pages = pages[start:end]
            text = "\n\n".join(_page_text(page) for page in window_pages)
            if not text.strip():
                continue

            chunks.append(
                TextChunk(
                    id=f"{document_id}:pages:{start + 1}-{end}",
                    text=text,
                    index=index,
                    metadata={
                        **metadata,
                        **_window_metadata(window_pages),
                        "page_start": start + 1,
                        "page_end": end,
                    },
                )
            )

            if end >= len(pages):
                break

        return chunks


def _page_text(page: str | dict) -> str:
    if isinstance(page, str):
        return page
    if isinstance(page, dict):
        return str(page.get("text", ""))
    return str(page)


def _page_metadata(page: str | dict) -> dict[str, Any]:
    if isinstance(page, dict):
        metadata = page.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


def _window_metadata(pages: list[str | dict]) -> dict[str, Any]:
    page_metadata = [_page_metadata(page) for page in pages]
    page_metadata = [metadata for metadata in page_metadata if metadata]
    if not page_metadata:
        return {}

    window_metadata = {}

    record_ids = [
        str(metadata["record_id"])
        for metadata in page_metadata
        if metadata.get("record_id") is not None
    ]
    if record_ids:
        window_metadata["record_ids"] = record_ids

    source_ids = []
    for metadata in page_metadata:
        source_id = (
            metadata.get("source")
            or metadata.get("record_id")
            or metadata.get("id")
        )
        if source_id is not None:
            source_ids.append(str(source_id))
    if source_ids:
        window_metadata["source_ids"] = source_ids

    collections = [
        str(metadata["collection"])
        for metadata in page_metadata
        if metadata.get("collection") is not None
    ]
    if collections:
        window_metadata["collections"] = list(dict.fromkeys(collections))

    return window_metadata
