"""Text chunking and sliding page windows for graph extraction."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from recon_graphrag.utils.tokens import TiktokenTokenCounter, TokenCounter


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
    ``token_end`` using accumulated caller counts (substring tokenization may
    differ from the full text). Boundaries preserve codepoints, not grapheme
    clusters. Overlap is rounded inward when necessary for forward progress;
    an unrepresentable nonempty chunk budget raises ``ValueError``.
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

        # Only the built-in exact counter promises a byte-boundary index.
        # Custom encode/decode methods need not preserve substrings or counts.
        boundaries: tuple[list[int], list[int]] | None = None
        if (
            isinstance(counter, TiktokenTokenCounter)
            and type(counter) is TiktokenTokenCounter
        ):
            try:
                boundaries = counter._token_boundaries(text)
            except UnicodeEncodeError:
                # Python strings can contain lone surrogates, unlike UTF-8.
                boundaries = None

        chunks: list[TextChunk] = []
        char_start = 0
        token_start = 0
        while char_start < len(text):
            if boundaries is not None:
                token_offsets, char_offsets = boundaries
                first_token = bisect_right(token_offsets, char_offsets[char_start]) - 1
                last_token = min(first_token + self.chunk_size, len(token_offsets) - 1)
                char_end = bisect_right(char_offsets, token_offsets[last_token]) - 1
                # Verify using the caller's actual count, not full-text token IDs:
                # re-encoding a substring can change its tokenization.
                char_end = max(char_start + 1, char_end)
                while (
                    char_end > char_start
                    and counter.count(text[char_start:char_end]) > self.chunk_size
                ):
                    char_end -= 1
            else:
                prefix = _token_prefix(counter, text[char_start:], self.chunk_size)
                char_end = char_start + len(prefix)

            if char_end == char_start:
                raise ValueError(
                    f"chunk_size={self.chunk_size} cannot fit a source prefix "
                    f"at character offset {char_start}"
                )

            chunk_text = text[char_start:char_end]
            chunk_tokens = counter.count(chunk_text)
            if chunk_text.strip():
                index = len(chunks)
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
                            "token_end": token_start + chunk_tokens,
                        },
                    )
                )
            if char_end == len(text):
                break

            if not self.chunk_overlap:
                next_start = char_end
            elif boundaries is not None:
                token_offsets, char_offsets = boundaries
                # Round overlap inward to original codepoints, then verify the
                # suffix budget. Exact overlap may be unrepresentable (emoji).
                last_token = bisect_left(token_offsets, char_offsets[char_end])
                overlap_token = max(0, last_token - self.chunk_overlap)
                next_start = bisect_left(char_offsets, token_offsets[overlap_token])
                next_start = min(char_end, max(char_start + 1, next_start))
            else:
                step = self.chunk_size - self.chunk_overlap
                advance = _token_prefix(counter, chunk_text, step)
                next_start = min(char_end, char_start + max(1, len(advance)))

            while (
                next_start < char_end
                and counter.count(text[next_start:char_end]) > self.chunk_overlap
            ):
                next_start += 1
            token_start += counter.count(text[char_start:next_start])
            char_start = next_start

        return chunks


def _token_prefix(counter: TokenCounter, text: str, budget: int) -> str:
    """Validate custom truncation without assuming monotonic prefix counts."""
    # Legacy counters may normalize whitespace or decode partial token bytes.
    # Use their result only as a boundary hint, then slice the original text.
    end = min(len(counter.truncate(text, budget)), len(text))
    if not end and counter.count(text) <= budget:
        return text
    while end and counter.count(text[:end]) > budget:
        end -= 1
    return text[:end]


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
        pages: Sequence[str | dict],
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


def _window_metadata(pages: Sequence[str | dict]) -> dict[str, Any]:
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
