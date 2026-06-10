"""Text chunking and sliding page windows for graph extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    id: str
    text: str
    index: int
    metadata: dict


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        metadata = metadata or {}
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
        pages: list[str],
        document_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        metadata = metadata or {}
        chunks = []
        step = self.window_size - self.window_overlap

        for index, start in enumerate(range(0, len(pages), step)):
            end = min(start + self.window_size, len(pages))
            text = "\n\n".join(pages[start:end])
            if not text.strip():
                continue

            chunks.append(
                TextChunk(
                    id=f"{document_id}:pages:{start + 1}-{end}",
                    text=text,
                    index=index,
                    metadata={
                        **metadata,
                        "page_start": start + 1,
                        "page_end": end,
                    },
                )
            )

            if end >= len(pages):
                break

        return chunks
