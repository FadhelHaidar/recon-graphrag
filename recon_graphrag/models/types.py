"""Shared data types for the SDK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from any search mode."""

    query: str
    mode: str  # "local", "global", "drift"
    answer: str
    context: str = ""

    def __str__(self) -> str:
        return f"[{self.mode.upper()}] {self.answer}"


@dataclass
class IndexConfig:
    """Configuration for Neo4j vector and fulltext indexes."""

    chunk_vector_index: str = "chunk-embeddings"
    entity_vector_index: str = "entity-embeddings"
    community_vector_index: str = "community-embeddings"
    entity_fulltext_index: str = "entity-names"
    entity_label: str = "__Entity__"
    chunk_label: str = "Chunk"
    community_label: str = "Community"
