"""Shared data types for the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field

from recon_graphrag.models.artifacts import Citation, DocumentSource


@dataclass
class SearchResult:
    """Result from any search mode."""

    query: str
    mode: str  # "local", "global", "drift"
    answer: str
    context: str = ""
    citations: list[Citation] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.mode.upper()}] {self.answer}"

    @property
    def sources(self) -> list[DocumentSource]:
        """Group citations by document for user-facing display."""
        by_doc: dict[str, list[Citation]] = {}
        for c in self.citations:
            by_doc.setdefault(c.document_id, []).append(c)
        return [
            DocumentSource(
                document_id=doc_id,
                document_name=chunk_list[0].document_name if chunk_list else None,
                chunk_list=chunk_list,
            )
            for doc_id, chunk_list in by_doc.items()
        ]


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
