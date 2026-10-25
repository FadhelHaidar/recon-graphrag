"""Graph database abstraction layer.

Defines the GraphStore protocol that all backends must implement,
plus GraphWriter protocol for write operations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.models.artifacts import CommunityReport
from recon_graphrag.models.types import IndexConfig


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph database operations.

    All SDK components depend on this protocol, not on any specific
    database driver. Implement this to add a new backend.
    """

    def execute_query(
        self, query: str, parameters: dict | None = None
    ) -> list[dict]:
        """Execute a raw query and return results as list of dicts.

        This is an escape hatch for operations that don't yet have a
        semantic method. The query language is backend-specific.
        """
        ...

    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        """Persist a GraphDocument to the store."""
        ...

    def create_indexes(
        self, config: IndexConfig | None = None, embedding_dim: int = 1536
    ) -> None:
        """Create all required vector, fulltext, and uniqueness indexes."""
        ...

    def drop_indexes(self, config: IndexConfig | None = None) -> None:
        """Drop existing indexes so they can be recreated."""
        ...

    async def resolve_entities_exact(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
    ) -> dict:
        """Merge duplicate entity nodes using exact matching."""
        ...

    async def resolve_entities_normalized(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
        *,
        conflict_properties: dict[str, list[str]] | list[str] | None = None,
    ) -> dict:
        """Merge duplicate entity nodes using normalized matching."""
        ...

    async def resolve_entities_fuzzy(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
    ) -> dict:
        """Merge duplicate entity nodes using fuzzy string matching."""
        ...

    async def resolve_entities_hybrid(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
        aliases: dict | None = None,
        embedder: Any | None = None,
        llm: Any | None = None,
        llm_guidance: str | None = None,
        allow_ai_auto_merge: bool = False,
        context_properties: dict[str, list[str]] | list[str] | None = None,
        conflict_properties: dict[str, list[str]] | list[str] | None = None,
        context_mode: str = "safe_defaults",
    ) -> dict:
        """Merge duplicate entity nodes using hybrid fuzzy + embedding + LLM matching."""
        ...

    def create_vector_index(
        self,
        name: str,
        label: str,
        embedding_property: str,
        dimensions: int,
        similarity_fn: str = "cosine",
    ) -> None:
        """Create a vector index."""
        ...

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        """Create a fulltext index."""
        ...

    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
        *,
        content_fingerprints: list[str] | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        """Batch upsert vector embeddings onto nodes by internal ID.

        ``content_fingerprints`` records the text each vector was built from so
        a later change to that text marks the vector stale.
        """
        ...

    def get_entities_needing_summary(
        self, graph_name: str, limit: int = 500
    ) -> list[dict]:
        """Fetch entity rows whose raw descriptions need summarization."""
        ...

    def get_relationships_needing_summary(
        self, graph_name: str, limit: int = 500
    ) -> list[dict]:
        """Fetch relationship rows whose raw descriptions need summarization."""
        ...

    def persist_entity_summaries(
        self, graph_name: str, summaries: list[dict]
    ) -> None:
        """Persist entity summary text and metadata by internal ID."""
        ...

    def persist_relationship_summaries(
        self, graph_name: str, summaries: list[dict]
    ) -> None:
        """Persist relationship summary text and metadata by internal ID."""
        ...

    def vector_search(
        self,
        index_name: str,
        query_vector: list[float],
        k: int,
        label: str | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Vector search returning rows with 'id' and 'score' keys."""
        ...

    def keyword_search(
        self,
        index_name: str,
        query_text: str,
        k: int,
        label: str | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Keyword/fulltext search returning rows with 'id' and 'score' keys."""
        ...

    def fetch_entity_context(
        self,
        matches: list[dict],
        retrieval_query: str | None = None,
        query_params: dict | None = None,
        mode: str = "local",
        graph_name: str | None = None,
    ) -> list[dict]:
        """Fetch formatted entity context for ranked entity matches."""
        ...

    def detect_communities(
        self,
        graph_name: str,
        relationship_types: list[str] | None = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 1e-4,
        relationship_weight_property: str | None = None,
        random_seed: int | None = 42,
        entity_label: str = "__Entity__",
        community_label: str = "Community",
    ) -> list[dict]:
        """Run community detection and return community stats.

        relationship_weight_property is the name of a numeric relationship
        property to use as the Leiden edge weight, e.g. "weight".
        """
        ...

    def get_communities(
        self,
        graph_name: str,
        level: int | None = None,
    ) -> list[dict]:
        """Get communities at a given level."""
        ...

    def get_community_stats(self, graph_name: str) -> list[dict]:
        """Get community statistics."""
        ...

    def store_community_report(
        self,
        report: CommunityReport,
        graph_name: str,
    ) -> None:
        """Store a structured community report."""
        ...

    def mark_community_report_failed(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        error: str,
    ) -> None:
        """Record a failed community report generation attempt."""
        ...

    def get_unembedded_entities(
        self, graph_name: str, limit: int = 500, model_fingerprint: str | None = None
    ) -> list[dict]:
        """Get graph-scoped entities whose stored vector is missing or stale."""
        ...

    def get_community_ranked_context(
        self,
        graph_name: str,
        community_id: str,
        level: int = 0,
    ) -> list[dict]:
        """Fetch degree-ranked entity and relationship rows for a community.

        Returns rows with keys: e_id, e_name, e_description, e_labels, e_degree,
        rel_type, rel_description, observation_count, combined_degree,
        other_id, other_name, other_description, other_labels, other_degree.
        """
        ...

    def get_child_community_reports(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        child_level: int,
    ) -> list[dict]:
        """Fetch child reports for a parent community."""
        ...

    def get_claims_for_entities(
        self,
        graph_name: str,
        entity_ids: list[str],
    ) -> list[dict]:
        """Fetch claims linked to the given entity IDs.

        Returns rows with keys: claim_id, entity_id, claim_type, description,
        status, object_entity_id, source_text, text_unit_id, chunk_id.
        """
        ...

    def resolve_chunk_citations(
        self,
        graph_name: str,
        chunk_ids: list[str],
    ) -> list[dict]:
        """Resolve chunk IDs to citation metadata.

        Returns rows with keys: chunk_id, document_id, document_name,
        page_start, page_end, excerpt, chunk_metadata, document_metadata.
        """
        ...

    def get_community_reports_by_keys(
        self,
        graph_name: str,
        keys: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Fetch reports for graph-scoped community keys."""
        ...

    def count_communities_with_reports(self, graph_name: str) -> int:
        """Count graph-scoped communities that already carry report text."""
        ...

    def get_unembedded_community_reports(
        self,
        graph_name: str,
        limit: int = 500,
    ) -> list[dict]:
        """Get community reports without embeddings.

        Returns rows with keys: id, level, report_text, title.
        """
        ...

    def upsert_community_report_vectors(
        self,
        node_ids: list[str],
        vectors: list[list[float]],
        graph_name: str,
        levels: list[int],
    ) -> None:
        """Batch upsert vector embeddings onto community report nodes."""
        ...

    def vector_search_community_reports(
        self,
        query_vector: list[float],
        graph_name: str,
        top_k: int = 3,
        level: int | None = None,
    ) -> list[dict]:
        """Vector search over community report embeddings.

        Returns rows with keys: id, level, report_text,
        report_json, rating, score.
        """
        ...

    def get_entity_count(self) -> int:
        """Count all __Entity__ nodes."""
        ...

    def get_chunk_count(self) -> int:
        """Count all Chunk nodes."""
        ...

    def get_evidence_link_count(self) -> int:
        """Count all FROM_CHUNK relationships."""
        ...

    def get_relationship_count(self) -> int:
        """Count all entity-to-entity relationships."""
        ...

    def backfill_descriptions(self) -> None:
        """Set description = '' on entities missing the property."""
        ...

    def validate_graph_build(self) -> dict:
        """Return counts of entities, chunks, evidence links, and relationships."""
        ...


@runtime_checkable
class GraphWriter(Protocol):
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        ...
