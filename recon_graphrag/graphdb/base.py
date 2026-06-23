"""Graph database abstraction layer.

Defines the GraphStore protocol that all backends must implement,
plus GraphWriter protocol for write operations.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.models.types import IndexConfig


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph database operations.

    All SDK components depend on this protocol, not on any specific
    database driver. Implement this to add a new backend.
    """

    # ------------------------------------------------------------------
    # Escape hatch — backend-specific query language
    # ------------------------------------------------------------------
    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        """Execute a raw query and return results as list of dicts.

        This is an escape hatch for operations that don't yet have a
        semantic method. The query language is backend-specific.
        """
        ...

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        """Persist a GraphDocument to the store."""
        ...

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    def create_indexes(self, config: IndexConfig, embedding_dim: int) -> None:
        """Create all required vector, fulltext, and uniqueness indexes."""
        ...

    def drop_indexes(self, config: IndexConfig) -> None:
        """Drop existing indexes so they can be recreated."""
        ...

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------
    async def resolve_entities(
        self,
        graph_name: str = "entity-graph",
        strategy: str = "normalized",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
        aliases: Optional[dict] = None,
        embedder: Optional[Any] = None,
        llm: Optional[Any] = None,
        llm_guidance: Optional[str] = None,
        allow_ai_auto_merge: bool = False,
    ) -> dict:
        """Merge duplicate entity nodes when possible."""
        ...

    # ------------------------------------------------------------------
    # Indexes (low-level)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        """Batch upsert vector embeddings onto nodes by internal ID."""
        ...

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def vector_search(
        self,
        index_name: str,
        query_vector: list[float],
        k: int,
        label: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Vector search returning rows with 'id' and 'score' keys."""
        ...

    def keyword_search(
        self,
        index_name: str,
        query_text: str,
        k: int,
        label: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Keyword/fulltext search returning rows with 'id' and 'score' keys."""
        ...

    def fetch_entity_context(
        self,
        matches: list[dict],
        retrieval_query: Optional[str] = None,
        query_params: Optional[dict] = None,
        mode: str = "local",
    ) -> list[dict]:
        """Fetch formatted entity context for ranked entity matches."""
        ...

    # ------------------------------------------------------------------
    # Communities
    # ------------------------------------------------------------------
    def search_communities(
        self,
        index_name: str,
        query_vector: list[float],
        graph_name: str,
        top_k: int,
        level: Optional[int] = None,
    ) -> list[dict]:
        """Vector search on community summary embeddings."""
        ...

    def detect_communities(
        self,
        graph_name: str,
        relationship_types: Optional[list[str]] = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 1e-4,
        relationship_weight_property: Optional[str] = None,
        random_seed: Optional[int] = 42,
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
        level: Optional[int] = None,
    ) -> list[dict]:
        """Get communities at a given level."""
        ...

    def get_community_stats(self, graph_name: str) -> list[dict]:
        """Get community statistics."""
        ...

    def store_community_summary(
        self,
        community_id: str,
        level: int,
        summary: str,
        graph_name: str,
    ) -> None:
        """Store a generated summary on a community node."""
        ...

    def get_unembedded_communities(
        self, graph_name: str, level: int
    ) -> list[dict]:
        """Get communities with summaries but no embeddings."""
        ...

    def get_unembedded_entities(self, limit: int = 500) -> list[dict]:
        """Get entities without embeddings."""
        ...

    def get_community_entity_context(
        self,
        graph_name: str,
        community_id: str,
        level: int = 0,
    ) -> list[dict]:
        """Fetch entity and relationship rows for a community."""
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

    def get_community_child_summary_context(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        child_level: int,
    ) -> list[dict]:
        """Fetch child community summaries for a parent community."""
        ...

    def get_claims_for_entities(
        self,
        graph_name: str,
        entity_ids: list[str],
    ) -> list[dict]:
        """Fetch claims linked to the given entity IDs.

        Returns rows with keys: claim_id, entity_id, claim_type, description,
        status, chunk_id.
        """
        ...

    def resolve_chunk_citations(
        self,
        chunk_ids: list[str],
    ) -> list[dict]:
        """Resolve chunk IDs to citation metadata.

        Returns rows with keys: chunk_id, document_id, document_name,
        page_start, page_end, text (excerpt).
        """
        ...

    def get_community_summaries_by_keys(
        self,
        graph_name: str,
        keys: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Fetch community summaries for graph-scoped community keys."""
        ...

    def get_community_entities_by_keys(
        self,
        graph_name: str,
        keys: list[dict],
    ) -> list[dict]:
        """Fetch entities and relationships for graph-scoped community keys."""
        ...

    # ------------------------------------------------------------------
    # Stats / validation
    # ------------------------------------------------------------------
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
