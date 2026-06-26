"""Neo4j graph store backend."""

from __future__ import annotations

from typing import Optional

import neo4j

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.neo4j.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.graphdb.neo4j.index_manager import IndexManager
from recon_graphrag.models.types import IndexConfig
from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter
from recon_graphrag.graphdb.store_base import BaseGraphStore


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j backend backed by the official Neo4j driver."""

    def __init__(
        self,
        driver: neo4j.Driver,
        database: Optional[str] = None,
    ):
        self._driver = driver
        self._database = database

    @property
    def driver(self) -> neo4j.Driver:
        """Return the underlying neo4j.Driver for advanced Neo4j integrations.

        This is NOT part of the GraphStore protocol; it is Neo4j-specific.
        """
        return self._driver

    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        with self._driver.session(
            database=self._database,
            notifications_disabled_categories=["DEPRECATION", "UNRECOGNIZED", "HINT"],
        ) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        writer = Neo4jGraphWriter(self)
        return writer.write_graph_document(graph_document)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    def create_indexes(self, config: IndexConfig, embedding_dim: int) -> None:
        mgr = IndexManager(self, embedding_dim=embedding_dim, index_config=config)
        mgr.create_indexes()

    def drop_indexes(self, config: IndexConfig) -> None:
        mgr = IndexManager(self, index_config=config)
        mgr._drop_indexes()

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
        embedder=None,
        llm=None,
        llm_guidance: Optional[str] = None,
        allow_ai_auto_merge: bool = False,
        context_properties: Optional[dict[str, list[str]] | list[str]] = None,
        conflict_properties: Optional[dict[str, list[str]] | list[str]] = None,
        context_mode: str = "safe_defaults",
    ) -> dict:
        from recon_graphrag.graphdb.neo4j.entity_resolution import (
            _Neo4jEntityResolver,
        )

        resolver = _Neo4jEntityResolver(self)
        return await resolver.resolve(
            graph_name=graph_name,
            strategy=strategy,
            resolve_property=resolve_property,
            dry_run=dry_run,
            merge_threshold=merge_threshold,
            review_threshold=review_threshold,
            max_candidates_per_entity=max_candidates_per_entity,
            aliases=aliases,
            embedder=embedder,
            llm=llm,
            llm_guidance=llm_guidance,
            allow_ai_auto_merge=allow_ai_auto_merge,
            context_properties=context_properties,
            conflict_properties=conflict_properties,
            context_mode=context_mode,
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def create_vector_index(
        self,
        name: str,
        label: str,
        embedding_property: str,
        dimensions: int,
        similarity_fn: str = "cosine",
    ) -> None:
        query = f"""
        CREATE VECTOR INDEX {escape_cypher_identifier(name)} IF NOT EXISTS
        FOR (n:{escape_cypher_identifier(label)})
        ON (n.{escape_cypher_identifier(embedding_property)})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {int(dimensions)},
            `vector.similarity_function`: {cypher_string_literal(similarity_fn)}
          }}
        }}
        """
        self.execute_query(query)

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        properties = ", ".join(
            f"n.{escape_cypher_identifier(prop)}" for prop in node_properties
        )
        query = f"""
        CREATE FULLTEXT INDEX {escape_cypher_identifier(name)} IF NOT EXISTS
        FOR (n:{escape_cypher_identifier(label)})
        ON EACH [{properties}]
        """
        self.execute_query(query)

    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        """Batch upsert vector embeddings onto nodes matched by elementId().

        This intentionally follows the current embedding call path, which reads
        elementId() values immediately before writing embeddings back.
        """
        if len(node_ids) != len(vectors):
            raise ValueError("node_ids and vectors must have the same length")
        if not node_ids:
            return

        rows = [
            {"id": node_id, "vector": vector}
            for node_id, vector in zip(node_ids, vectors)
        ]
        self.execute_query(
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE elementId(n) = row.id
            CALL db.create.setNodeVectorProperty(n, $embedding_property, row.vector)
            RETURN count(n) AS updated_count
            """,
            {"rows": rows, "embedding_property": embedding_property},
        )

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
        label_filter = ""
        if label:
            label_filter = f"WHERE node:{escape_cypher_identifier(label)}"
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
        YIELD node, score
        {label_filter}
        RETURN elementId(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.execute_query(
            query,
            {
                "index_name": index_name,
                "k": k,
                "top_k": k,
                "query_vector": query_vector,
            },
        )

    def keyword_search(
        self,
        index_name: str,
        query_text: str,
        k: int,
        label: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        label_filter = ""
        if label:
            label_filter = f"WHERE node:{escape_cypher_identifier(label)}"
        query = f"""
        CALL db.index.fulltext.queryNodes(
            $index_name,
            $query_text,
            {{limit: $k}}
        )
        YIELD node, score
        {label_filter}
        RETURN elementId(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.execute_query(
            query,
            {
                "index_name": index_name,
                "query_text": query_text,
                "k": k,
                "top_k": k,
            },
        )

    def fetch_entity_context(
        self,
        matches: list[dict],
        retrieval_query: Optional[str] = None,
        query_params: Optional[dict] = None,
        mode: str = "local",
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            DEFAULT_DRIFT_RETRIEVAL_QUERY,
            DEFAULT_LOCAL_RETRIEVAL_QUERY,
        )

        if retrieval_query is None:
            if mode == "drift":
                retrieval_query = DEFAULT_DRIFT_RETRIEVAL_QUERY
            else:
                retrieval_query = DEFAULT_LOCAL_RETRIEVAL_QUERY

        query = f"""
        UNWIND $matches AS match
        MATCH (node)
        WHERE elementId(node) = match.id
        WITH node, match.score AS score
        {retrieval_query}
        """
        parameters = {"matches": matches}
        if query_params:
            for key, value in query_params.items():
                parameters.setdefault(key, value)
        return self.execute_query(query, parameters)

    # ------------------------------------------------------------------
    # Communities
    # ------------------------------------------------------------------
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
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        detector = CommunityDetector(
            self,
            relationship_types=relationship_types,
            max_levels=max_levels,
            gamma=gamma,
            theta=theta,
            tolerance=tolerance,
            graph_name=graph_name,
            relationship_weight_property=relationship_weight_property,
            random_seed=random_seed,
            entity_label=entity_label,
            community_label=community_label,
        )
        return detector.detect()

    def get_unembedded_entities(self, limit: int = 500) -> list[dict]:
        query = """
        MATCH (e:__Entity__)
        WHERE e.embedding IS NULL
        RETURN elementId(e) AS id, labels(e) AS labels,
               e.name AS name,
               CASE WHEN e.description IS NOT NULL THEN e.description ELSE '' END AS description
        LIMIT $limit
        """
        return self.execute_query(query, {"limit": limit})

    def get_community_ranked_context(
        self,
        graph_name: str,
        community_id: str,
        level: int = 0,
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            COMMUNITY_RANKED_CONTEXT_QUERY,
        )

        return self.execute_query(
            COMMUNITY_RANKED_CONTEXT_QUERY,
            {"graph_name": graph_name, "cid": community_id, "level": level},
        )

    def get_community_child_summary_context(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        child_level: int,
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            COMMUNITY_CHILD_SUMMARY_QUERY,
        )

        return self.execute_query(
            COMMUNITY_CHILD_SUMMARY_QUERY,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "child_level": child_level,
            },
        )

    def get_community_summaries_by_keys(
        self,
        graph_name: str,
        keys: list[dict],
        top_k: int,
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            DRIFT_COMMUNITY_SUMMARIES_QUERY,
        )

        return self.execute_query(
            DRIFT_COMMUNITY_SUMMARIES_QUERY,
            {"keys": keys, "graph_name": graph_name, "top_k": top_k},
        )

    def get_community_entities_by_keys(
        self,
        graph_name: str,
        keys: list[dict],
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            DRIFT_COMMUNITY_ENTITIES_QUERY,
        )

        return self.execute_query(
            DRIFT_COMMUNITY_ENTITIES_QUERY,
            {"keys": keys, "graph_name": graph_name},
        )

    # ------------------------------------------------------------------
    # Stats / validation
    # ------------------------------------------------------------------
    def backfill_descriptions(self) -> None:
        self.execute_query(
            "MATCH (e:__Entity__) WHERE e.description IS NULL SET e.description = ''"
        )
