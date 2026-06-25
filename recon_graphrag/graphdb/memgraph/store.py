"""Memgraph graph store backend."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import neo4j

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.memgraph.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.graphdb.memgraph.index_manager import IndexManager
from recon_graphrag.models.types import IndexConfig
from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter
from recon_graphrag.graphdb.store_base import BaseGraphStore


def _escape_tantivy_term(value: str) -> str:
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", value)


def _format_tantivy_query(query_text: str) -> str:
    """Format user text as a Tantivy query for Memgraph text indexes."""
    phrases = [
        match.replace("\\", "\\\\").replace('"', '\\"').strip()
        for match in re.findall(r'"([^"]+)"', query_text)
        if match.strip()
    ]
    without_phrases = re.sub(r'"[^"]+"', " ", query_text)
    tokens = [
        _escape_tantivy_term(token)
        for token in re.findall(r"[A-Za-z0-9_]+", without_phrases)
    ]

    parts = [f'"{phrase}"' for phrase in phrases]
    parts.extend(f'"{token}"' for token in tokens)
    if parts:
        return " OR ".join(parts)

    escaped = query_text.replace("\\", "\\\\").replace('"', '\\"').strip()
    return f'"{escaped}"'


class MemgraphGraphStore(BaseGraphStore):
    """Memgraph backend backed by a Bolt-compatible driver (e.g., neo4j.Driver)."""

    def __init__(
        self,
        driver: neo4j.Driver,
        database: Optional[str] = None,
    ):
        self._driver = driver
        self._database = database

    @property
    def driver(self) -> neo4j.Driver:
        """Return the underlying driver for advanced Memgraph integrations.

        This is NOT part of the GraphStore protocol; it is Memgraph-specific.
        """
        return self._driver

    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        with self._driver.session(
            database=self._database,
        ) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        writer = MemgraphGraphWriter(self)
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
        from recon_graphrag.graphdb.memgraph.entity_resolution import (
            _MemgraphEntityResolver,
        )

        resolver = _MemgraphEntityResolver(self)
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
        metric = "cos" if similarity_fn == "cosine" else similarity_fn
        query = f"""
        CREATE VECTOR INDEX {escape_cypher_identifier(name)}
        ON :{escape_cypher_identifier(label)}({escape_cypher_identifier(embedding_property)})
        WITH CONFIG {{
          dimension: {int(dimensions)},
          capacity: 10000,
          metric: {cypher_string_literal(metric)}
        }}
        """
        try:
            self.execute_query(query)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        if node_properties:
            props = ", ".join(
                escape_cypher_identifier(prop) for prop in node_properties
            )
            query = f"""
            CREATE TEXT INDEX {escape_cypher_identifier(name)}
            ON :{escape_cypher_identifier(label)}({props})
            """
        else:
            query = f"""
            CREATE TEXT INDEX {escape_cypher_identifier(name)}
            ON :{escape_cypher_identifier(label)}
            """
        try:
            self.execute_query(query)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        """Batch upsert vector embeddings onto nodes matched by id()."""
        if len(node_ids) != len(vectors):
            raise ValueError("node_ids and vectors must have the same length")
        if not node_ids:
            return

        rows = [
            {"id": node_id, "vector": vector}
            for node_id, vector in zip(node_ids, vectors)
        ]
        self.execute_query(
            f"""
            UNWIND $rows AS row
            MATCH (n)
            WHERE id(n) = row.id
            SET n.{escape_cypher_identifier(embedding_property)} = row.vector
            RETURN count(n) AS updated_count
            """,
            {"rows": rows},
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
        overfetch_k = max(k * 5, k) if label else k
        label_filter = ""
        if label:
            label_filter = f"WHERE node:{escape_cypher_identifier(label)}"
        query = f"""
        CALL vector_search.search({cypher_string_literal(index_name)}, $k, $query_vector)
        YIELD node, similarity
        {label_filter}
        RETURN id(node) AS id, similarity AS score
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        return self.execute_query(
            query,
            {
                "k": overfetch_k,
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
        formatted_query = _format_tantivy_query(query_text)
        overfetch_k = max(k * 5, k) if label else k
        label_filter = ""
        if label:
            label_filter = f"WHERE node:{escape_cypher_identifier(label)}"
        query = f"""
        CALL text_search.search({cypher_string_literal(index_name)}, $query_text, $k)
        YIELD node, score
        {label_filter}
        RETURN id(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        try:
            return self.execute_query(
                query,
                {
                    "query_text": formatted_query,
                    "k": overfetch_k,
                    "top_k": k,
                },
            )
        except Exception as exc:
            message = str(exc).lower()
            if "text_search.search" in message or "tantivy error" in message:
                return []
            raise

    def fetch_entity_context(
        self,
        matches: list[dict],
        retrieval_query: Optional[str] = None,
        query_params: Optional[dict] = None,
        mode: str = "local",
    ) -> list[dict]:
        from recon_graphrag.retrieval.memgraph.queries import (
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
        WHERE id(node) = toInteger(match.id)
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
    def search_communities(
        self,
        index_name: str,
        query_vector: list[float],
        graph_name: str,
        top_k: int,
        level: Optional[int] = None,
    ) -> list[dict]:
        overfetch_k = max(top_k * 5, top_k)
        if level is not None:
            query = """
            CALL vector_search.search($index_name, $k, $query_vector)
            YIELD node AS community, similarity
            WHERE community.graph_name = $graph_name
              AND community.level = $level
              AND community.summary IS NOT NULL
            RETURN community.id AS id,
                   community.summary AS summary,
                   community.level AS level,
                   similarity AS score
            ORDER BY similarity DESC
            LIMIT $top_k
            """
            params = {
                "index_name": index_name,
                "k": overfetch_k,
                "top_k": top_k,
                "query_vector": query_vector,
                "level": level,
                "graph_name": graph_name,
            }
        else:
            query = """
            CALL vector_search.search($index_name, $k, $query_vector)
            YIELD node AS community, similarity
            WHERE community.graph_name = $graph_name
              AND community.summary IS NOT NULL
            RETURN community.id AS id,
                   community.summary AS summary,
                   community.level AS level,
                   similarity AS score
            ORDER BY similarity DESC
            LIMIT $top_k
            """
            params = {
                "index_name": index_name,
                "k": overfetch_k,
                "top_k": top_k,
                "query_vector": query_vector,
                "graph_name": graph_name,
            }
        return self.execute_query(query, params)

    def detect_communities(
        self,
        graph_name: str = "entity-graph",
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
        from recon_graphrag.communities.memgraph.detection import CommunityDetector

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

    def get_unembedded_communities(
        self, graph_name: str, level: int
    ) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE c.summary IS NOT NULL AND c.embedding IS NULL
        RETURN id(c) AS id,
               c.id AS community_id,
               c.level AS level,
               c.summary AS summary
        """
        return self.execute_query(
            query, {"graph_name": graph_name, "level": level}
        )

    def get_unembedded_entities(self, limit: int = 500) -> list[dict]:
        query = """
        MATCH (e:__Entity__)
        WHERE e.embedding IS NULL
        RETURN id(e) AS id, labels(e) AS labels,
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
        from recon_graphrag.retrieval.memgraph.queries import (
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
        from recon_graphrag.retrieval.memgraph.queries import (
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
        from recon_graphrag.retrieval.memgraph.queries import (
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
        from recon_graphrag.retrieval.memgraph.queries import (
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

    def _extra_validation_count_queries(self) -> dict[str, str]:
        return {
            "community_count": "MATCH (c:Community) RETURN count(c) AS cnt",
            "community_summary_count": (
                "MATCH (c:Community) WHERE c.summary IS NOT NULL "
                "RETURN count(c) AS cnt"
            ),
            "entity_self_loop_count": (
                "MATCH (e:__Entity__)-[r]-(e) "
                "WHERE type(r) <> 'IN_COMMUNITY' "
                "RETURN count(DISTINCT r) AS cnt"
            ),
        }
