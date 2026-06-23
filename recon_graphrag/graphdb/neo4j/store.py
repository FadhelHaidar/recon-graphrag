"""Neo4j graph store backend."""

from __future__ import annotations

from typing import Optional

import neo4j

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.neo4j.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.graphdb.neo4j.index_manager import IndexManager
from recon_graphrag.models.types import IndexConfig
from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter


class Neo4jGraphStore:
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
            CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
            YIELD node AS community, score
            WHERE community.graph_name = $graph_name
              AND community.level = $level
              AND community.summary IS NOT NULL
            RETURN community.id AS id,
                   community.summary AS summary,
                   community.level AS level,
                   score
            ORDER BY score DESC
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
            CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
            YIELD node AS community, score
            WHERE community.graph_name = $graph_name
              AND community.summary IS NOT NULL
            RETURN community.id AS id,
                   community.summary AS summary,
                   community.level AS level,
                   score
            ORDER BY score DESC
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

    def get_communities(
        self,
        graph_name: str,
        level: Optional[int] = None,
    ) -> list[dict]:
        community_label = escape_cypher_identifier("Community")
        if level is not None:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name, level: $level}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 count(DISTINCT e) AS entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY entity_count DESC
            """
            params = {"graph_name": graph_name, "level": level}
        else:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 count(DISTINCT e) AS entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY entity_count DESC
            """
            params = {"graph_name": graph_name}
        return self.execute_query(query, params)

    def get_community_stats(self, graph_name: str) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
        RETURN c.id AS community_id,
               c.level AS level,
               count(DISTINCT e) AS entity_count,
               count(DISTINCT child) AS child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.execute_query(query, {"graph_name": graph_name})

    def store_community_summary(
        self,
        community_id: str,
        level: int,
        summary: str,
        graph_name: str,
    ) -> None:
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.summary = $summary,
            c.embedding = NULL,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "summary": summary,
            },
        )

    def get_unembedded_communities(
        self, graph_name: str, level: int
    ) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE c.summary IS NOT NULL AND c.embedding IS NULL
        RETURN elementId(c) AS id,
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
        RETURN elementId(e) AS id, labels(e) AS labels,
               e.name AS name,
               CASE WHEN e.description IS NOT NULL THEN e.description ELSE '' END AS description
        LIMIT $limit
        """
        return self.execute_query(query, {"limit": limit})

    def get_community_entity_context(
        self,
        graph_name: str,
        community_id: str,
        level: int = 0,
    ) -> list[dict]:
        from recon_graphrag.retrieval.neo4j.queries import (
            COMMUNITY_ENTITY_CONTEXT_QUERY,
        )

        return self.execute_query(
            COMMUNITY_ENTITY_CONTEXT_QUERY,
            {"graph_name": graph_name, "cid": community_id, "level": level},
        )

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
    # Claims
    # ------------------------------------------------------------------
    def get_claims_for_entities(
        self,
        graph_name: str,
        entity_ids: list[str],
    ) -> list[dict]:
        if not entity_ids:
            return []
        query = """
        UNWIND $entity_ids AS eid
        MATCH (c:Claim)-[:SUBJECT_OF]->(e:__Entity__ {id: eid})
        OPTIONAL MATCH (c)-[:SOURCED_FROM]->(ch:Chunk)
        RETURN c.id AS claim_id,
               e.id AS entity_id,
               c.claim_type AS claim_type,
               c.description AS description,
               c.status AS status,
               ch.id AS chunk_id
        ORDER BY c.claim_type, c.id
        """
        return self.execute_query(
            query,
            {"entity_ids": entity_ids},
        )

    # ------------------------------------------------------------------
    # Stats / validation
    # ------------------------------------------------------------------
    def get_entity_count(self) -> int:
        result = self.execute_query(
            "MATCH (e:__Entity__) RETURN count(e) AS cnt"
        )
        return result[0]["cnt"] if result else 0

    def get_chunk_count(self) -> int:
        result = self.execute_query(
            "MATCH (c:Chunk) RETURN count(c) AS cnt"
        )
        return result[0]["cnt"] if result else 0

    def get_evidence_link_count(self) -> int:
        result = self.execute_query(
            "MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__) RETURN count(r) AS cnt"
        )
        return result[0]["cnt"] if result else 0

    def get_relationship_count(self) -> int:
        result = self.execute_query(
            "MATCH (:__Entity__)-[r]-(:__Entity__) RETURN count(r) AS cnt"
        )
        return result[0]["cnt"] if result else 0

    def backfill_descriptions(self) -> None:
        self.execute_query(
            "MATCH (e:__Entity__) WHERE e.description IS NULL SET e.description = ''"
        )

    def validate_graph_build(self) -> dict:
        query = """
        CALL {
            MATCH (e:__Entity__)
            RETURN count(e) AS entity_count
        }
        CALL {
            MATCH (c:Chunk)
            RETURN count(c) AS chunk_count
        }
        CALL {
            MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__)
            RETURN count(r) AS evidence_link_count
        }
        CALL {
            MATCH (:__Entity__)-[r]-(:__Entity__)
            RETURN count(r) AS entity_relationship_count
        }
        RETURN entity_count,
               chunk_count,
               evidence_link_count,
               entity_relationship_count
        """
        result = self.execute_query(query)
        return result[0] if result else {}
