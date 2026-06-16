"""FalkorDB graph store backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from falkordb import FalkorDB

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.falkordb.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.graphdb.falkordb.index_manager import IndexManager
from recon_graphrag.models.types import IndexConfig
from recon_graphrag.pipelines.falkordb.writer import FalkorDBGraphWriter


class FalkorDBGraphStore:
    """FalkorDB backend backed by the official falkordb client."""

    def __init__(
        self,
        client: FalkorDB,
        graph_name: str = "entity-graph",
    ):
        self._client = client
        self._graph_name = graph_name
        self._graph = client.select_graph(graph_name)
        self._vector_index_registry: dict[str, tuple[str, str]] = {}
        self._fulltext_index_registry: dict[str, str] = {}

    @property
    def client(self) -> FalkorDB:
        """Return the underlying falkordb.FalkorDB client.

        This is NOT part of the GraphStore protocol; it is FalkorDB-specific.
        """
        return self._client

    @property
    def graph(self) -> Any:
        """Return the selected falkordb Graph handle."""
        return self._graph

    @property
    def graph_name(self) -> str:
        """Return the logical graph name used for multi-tenancy filtering."""
        return self._graph_name

    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        result = self._graph.query(query, parameters or {})
        if not result.result_set:
            return []
        columns = [entry[1] for entry in result.header]
        return [
            {col: row[i] for i, col in enumerate(columns)}
            for row in result.result_set
        ]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        writer = FalkorDBGraphWriter(self)
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
        from recon_graphrag.graphdb.falkordb.entity_resolution import (
            _FalkorDBEntityResolver,
        )

        resolver = _FalkorDBEntityResolver(self)
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
        self._vector_index_registry[name] = (label, embedding_property)
        query = f"""
        CREATE VECTOR INDEX FOR (n:{escape_cypher_identifier(label)})
        ON (n.{escape_cypher_identifier(embedding_property)})
        OPTIONS {{
          dimension: {int(dimensions)},
          similarityFunction: {cypher_string_literal(similarity_fn)},
          M: 32,
          efConstruction: 200
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
        self._fulltext_index_registry[name] = label
        props = ", ".join(
            f"{cypher_string_literal(prop)}" for prop in node_properties
        )
        # FalkorDB exposes fulltext index creation via procedure.
        query = f"CALL db.idx.fulltext.createNodeIndex({cypher_string_literal(label)}, {props})"
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
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE id(n) = row.id
            SET n.embedding = vecf32(row.vector)
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
        resolved_label, prop = self._vector_index_registry.get(
            index_name, (label or "__Entity__", "embedding")
        )
        if label:
            resolved_label = label

        query = f"""
        CALL db.idx.vector.queryNodes({cypher_string_literal(resolved_label)}, {cypher_string_literal(prop)}, $k, vecf32($query_vector))
        YIELD node, score
        RETURN id(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.execute_query(
            query,
            {
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
        resolved_label = self._fulltext_index_registry.get(index_name, label or "__Entity__")
        if label:
            resolved_label = label

        query = f"""
        CALL db.idx.fulltext.queryNodes({cypher_string_literal(resolved_label)}, $query_text)
        YIELD node, score
        RETURN id(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.execute_query(
            query,
            {
                "query_text": query_text,
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
        from recon_graphrag.retrieval.falkordb.queries import (
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
        WHERE id(node) = match.id
        WITH node, match.score AS score
        {retrieval_query}
        """
        parameters = {"matches": matches}
        if query_params:
            for key, value in query_params.items():
                parameters.setdefault(key, value)
        rows = self.execute_query(query, parameters)
        return self._aggregate_context_rows(rows)

    @staticmethod
    def _aggregate_context_rows(rows: list[dict]) -> list[dict]:
        """Aggregate flat retrieval rows into the structure expected by retrievers.

        FalkorDB retrieval queries return one row per relationship. This method
        groups rows by entity title and collects relationships/source_texts.
        """
        grouped: dict[str, dict] = {}
        for row in rows:
            title = row.get("title") or ""
            if title not in grouped:
                grouped[title] = {
                    "title": title,
                    "relationships": [],
                    "source_text": [],
                    "score": row.get("score"),
                    "communities": [],
                }

            rel = row.get("relationship")
            if rel and rel not in grouped[title]["relationships"]:
                grouped[title]["relationships"].append(rel)

            source_text = row.get("source_text")
            if source_text:
                if isinstance(source_text, list):
                    for st in source_text:
                        if st and st not in grouped[title]["source_text"]:
                            grouped[title]["source_text"].append(st)
                elif source_text not in grouped[title]["source_text"]:
                    grouped[title]["source_text"].append(source_text)

            community_id = row.get("community_id")
            if community_id is not None:
                comm = {
                    "id": community_id,
                    "level": row.get("community_level"),
                    "graph_name": row.get("community_graph_name"),
                    "summary": row.get("community_summary"),
                }
                if comm not in grouped[title]["communities"]:
                    grouped[title]["communities"].append(comm)

        return list(grouped.values())

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
        resolved_label, prop = self._vector_index_registry.get(
            index_name, ("Community", "embedding")
        )
        overfetch_k = max(top_k * 5, top_k)
        label_literal = cypher_string_literal(resolved_label)
        prop_literal = cypher_string_literal(prop)

        if level is not None:
            query = f"""
            CALL db.idx.vector.queryNodes({label_literal}, {prop_literal}, $k, vecf32($query_vector))
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
                "k": overfetch_k,
                "top_k": top_k,
                "query_vector": query_vector,
                "level": level,
                "graph_name": graph_name,
            }
        else:
            query = f"""
            CALL db.idx.vector.queryNodes({label_literal}, {prop_literal}, $k, vecf32($query_vector))
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
        from recon_graphrag.communities.falkordb.detection import CommunityDetector

        detector = CommunityDetector(
            self,
            relationship_types=relationship_types,
            graph_name=graph_name,
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
            WITH c,
                 count(DISTINCT e) AS entity_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   0 AS child_community_count
            ORDER BY entity_count DESC
            """
            params = {"graph_name": graph_name, "level": level}
        else:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c,
                 count(DISTINCT e) AS entity_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   0 AS child_community_count
            ORDER BY entity_count DESC
            """
            params = {"graph_name": graph_name}
        return self.execute_query(query, params)

    def get_community_stats(self, graph_name: str) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        RETURN c.id AS community_id,
               c.level AS level,
               count(DISTINCT e) AS entity_count,
               0 AS child_community_count
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

    def get_community_entity_context(
        self,
        graph_name: str,
        community_id: str,
        level: int = 0,
    ) -> list[dict]:
        from recon_graphrag.retrieval.falkordb.queries import (
            COMMUNITY_ENTITY_CONTEXT_QUERY,
        )

        return self.execute_query(
            COMMUNITY_ENTITY_CONTEXT_QUERY,
            {"graph_name": graph_name, "cid": community_id, "level": level},
        )

    def get_community_child_summary_context(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        child_level: int,
    ) -> list[dict]:
        from recon_graphrag.retrieval.falkordb.queries import (
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
        from recon_graphrag.retrieval.falkordb.queries import (
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
        from recon_graphrag.retrieval.falkordb.queries import (
            DRIFT_COMMUNITY_ENTITIES_QUERY,
        )

        return self.execute_query(
            DRIFT_COMMUNITY_ENTITIES_QUERY,
            {"keys": keys, "graph_name": graph_name},
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
        return {
            "entity_count": self.get_entity_count(),
            "chunk_count": self.get_chunk_count(),
            "evidence_link_count": self.get_evidence_link_count(),
            "entity_relationship_count": self.get_relationship_count(),
        }
