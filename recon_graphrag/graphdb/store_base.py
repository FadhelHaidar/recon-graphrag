"""Shared graph store helpers.

Concrete backends keep their own packages and dialect-specific operations.
This base only contains backend-neutral Cypher-shaped helpers whose return
shape is part of the shared GraphStore contract.
"""

from __future__ import annotations

from recon_graphrag.models.artifacts import (
    CommunityReport,
    report_to_json,
    report_to_text,
)
from recon_graphrag.graphdb.cypher import escape_cypher_identifier


class BaseGraphStore:
    """Mixin for graph-store methods that are identical across backends."""

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        raise NotImplementedError

    def get_entity_count(self) -> int:
        return self._count("MATCH (e:__Entity__) RETURN count(e) AS cnt")

    def get_chunk_count(self) -> int:
        return self._count("MATCH (c:Chunk) RETURN count(c) AS cnt")

    def get_evidence_link_count(self) -> int:
        return self._count(
            "MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__) RETURN count(r) AS cnt"
        )

    def get_relationship_count(self) -> int:
        return self._count(
            "MATCH (:__Entity__)-[r]-(:__Entity__) RETURN count(r) AS cnt"
        )

    def validate_graph_build(self) -> dict:
        counts = {
            "entity_count": self.get_entity_count(),
            "chunk_count": self.get_chunk_count(),
            "evidence_link_count": self.get_evidence_link_count(),
            "entity_relationship_count": self.get_relationship_count(),
        }

        for key, query in self._extra_validation_count_queries().items():
            counts[key] = self._count(query)
        return counts

    def _extra_validation_count_queries(self) -> dict[str, str]:
        return {}

    def _count(self, query: str) -> int:
        result = self.execute_query(query)
        return result[0]["cnt"] if result else 0

    def get_community_stats(self, graph_name: str) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, count(DISTINCT e) AS entity_count
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
        WITH c, entity_count, count(DISTINCT child) AS child_community_count
        RETURN c.id AS community_id,
               c.level AS level,
               entity_count,
               child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.execute_query(query, {"graph_name": graph_name})

    def get_communities(
        self,
        graph_name: str,
        level: int | None = None,
    ) -> list[dict]:
        community_label = escape_cypher_identifier("Community")
        if level is not None:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name, level: $level}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, count(DISTINCT e) AS entity_count
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY entity_count DESC, id
            """
            params = {"graph_name": graph_name, "level": level}
        else:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, count(DISTINCT e) AS entity_count
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY c.level, entity_count DESC, id
            """
            params = {"graph_name": graph_name}
        return self.execute_query(query, params)

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

    def store_community_report(
        self,
        report: CommunityReport,
        graph_name: str,
    ) -> None:
        report_text = report_to_text(report)
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.report_json = $report_json,
            c.report_text = $report_text,
            c.title = $title,
            c.summary = $report_text,
            c.rating = $rating,
            c.rating_explanation = $rating_explanation,
            c.report_status = 'success',
            c.report_error = NULL,
            c.schema_version = $schema_version,
            c.prompt_version = $prompt_version,
            c.context_tokens_used = $context_tokens_used,
            c.context_truncated = $context_truncated,
            c.updated = timestamp()
        WITH c
        WHERE c.input_fingerprint <> $input_fingerprint
           OR c.input_fingerprint IS NULL
        SET c.input_fingerprint = $input_fingerprint,
            c.report_embedding = NULL
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": report.community_id,
                "level": report.level,
                "report_json": report_to_json(report),
                "report_text": report_text,
                "title": report.title,
                "rating": report.rating,
                "rating_explanation": report.rating_explanation,
                "schema_version": report.version.schema_version,
                "prompt_version": report.version.prompt_version,
                "input_fingerprint": report.version.input_fingerprint,
                "context_tokens_used": report.context_tokens_used,
                "context_truncated": report.context_truncated,
            },
        )

    def mark_community_report_failed(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        error: str,
    ) -> None:
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.report_status = 'failed',
            c.report_error = $error,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "error": error,
            },
        )

    def get_claims_for_entities(
        self,
        graph_name: str,
        entity_ids: list[str],
    ) -> list[dict]:
        if not entity_ids:
            return []
        query = """
        UNWIND $entity_ids AS eid
        MATCH (c:Claim {graph_name: $graph_name})-[:SUBJECT_OF]->
              (e:__Entity__ {graph_name: $graph_name})
        WHERE e.id = eid OR e.canonical_key = eid OR e.human_readable_id = eid
        OPTIONAL MATCH (c)-[:SOURCED_FROM]->(ch:Chunk {graph_name: $graph_name})
        RETURN c.id AS claim_id,
               coalesce(e.human_readable_id, e.canonical_key, e.id) AS entity_id,
               c.claim_type AS claim_type,
               c.description AS description,
               c.status AS status,
               ch.id AS chunk_id
        ORDER BY c.claim_type, c.id
        """
        return self.execute_query(
            query,
            {"graph_name": graph_name, "entity_ids": entity_ids},
        )

    def resolve_chunk_citations(
        self,
        graph_name: str,
        chunk_ids: list[str],
    ) -> list[dict]:
        if not chunk_ids:
            return []
        query = """
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {id: cid, graph_name: $graph_name})
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Document {graph_name: $graph_name})
        RETURN c.id AS chunk_id,
               d.id AS document_id,
               coalesce(d.title, d.source, d.filename) AS document_name,
               c.page_start AS page_start,
               c.page_end AS page_end,
               properties(c) AS chunk_metadata,
               properties(d) AS document_metadata,
               substring(c.text, 0, 200) AS excerpt
        """
        return self.execute_query(
            query,
            {"graph_name": graph_name, "chunk_ids": chunk_ids},
        )

    def get_unembedded_community_reports(
        self,
        graph_name: str,
        limit: int = 500,
    ) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE c.report_embedding IS NULL
          AND coalesce(c.report_text, c.summary, '') <> ''
        RETURN c.id AS id,
               c.level AS level,
               coalesce(c.report_text, c.summary) AS report_text,
               coalesce(c.report_text, c.summary) AS summary,
               coalesce(c.title, '') AS title
        LIMIT $limit
        """
        return self.execute_query(
            query, {"graph_name": graph_name, "limit": limit}
        )

    def upsert_community_report_vectors(
        self,
        node_ids: list[str],
        vectors: list[list[float]],
    ) -> None:
        if not node_ids:
            return
        query = """
        UNWIND $pairs AS pair
        MATCH (c:Community)
        WHERE c.id = pair.node_id
        SET c.report_embedding = pair.vector
        """
        pairs = [
            {"node_id": nid, "vector": vec}
            for nid, vec in zip(node_ids, vectors)
        ]
        self.execute_query(query, {"pairs": pairs})

    def vector_search_community_reports(
        self,
        query_vector: list[float],
        graph_name: str,
        top_k: int = 3,
        level: int | None = None,
    ) -> list[dict]:
        if level is not None:
            query = """
            CALL db.index.vector.queryNodes(
                $index_name, $top_k, $query_vector
            ) YIELD node AS c, score
            WHERE c.graph_name = $graph_name AND c.level = $level
            RETURN c.id AS id,
                   c.level AS level,
                   coalesce(c.report_text, c.summary) AS summary,
                   coalesce(c.report_text, '') AS report_text,
                   coalesce(c.report_json, '') AS report_json,
                   c.rating AS rating,
                   score
            ORDER BY score DESC
            """
            params = {
                "index_name": "community-report-embeddings",
                "top_k": top_k,
                "query_vector": query_vector,
                "graph_name": graph_name,
                "level": level,
            }
        else:
            query = """
            CALL db.index.vector.queryNodes(
                $index_name, $top_k, $query_vector
            ) YIELD node AS c, score
            WHERE c.graph_name = $graph_name
            RETURN c.id AS id,
                   c.level AS level,
                   coalesce(c.report_text, c.summary) AS summary,
                   coalesce(c.report_text, '') AS report_text,
                   coalesce(c.report_json, '') AS report_json,
                   c.rating AS rating,
                   score
            ORDER BY score DESC
            """
            params = {
                "index_name": "community-report-embeddings",
                "top_k": top_k,
                "query_vector": query_vector,
                "graph_name": graph_name,
            }
        return self.execute_query(query, params)
