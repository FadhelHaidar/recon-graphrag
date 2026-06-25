"""Memgraph graph writer implementation.

Maps the neutral GraphDocument into Memgraph using Cypher MERGE queries.
"""

from __future__ import annotations

from recon_graphrag.graphdb.memgraph.cypher import escape_cypher_identifier
from recon_graphrag.pipelines.writer_base import BaseGraphWriter


class MemgraphGraphWriter(BaseGraphWriter):
    """Write GraphDocument records to Memgraph.

    Preserves the graph shape expected by retrieval, embedding, and community detection:

    - (:Document)
    - (:Chunk)
    - (:Chunk)-[:PART_OF]->(:Document)
    - (:Chunk)-[:FROM_CHUNK]->(:__Entity__)
    - (:__Entity__:DomainLabel)
    - (:__Entity__)-[:DOMAIN_RELATIONSHIP]->(:__Entity__)
    """

    def _write_documents(self, documents: list) -> None:
        if not documents:
            return

        self.graph_store.execute_query(
            """
            UNWIND $documents AS row
            MERGE (d:Document {id: row.id, graph_name: row.graph_name})
            SET d += row.metadata,
                d.text_hash = row.text_hash,
                d.updated = timestamp(),
                d.created = coalesce(d.created, timestamp())
            """,
            {"documents": self._document_rows(documents)},
        )

    def _write_chunks(self, chunks: list) -> None:
        if not chunks:
            return

        self.graph_store.execute_query(
            """
            UNWIND $chunks AS row
            MATCH (d:Document {id: row.document_id, graph_name: row.graph_name})
            MERGE (c:Chunk {id: row.id, graph_name: row.graph_name})
            SET c.text = row.text,
                c.index = row.index,
                c += row.metadata,
                c.updated = timestamp(),
                c.created = coalesce(c.created, timestamp())
            MERGE (c)-[:PART_OF]->(d)
            """,
            {"chunks": self._chunk_rows(chunks)},
        )

    def _write_entities(self, entities: list) -> None:
        if not entities:
            return

        for entity_type, group in self._group_by_type(entities).items():
            label = escape_cypher_identifier(entity_type)
            self.graph_store.execute_query(
                f"""
                UNWIND $entities AS row
                MERGE (e:__Entity__:{label} {{id: row.id, graph_name: row.graph_name}})
                SET e += row.properties,
                    e.type = row.type,
                    e.canonical_key = coalesce(row.canonical_key, row.properties.canonical_key),
                    e.human_readable_id = coalesce(row.human_readable_id, row.properties.human_readable_id),
                    e.name = coalesce(row.properties.name, row.properties.title, row.human_readable_id, row.id),
                    e.title = coalesce(row.properties.title, row.properties.name, row.human_readable_id, row.id),
                    e.description = coalesce(row.properties.description, ''),
                    e.updated = timestamp(),
                    e.created = coalesce(e.created, timestamp())
                """,
                {"entities": self._entity_rows(group)},
            )

    def _write_evidence_links(self, links: list) -> None:
        if not links:
            return

        self.graph_store.execute_query(
            """
            UNWIND $links AS row
            MATCH (c:Chunk {id: row.chunk_id, graph_name: row.graph_name})
            MATCH (e:__Entity__ {id: row.entity_id, graph_name: row.graph_name})
            MERGE (c)-[r:FROM_CHUNK]->(e)
            SET r.graph_name = row.graph_name
            """,
            {"links": self._evidence_link_rows(links)},
        )

    def _write_relationships(self, relationships: list) -> None:
        if not relationships:
            return

        for rel_type, group in self._group_by_type(relationships).items():
            rel_label = escape_cypher_identifier(rel_type)
            self.graph_store.execute_query(
                f"""
                UNWIND $relationships AS row
                MATCH (source:__Entity__ {{id: row.source_id, graph_name: row.graph_name}})
                MATCH (target:__Entity__ {{id: row.target_id, graph_name: row.graph_name}})
                MERGE (source)-[r:{rel_label}]->(target)
                SET r.id = row.id,
                    r += row.properties,
                    r.graph_name = row.graph_name,
                    r.updated = timestamp(),
                    r.created = coalesce(r.created, timestamp())
                """,
                {"relationships": self._relationship_rows(group)},
            )

    def _write_claims(self, claims: list) -> None:
        """Write Claim nodes with SUBJECT_OF and SOURCED_FROM edges."""
        if not claims:
            return

        self.graph_store.execute_query(
            """
            UNWIND $claims AS row
            MERGE (c:Claim {id: row.id, graph_name: row.graph_name})
            SET c.claim_type = row.claim_type,
                c.description = row.description,
                c.status = row.status,
                c.updated = timestamp(),
                c.created = coalesce(c.created, timestamp())
            WITH c, row
            MATCH (e:__Entity__ {id: row.entity_id, graph_name: row.graph_name})
            MERGE (c)-[:SUBJECT_OF]->(e)
            WITH c, row
            MATCH (ch:Chunk {id: row.chunk_id, graph_name: row.graph_name})
            MERGE (c)-[:SOURCED_FROM]->(ch)
            """,
            {"claims": self._claim_rows(claims)},
        )
