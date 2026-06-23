"""Memgraph graph writer implementation.

Maps the neutral GraphDocument into Memgraph using Cypher MERGE queries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.memgraph.cypher import escape_cypher_identifier


class MemgraphGraphWriter:
    """Write GraphDocument records to Memgraph.

    Preserves the graph shape expected by retrieval, embedding, and community detection:

    - (:Document)
    - (:Chunk)
    - (:Chunk)-[:PART_OF]->(:Document)
    - (:Chunk)-[:FROM_CHUNK]->(:__Entity__)
    - (:__Entity__:DomainLabel)
    - (:__Entity__)-[:DOMAIN_RELATIONSHIP]->(:__Entity__)
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        self._write_documents([graph_document.document])
        self._write_chunks(graph_document.chunks)
        self._write_entities(graph_document.entities)
        self._write_evidence_links(graph_document.evidence_links)
        self._write_relationships(graph_document.relationships)
        self._write_claims(graph_document.claims)

        return {
            "documents": 1,
            "chunks": len(graph_document.chunks),
            "entities": len(graph_document.entities),
            "relationships": len(graph_document.relationships),
            "evidence_links": len(graph_document.evidence_links),
            "claims": len(graph_document.claims),
        }

    def _write_documents(self, documents: list) -> None:
        if not documents:
            return

        rows = [
            {
                "id": doc.id,
                "text_hash": doc.text_hash,
                "graph_name": doc.graph_name,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        self.graph_store.execute_query(
            """
            UNWIND $documents AS row
            MERGE (d:Document {id: row.id})
            SET d += row.metadata,
                d.text_hash = row.text_hash,
                d.graph_name = row.graph_name,
                d.updated = timestamp(),
                d.created = coalesce(d.created, timestamp())
            """,
            {"documents": rows},
        )

    def _write_chunks(self, chunks: list) -> None:
        if not chunks:
            return

        rows = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "index": chunk.index,
                "graph_name": chunk.graph_name,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        self.graph_store.execute_query(
            """
            UNWIND $chunks AS row
            MATCH (d:Document {id: row.document_id})
            MERGE (c:Chunk {id: row.id})
            SET c.text = row.text,
                c.index = row.index,
                c.graph_name = row.graph_name,
                c += row.metadata,
                c.updated = timestamp(),
                c.created = coalesce(c.created, timestamp())
            MERGE (c)-[:PART_OF]->(d)
            """,
            {"chunks": rows},
        )

    def _write_entities(self, entities: list) -> None:
        if not entities:
            return

        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)

        for entity_type, group in by_type.items():
            label = escape_cypher_identifier(entity_type)
            rows = [
                {
                    "id": entity.id,
                    "type": entity.type,
                    "graph_name": entity.graph_name,
                    "properties": entity.properties,
                }
                for entity in group
            ]

            self.graph_store.execute_query(
                f"""
                UNWIND $entities AS row
                MERGE (e:__Entity__:{label} {{id: row.id}})
                SET e += row.properties,
                    e.type = row.type,
                    e.graph_name = row.graph_name,
                    e.name = coalesce(row.properties.name, row.id),
                    e.description = coalesce(row.properties.description, ''),
                    e.updated = timestamp(),
                    e.created = coalesce(e.created, timestamp())
                """,
                {"entities": rows},
            )

    def _write_evidence_links(self, links: list) -> None:
        if not links:
            return

        rows = [
            {
                "chunk_id": link.chunk_id,
                "entity_id": link.entity_id,
                "graph_name": link.graph_name,
            }
            for link in links
        ]

        self.graph_store.execute_query(
            """
            UNWIND $links AS row
            MATCH (c:Chunk {id: row.chunk_id})
            MATCH (e:__Entity__ {id: row.entity_id})
            MERGE (c)-[r:FROM_CHUNK]->(e)
            SET r.graph_name = row.graph_name
            """,
            {"links": rows},
        )

    def _write_relationships(self, relationships: list) -> None:
        if not relationships:
            return

        by_type = defaultdict(list)
        for rel in relationships:
            by_type[rel.type].append(rel)

        for rel_type, group in by_type.items():
            rel_label = escape_cypher_identifier(rel_type)
            rows = [
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "graph_name": rel.graph_name,
                    "properties": rel.properties,
                }
                for rel in group
            ]

            self.graph_store.execute_query(
                f"""
                UNWIND $relationships AS row
                MATCH (source:__Entity__ {{id: row.source_id}})
                MATCH (target:__Entity__ {{id: row.target_id}})
                MERGE (source)-[r:{rel_label}]->(target)
                SET r += row.properties,
                    r.graph_name = row.graph_name,
                    r.updated = timestamp(),
                    r.created = coalesce(r.created, timestamp())
                """,
                {"relationships": rows},
            )

    def _write_claims(self, claims: list) -> None:
        """Write Claim nodes with SUBJECT_OF and SOURCED_FROM edges."""
        if not claims:
            return

        rows = [
            {
                "id": claim.id,
                "entity_id": claim.entity_id,
                "chunk_id": claim.source.chunk_id,
                "claim_type": claim.claim_type,
                "description": claim.description,
                "status": claim.status,
                "graph_name": claim.graph_name,
            }
            for claim in claims
        ]

        self.graph_store.execute_query(
            """
            UNWIND $claims AS row
            MERGE (c:Claim {id: row.id})
            SET c.claim_type = row.claim_type,
                c.description = row.description,
                c.status = row.status,
                c.graph_name = row.graph_name,
                c.updated = timestamp(),
                c.created = coalesce(c.created, timestamp())
            WITH c, row
            MATCH (e:__Entity__ {id: row.entity_id})
            MERGE (c)-[:SUBJECT_OF]->(e)
            WITH c, row
            MATCH (ch:Chunk {id: row.chunk_id})
            MERGE (c)-[:SOURCED_FROM]->(ch)
            """,
            {"claims": rows},
        )
