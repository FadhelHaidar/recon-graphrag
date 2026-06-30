"""Neo4j graph writer implementation.

Maps neutral GraphDocument into Neo4j using Cypher MERGE queries.
"""

from __future__ import annotations

from recon_graphrag.graphdb.neo4j.cypher import escape_cypher_identifier
from recon_graphrag.pipelines.writer_base import BaseGraphWriter


class Neo4jGraphWriter(BaseGraphWriter):
    """Write GraphDocument records to Neo4j.

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
                WITH e, row,
                     e.description_summary_status AS existing_summary_status,
                     CASE
                       WHEN e.descriptions IS NOT NULL THEN e.descriptions
                       WHEN e.description IS NULL OR e.description = '' THEN []
                       ELSE [e.description]
                     END AS existing_descriptions
                SET e += row.properties,
                    e.type = row.type,
                    e.canonical_key = coalesce(row.canonical_key, row.properties.canonical_key),
                    e.human_readable_id = coalesce(row.human_readable_id, row.properties.human_readable_id),
                    e.name = coalesce(row.properties.name, row.properties.title, row.human_readable_id, row.id),
                    e.title = coalesce(row.properties.title, row.properties.name, row.human_readable_id, row.id),
                    e.updated = timestamp(),
                    e.created = coalesce(e.created, timestamp())
                WITH e, row, existing_summary_status, existing_descriptions,
                     CASE
                       WHEN row.description = '' THEN existing_descriptions
                       ELSE reduce(acc = [], item IN existing_descriptions + [row.description] |
                         CASE WHEN item IN acc THEN acc ELSE acc + [item] END)
                     END AS merged_descriptions
                SET e.descriptions = merged_descriptions,
                    e.observation_count = size(merged_descriptions),
                    e.description_summary_status = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE existing_summary_status
                    END,
                    e.description_input_fingerprint = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE e.description_input_fingerprint
                    END,
                    e.description_summary_error = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE e.description_summary_error
                    END,
                    e.description = CASE
                        WHEN existing_summary_status IS NULL THEN
                            reduce(text = '', item IN merged_descriptions |
                                text + CASE WHEN text = '' THEN '' ELSE '\n' END + item)
                        ELSE e.description
                    END
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
                WITH r, row, coalesce(r.source_chunk_ids, []) AS existing_chunk_ids,
                     r.description_summary_status AS existing_summary_status,
                     coalesce(r.descriptions,
                       CASE WHEN r.description IS NULL OR r.description = '' THEN []
                            ELSE [r.description] END) AS existing_descriptions
                SET r.id = row.id,
                    r += row.properties,
                    r.graph_name = row.graph_name,
                    r.updated = timestamp(),
                    r.created = coalesce(r.created, timestamp())
                WITH r, row, existing_descriptions, existing_summary_status,
                     reduce(acc = [], item IN existing_chunk_ids + row.source_chunk_ids |
                       CASE WHEN item IN acc THEN acc ELSE acc + [item] END) AS merged_chunk_ids
                WITH r, row, merged_chunk_ids, existing_descriptions, existing_summary_status,
                     reduce(acc = [], item IN existing_descriptions +
                       CASE WHEN row.properties.description IS NULL OR
                                      row.properties.description = '' THEN []
                            ELSE [row.properties.description] END |
                       CASE WHEN item IN acc THEN acc ELSE acc + [item] END
                     ) AS merged_descriptions
                SET r.source_chunk_ids = merged_chunk_ids,
                    r.observation_count = size(merged_chunk_ids),
                    r.weight = toFloat(size(merged_chunk_ids)),
                    r.descriptions = merged_descriptions,
                    r.description_summary_status = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE existing_summary_status
                    END,
                    r.description_input_fingerprint = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE r.description_input_fingerprint
                    END,
                    r.description_summary_error = CASE
                        WHEN existing_summary_status IS NOT NULL
                         AND merged_descriptions <> existing_descriptions THEN NULL
                        ELSE r.description_summary_error
                    END,
                    r.description = CASE
                        WHEN existing_summary_status IS NULL THEN
                            reduce(text = '', item IN merged_descriptions |
                                text + CASE WHEN text = '' THEN '' ELSE '\n' END + item)
                        ELSE r.description
                    END,
                    r.strength = CASE
                        WHEN row.strength IS NULL THEN r.strength
                        WHEN r.strength IS NULL OR row.strength > r.strength THEN row.strength
                        ELSE r.strength
                    END
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
                c.start_date = row.start_date,
                c.end_date = row.end_date,
                c.object_entity_id = row.object_entity_id,
                c.source_text = row.source_text,
                c.text_unit_id = row.text_unit_id,
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
