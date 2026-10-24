"""Cypher graph writer shared by all Cypher-compatible backends.

Maps the neutral GraphDocument into the graph using Cypher MERGE queries. The
Cypher is identical across Neo4j and Memgraph, so a single writer serves both;
backend dialect differences live in the graph store, not here.

Preserves the graph shape expected by retrieval, embedding, and community
detection:

- (:Document)
- (:Chunk)
- (:Chunk)-[:PART_OF]->(:Document)
- (:Chunk)-[:FROM_CHUNK]->(:__Entity__)
- (:__Entity__:DomainLabel)
- (:__Entity__)-[:DOMAIN_RELATIONSHIP]->(:__Entity__)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.cypher import escape_cypher_identifier

# Properties the writer's own Cypher sets. A free-form metadata/property map
# that carries one would overwrite SDK-owned graph state after MERGE.
_SDK_OWNED_KEYS = frozenset(
    {
        "id",
        "graph_name",
        "created",
        "updated",
        "embedding",
        "description_summary_status",
        "description_input_fingerprint",
        "description_summary_updated",
        "description_summary_error",
        "description_summary_fallback",
    }
)
_DOCUMENT_METADATA_RESERVED = _SDK_OWNED_KEYS | {"text_hash"}
_CHUNK_METADATA_RESERVED = _SDK_OWNED_KEYS | {"text", "index"}
_ENTITY_PROPERTIES_RESERVED = _SDK_OWNED_KEYS | {
    "type",
    "descriptions",
    "observation_count",
}
_RELATIONSHIP_PROPERTIES_RESERVED = _SDK_OWNED_KEYS | {"descriptions"}


class CypherGraphWriter:
    """Write GraphDocument records to a Cypher graph store."""

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def write_graph_document(self, graph_document: GraphDocument) -> dict[str, int]:
        self._reject_reserved_keys(graph_document)
        self._write_documents([graph_document.document])
        self._write_chunks(graph_document.chunks)
        self._write_entities(graph_document.entities)
        self._write_evidence_links(graph_document.evidence_links)
        self._write_relationships(graph_document.relationships)
        self._write_claims(graph_document.claims)

        return self._stats_for(graph_document)

    def _reject_reserved_keys(self, graph_document: GraphDocument) -> None:
        """Fail before any query when a free-form map would overwrite SDK fields."""
        sources = (
            ("DocumentRecord.metadata", _DOCUMENT_METADATA_RESERVED,
             [graph_document.document.metadata]),
            ("ChunkRecord.metadata", _CHUNK_METADATA_RESERVED,
             [chunk.metadata for chunk in graph_document.chunks]),
            ("EntityRecord.properties", _ENTITY_PROPERTIES_RESERVED,
             [entity.properties for entity in graph_document.entities]),
            ("RelationshipRecord.properties", _RELATIONSHIP_PROPERTIES_RESERVED,
             [rel.properties for rel in graph_document.relationships]),
        )
        for record_map, reserved, maps in sources:
            offending = sorted({key for m in maps for key in m if key in reserved})
            if offending:
                raise ValueError(
                    f"{record_map} contains reserved persistence field(s) "
                    f"{', '.join(offending)}; these graph properties are owned by "
                    "the SDK. Supply document identity via metadata['document_id'], "
                    "and rename application identifiers to a nonreserved name such "
                    "as 'external_id' or 'employee_id', then resubmit."
                )

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

    def _stats_for(self, graph_document: GraphDocument) -> dict[str, int]:
        return {
            "documents": 1,
            "chunks": len(graph_document.chunks),
            "entities": len(graph_document.entities),
            "relationships": len(graph_document.relationships),
            "evidence_links": len(graph_document.evidence_links),
            "claims": len(graph_document.claims),
        }

    def _document_rows(self, documents: list) -> list[dict[str, Any]]:
        return [
            {
                "id": doc.id,
                "text_hash": doc.text_hash,
                "graph_name": doc.graph_name,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

    def _chunk_rows(self, chunks: list) -> list[dict[str, Any]]:
        return [
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

    def _entity_rows(self, entities: list) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entity in entities:
            properties = dict(entity.properties)
            description = properties.pop("description", "") or ""
            rows.append(
                {
                    "id": entity.id,
                    "type": entity.type,
                    "graph_name": entity.graph_name,
                    "canonical_key": entity.canonical_key,
                    "human_readable_id": entity.human_readable_id,
                    "description": description,
                    "properties": properties,
                }
            )
        return rows

    def _evidence_link_rows(self, links: list) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": link.chunk_id,
                "entity_id": link.entity_id,
                "graph_name": link.graph_name,
            }
            for link in links
        ]

    def _relationship_rows(self, relationships: list) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rel in relationships:
            properties = dict(rel.properties)
            source_chunk_ids = properties.pop("source_chunk_ids", []) or []
            properties.pop("observation_count", None)
            properties.pop("weight", None)
            if rel.strength is not None:
                properties.pop("strength", None)
            observation_count = max(rel.observation_count, len(set(source_chunk_ids)), 1)
            rows.append(
                {
                    "id": rel.id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "graph_name": rel.graph_name,
                    "source_chunk_ids": list(dict.fromkeys(source_chunk_ids)),
                    "observation_count": observation_count,
                    "weight": float(observation_count),
                    "strength": rel.strength,
                    "properties": properties,
                }
            )
        return rows

    def _claim_rows(self, claims: list) -> list[dict[str, Any]]:
        return [
            {
                "id": claim.id,
                "entity_id": claim.entity_id,
                "chunk_id": claim.source.chunk_id,
                "claim_type": claim.claim_type,
                "description": claim.description,
                "status": claim.status,
                "start_date": claim.start_date,
                "end_date": claim.end_date,
                "object_entity_id": claim.object_entity_id,
                "source_text": claim.source_text,
                "text_unit_id": claim.text_unit_id,
                "graph_name": claim.graph_name,
            }
            for claim in claims
        ]

    def _group_by_type(self, records: list) -> dict[str, list]:
        grouped: dict[str, list] = defaultdict(list)
        for record in records:
            grouped[record.type].append(record)
        return dict(grouped)
