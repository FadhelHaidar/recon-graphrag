"""Cypher graph writer shared by all Cypher-compatible backends.

Maps the neutral GraphDocument into the graph using Cypher MERGE queries. The
Cypher is identical across Neo4j and Memgraph, so a single writer serves both;
backend dialect differences live in the graph store, not here.

Writing a document replaces that document's previous contribution in its
``graph_name``: new records are written first, then evidence, claims, chunks,
relationships, entities, and description text that only the old version
supplied are removed. Contributions from other documents and graph names are
left untouched. Relationship chunk provenance, and description text per
document, are recomputed from surviving contributions rather than accumulated.

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
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from recon_graphrag.entity_text import entity_embedding_fingerprint
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb._provenance import (
    build_uid as _build_uid,
    contribution_entries,
    encode_contributions,
    raw_descriptions,
)
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
        "description_sources",
        "strength_sources",
        "embedding_input_fingerprint",
        "embedding_content_fingerprint",
        "embedding_model_fingerprint",
        "uid",
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


@dataclass
class _DocumentState:
    """What the graph already holds for one document, read before writing."""

    chunk_ids: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    entities: dict[str, dict] = field(default_factory=dict)


def _description_entries(value: Any) -> list[list[str]]:
    """Decode property-compatible description provenance, rejecting invalid text."""
    entries = contribution_entries(value)
    if any(not isinstance(text, str) or not text.strip() for _, text in entries):
        raise ValueError("Invalid description provenance; re-ingest the affected documents.")
    return entries


def _merge_contributions(
    entries: list[list[str]],
    document_id: str,
    new_descriptions: list[str],
) -> tuple[list[list[str]], list[str]]:
    """Replace one document's description contribution and rebuild the union."""
    merged = [entry for entry in entries if entry[0] != document_id]
    merged.extend(
        [document_id, text] for text in new_descriptions if str(text).strip()
    )
    descriptions: list[str] = []
    for _, text in merged:
        if text not in descriptions:
            descriptions.append(text)
    return merged, descriptions


def _surviving_chunk_ids(
    existing: list[str], owned_chunk_ids: list[str], new_chunk_ids: list[str]
) -> list[str]:
    """Drop this document's previous chunk ids, then add its current ones."""
    owned = set(owned_chunk_ids)
    surviving = [chunk_id for chunk_id in existing if chunk_id not in owned]
    surviving.extend(new_chunk_ids)
    return list(dict.fromkeys(surviving))


def _joined(descriptions: list[str]) -> str:
    return "\n".join(descriptions)


class CypherGraphWriter:
    """Write GraphDocument records to a Cypher graph store."""

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def write_graph_document(self, graph_document: GraphDocument) -> dict[str, int]:
        transaction = getattr(self.graph_store, "_transaction", None)
        if transaction is None:
            return self._write_document(graph_document)
        # One transaction per document, so a mid-write failure leaves nothing
        # behind. A store without the hook keeps the auto-commit path.
        with transaction() as executor:
            scoped = copy(self)
            scoped.graph_store = executor
            return scoped._write_document(graph_document)

    def _write_document(self, graph_document: GraphDocument) -> dict[str, int]:
        self._reject_reserved_keys(graph_document)
        state = self._read_document_state(graph_document)
        entity_rows = self._entity_rows(graph_document, state)
        relationship_rows = self._relationship_rows(graph_document, state)

        rebuild_relationships = self._rebuild_relationship_rows(
            graph_document, state, relationship_rows
        )
        # Validate removed contributions too, before the first mutation.
        self._rebuild_entity_rows(graph_document, state, entity_rows)
        self._validate_references(graph_document, state)
        documents = self._write_documents([graph_document.document])
        chunks = self._write_chunks(graph_document.chunks)
        entities = self._write_entities(graph_document.entities, entity_rows)
        evidence_links = self._write_evidence_links(graph_document.evidence_links)
        relationships = self._write_relationships(
            graph_document, relationship_rows, rebuild_relationships
        )
        claims = self._write_claims(graph_document.claims)
        self._reconcile(
            graph_document,
            state,
            entity_rows,
            relationship_rows,
            rebuild_relationships,
        )

        # A receipt of what the database confirmed, not of what was submitted.
        return {
            "documents": documents,
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "evidence_links": evidence_links,
            "claims": claims,
        }

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

    def _validate_references(
        self, graph_document: GraphDocument, state: _DocumentState
    ) -> None:
        """Fail before the first write when a reference will not resolve.

        The write queries MATCH their endpoints, so a record naming a missing
        entity or chunk is dropped without an error. Endpoints owned by other
        documents resolve against the graph.
        """
        chunk_ids = {chunk.id for chunk in graph_document.chunks}
        chunk_ids.update(state.chunk_ids)
        entity_ids = {entity.id for entity in graph_document.entities}
        entity_ids.update(state.entities)

        chunk_refs = [link.chunk_id for link in graph_document.evidence_links]
        chunk_refs += [claim.source.chunk_id for claim in graph_document.claims]
        entity_refs = [link.entity_id for link in graph_document.evidence_links]
        entity_refs += [claim.entity_id for claim in graph_document.claims]
        for relationship in graph_document.relationships:
            entity_refs += [relationship.source_id, relationship.target_id]

        missing_chunks = {
            ref for ref in chunk_refs if ref and ref not in chunk_ids
        }
        missing_entities = {
            ref for ref in entity_refs if ref and ref not in entity_ids
        }
        if missing_entities:
            missing_entities -= self._existing_entity_ids(
                graph_document.document.graph_name, sorted(missing_entities)
            )
        if not missing_chunks and not missing_entities:
            return

        missing = []
        if missing_chunks:
            missing.append(f"chunk(s) {', '.join(sorted(missing_chunks))}")
        if missing_entities:
            missing.append(f"entity(ies) {', '.join(sorted(missing_entities))}")
        raise ValueError(
            f"GraphDocument {graph_document.document.id!r} references missing "
            f"{' and '.join(missing)}; relationships, evidence links, and claims "
            "are only written when their endpoints exist. Include the referenced "
            "records in the document, or write them first."
        )

    def _existing_entity_ids(
        self, graph_name: str, entity_ids: list[str]
    ) -> set[str]:
        return {
            row["id"]
            for row in self.graph_store.execute_query(
                """
                UNWIND $entity_ids AS id
                MATCH (e:__Entity__ {id: id, graph_name: $graph_name})
                RETURN e.id AS id
                """,
                {"graph_name": graph_name, "entity_ids": entity_ids},
            )
        }

    # ------------------------------------------------------------------
    # Existing contribution state
    # ------------------------------------------------------------------
    def _read_document_state(self, graph_document: GraphDocument) -> _DocumentState:
        """Read what this document previously contributed, before any write."""
        document = graph_document.document
        params = {"graph_name": document.graph_name, "document_id": document.id}

        state = _DocumentState(
            chunk_ids=[
                row["id"]
                for row in self.graph_store.execute_query(
                    """
                    MATCH (d:Document {id: $document_id, graph_name: $graph_name})
                    MATCH (c:Chunk {graph_name: $graph_name})-[:PART_OF]->(d)
                    RETURN c.id AS id
                    """,
                    params,
                )
            ],
            evidence=list(
                self.graph_store.execute_query(
                    """
                    MATCH (d:Document {id: $document_id, graph_name: $graph_name})
                    MATCH (c:Chunk {graph_name: $graph_name})-[:PART_OF]->(d)
                    MATCH (c)-[:FROM_CHUNK]->(e:__Entity__)
                    RETURN c.id AS chunk_id, e.id AS entity_id
                    """,
                    params,
                )
            ),
            claims=list(
                self.graph_store.execute_query(
                    """
                    MATCH (d:Document {id: $document_id, graph_name: $graph_name})
                    MATCH (c:Chunk {graph_name: $graph_name})-[:PART_OF]->(d)
                    MATCH (cl:Claim {graph_name: $graph_name})-[:SOURCED_FROM]->(c)
                    OPTIONAL MATCH (cl)-[:SUBJECT_OF]->(e:__Entity__)
                    RETURN cl.id AS id, c.id AS chunk_id, e.id AS entity_id
                    """,
                    params,
                )
            ),
        )

        state.relationships = list(
            self.graph_store.execute_query(
                """
                MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
                WHERE r.graph_name = $graph_name
                  AND (r.id IN $rel_ids OR any(chunk IN coalesce(r.source_chunk_ids, [])
                                               WHERE chunk IN $chunk_ids))
                RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                       type(r) AS type,
                       coalesce(r.source_chunk_ids, []) AS source_chunk_ids,
                       r.descriptions AS descriptions,
                       r.description_sources AS description_sources,
                       r.description AS description,
                       r.description_summary_status AS description_summary_status,
                       r.strength AS strength,
                       r.strength_sources AS strength_sources,
                       r.properties AS properties
                """,
                {
                    **params,
                    "rel_ids": [rel.id for rel in graph_document.relationships],
                    "chunk_ids": state.chunk_ids,
                },
            )
        )

        entity_ids = list(
            dict.fromkeys(
                [entity.id for entity in graph_document.entities]
                + [row.get("entity_id") for row in state.evidence if row.get("entity_id")]
                + [row.get("entity_id") for row in state.claims if row.get("entity_id")]
                + [row.get("source_id") for row in state.relationships]
                + [row.get("target_id") for row in state.relationships]
            )
        )
        if entity_ids:
            state.entities = {
                row["id"]: row
                for row in self.graph_store.execute_query(
                    """
                    UNWIND $entity_ids AS id
                    MATCH (e:__Entity__ {id: id, graph_name: $graph_name})
                    RETURN e.id AS id,
                           e.descriptions AS descriptions,
                           e.description_sources AS description_sources,
                           e.description AS description,
                           e.description_summary_status AS description_summary_status,
                           e.name AS name,
                           labels(e) AS labels,
                           e.embedding_input_fingerprint AS embedding_input_fingerprint
                    """,
                    {"graph_name": document.graph_name, "entity_ids": entity_ids},
                )
            }
        return state

    # ------------------------------------------------------------------
    # Row construction (replacement-aware, pure)
    # ------------------------------------------------------------------
    def _entity_rows(
        self, graph_document: GraphDocument, state: _DocumentState
    ) -> list[dict[str, Any]]:
        document_id = graph_document.document.id
        rows: list[dict[str, Any]] = []
        for entity in graph_document.entities:
            properties = dict(entity.properties)
            description = str(properties.pop("description", "") or "")
            stored = state.entities.get(entity.id) or {}
            stored_descriptions = raw_descriptions(stored)
            entries = _accessible_entries(
                stored.get("description_sources"), stored_descriptions,
                [description], document_id, entity.id,
            )
            merged_entries, descriptions = _merge_contributions(
                entries, document_id, [description]
            )
            changed = descriptions != stored_descriptions
            keep_description = (
                stored.get("description_summary_status") is not None and not changed
            )
            description_text = (
                None if keep_description else _joined(descriptions)
            )
            name = (
                properties.get("name")
                or properties.get("title")
                or entity.human_readable_id
                or entity.id
            )
            final_description = (
                stored.get("description") or ""
                if description_text is None
                else description_text
            )
            rows.append(
                {
                    "id": entity.id,
                    "type": entity.type,
                    "graph_name": entity.graph_name,
                    "uid": _build_uid(entity.graph_name, entity.id),
                    "canonical_key": entity.canonical_key,
                    "human_readable_id": entity.human_readable_id,
                    "description_sources": encode_contributions(merged_entries),
                    "descriptions": descriptions,
                    "description_text": description_text,
                    "descriptions_changed": changed,
                    "embedding_input_fingerprint": entity_embedding_fingerprint(
                        labels=[entity.type, "__Entity__"],
                        name=name,
                        description=final_description,
                    ),
                    "properties": properties,
                }
            )
        return rows

    def _relationship_rows(
        self, graph_document: GraphDocument, state: _DocumentState
    ) -> list[dict[str, Any]]:
        document_id = graph_document.document.id
        stored_by_key = {
            _relationship_key(row): row for row in state.relationships
        }
        rows: list[dict[str, Any]] = []
        for rel in graph_document.relationships:
            properties = dict(rel.properties)
            new_chunk_ids = list(dict.fromkeys(properties.pop("source_chunk_ids", []) or []))
            properties.pop("observation_count", None)
            properties.pop("weight", None)
            strength = rel.strength if rel.strength is not None else properties.get("strength")
            properties.pop("strength", None)
            description = str(properties.get("description") or "")
            stored = stored_by_key.get(_relationship_key(rel)) or {}
            stored_descriptions = raw_descriptions(stored)
            entries = _accessible_entries(
                stored.get("description_sources"), stored_descriptions,
                [description], document_id, rel.id,
            )
            merged_entries, descriptions = _merge_contributions(
                entries, document_id, [description]
            )
            chunk_ids = _surviving_chunk_ids(
                [str(chunk) for chunk in (stored.get("source_chunk_ids") or [])],
                state.chunk_ids,
                new_chunk_ids,
            )
            strength_entries, strength_value = _replace_strength(stored, document_id, strength)
            observation_count = max(len(chunk_ids), rel.observation_count, 1)
            changed = descriptions != stored_descriptions
            rows.append(
                {
                    "id": rel.id,
                    "type": rel.type,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "graph_name": rel.graph_name,
                    "source_chunk_ids": chunk_ids,
                    "observation_count": observation_count,
                    "weight": float(observation_count),
                    "strength": strength_value,
                    "strength_sources": encode_contributions(strength_entries),
                    "description_sources": encode_contributions(merged_entries),
                    "descriptions": descriptions,
                    "description_text": (
                        stored.get("description")
                        if stored.get("description_summary_status") is not None
                        and not changed
                        else _joined(descriptions)
                    ),
                    "keep_description": (
                        stored.get("description_summary_status") is not None
                        and not changed
                    ),
                    "descriptions_changed": changed,
                    "properties": properties,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _write_documents(self, documents: list) -> int:
        if not documents:
            return 0

        rows = self._document_rows(documents)
        result = self.graph_store.execute_query(
            """
            UNWIND $documents AS row
            MERGE (d:Document {id: row.id, graph_name: row.graph_name})
            SET d += row.metadata,
                d.text_hash = row.text_hash,
                d.uid = row.uid,
                d.updated = timestamp(),
                d.created = coalesce(d.created, timestamp())
            RETURN count(*) AS written
            """,
            {"documents": rows},
        )
        return _confirmed(result, len(rows))

    def _write_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0

        rows = self._chunk_rows(chunks)
        result = self.graph_store.execute_query(
            """
            UNWIND $chunks AS row
            MATCH (d:Document {id: row.document_id, graph_name: row.graph_name})
            MERGE (c:Chunk {id: row.id, graph_name: row.graph_name})
            SET c.text = row.text,
                c.index = row.index,
                c.uid = row.uid,
                c += row.metadata,
                c.updated = timestamp(),
                c.created = coalesce(c.created, timestamp())
            MERGE (c)-[:PART_OF]->(d)
            RETURN count(*) AS written
            """,
            {"chunks": rows},
        )
        return _confirmed(result, len(rows))

    def _write_entities(
        self, entities: list, rows: list[dict[str, Any]]
    ) -> int:
        if not entities:
            return 0

        written = 0
        for entity_type, group in self._group_by_type(entities).items():
            label = escape_cypher_identifier(entity_type)
            group_ids = {entity.id for entity in group}
            group_rows = [row for row in rows if row["id"] in group_ids]
            result = self.graph_store.execute_query(
                f"""
                UNWIND $entities AS row
                MERGE (e:__Entity__:{label} {{id: row.id, graph_name: row.graph_name}})
                SET e.uid = row.uid,
                    e += row.properties,
                    e.type = row.type,
                    e.canonical_key = coalesce(row.canonical_key, row.properties.canonical_key),
                    e.human_readable_id = coalesce(row.human_readable_id, row.properties.human_readable_id),
                    e.name = coalesce(row.properties.name, row.properties.title, row.human_readable_id, row.id),
                    e.title = coalesce(row.properties.title, row.properties.name, row.human_readable_id, row.id),
                    e.updated = timestamp(),
                    e.created = coalesce(e.created, timestamp()),
                    e.description_sources = row.description_sources,
                    e.descriptions = row.descriptions,
                    e.observation_count = size(row.descriptions),
                    e.embedding_input_fingerprint = row.embedding_input_fingerprint,
                    e.description = CASE
                        WHEN row.description_text IS NULL THEN e.description
                        ELSE row.description_text
                    END,
                    e.description_summary_status = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_status
                    END,
                    e.description_summary_error = CASE WHEN row.descriptions_changed THEN NULL ELSE e.description_summary_error END,
                    e.description_summary_updated = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_updated
                    END,
                    e.description_summary_fallback = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_fallback
                    END,
                    e.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_input_fingerprint
                    END
                """,
                {"entities": group_rows},
            )
            written += _confirmed(result, len(group_rows))
        return written

    def _write_evidence_links(self, links: list) -> int:
        if not links:
            return 0

        rows = self._evidence_link_rows(links)
        result = self.graph_store.execute_query(
            """
            UNWIND $links AS row
            MATCH (c:Chunk {id: row.chunk_id, graph_name: row.graph_name})
            MATCH (e:__Entity__ {id: row.entity_id, graph_name: row.graph_name})
            MERGE (c)-[r:FROM_CHUNK]->(e)
            SET r.graph_name = row.graph_name
            RETURN count(*) AS written
            """,
            {"links": rows},
        )
        return _confirmed(result, len(rows))

    def _write_relationships(
        self,
        graph_document: GraphDocument,
        rows: list[dict[str, Any]],
        rebuild: list[dict[str, Any]],
    ) -> int:
        """Write this document's relationships, plus provenance-only rebuilds."""
        relationships = graph_document.relationships
        by_key = {_relationship_key(row): row for row in rows}
        written = 0
        for rel_type, group in self._group_by_type(relationships).items():
            rel_label = escape_cypher_identifier(rel_type)
            group_rows = [
                by_key[_relationship_key(rel)]
                for rel in group
                if _relationship_key(rel) in by_key
            ]
            result = self.graph_store.execute_query(
                f"""
                UNWIND $relationships AS row
                MATCH (source:__Entity__ {{id: row.source_id, graph_name: row.graph_name}})
                MATCH (target:__Entity__ {{id: row.target_id, graph_name: row.graph_name}})
                MERGE (source)-[r:{rel_label}]->(target)
                SET r.id = row.id,
                    r += row.properties,
                    r.graph_name = row.graph_name,
                    r.updated = timestamp(),
                    r.created = coalesce(r.created, timestamp()),
                    r.source_chunk_ids = row.source_chunk_ids,
                    r.observation_count = row.observation_count,
                    r.weight = row.weight,
                    r.strength = row.strength,
                    r.strength_sources = row.strength_sources,
                    r.description_sources = row.description_sources,
                    r.descriptions = row.descriptions,
                    r.description = CASE
                        WHEN row.description_text IS NULL THEN r.description
                        ELSE row.description_text
                    END,
                    r.description_summary_status = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_status
                    END,
                    r.description_summary_error = CASE WHEN row.descriptions_changed THEN NULL ELSE r.description_summary_error END,
                    r.description_summary_updated = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_updated
                    END,
                    r.description_summary_fallback = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_fallback
                    END,
                    r.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_input_fingerprint
                    END
                RETURN count(*) AS written
                """,
                {"relationships": group_rows},
            )
            written += _confirmed(result, len(group_rows))

        if rebuild:
            self.graph_store.execute_query(
                """
                UNWIND $relationships AS row
                MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
                WHERE r.graph_name = $graph_name AND r.id = row.id
                SET r.source_chunk_ids = row.source_chunk_ids,
                    r.observation_count = row.observation_count,
                    r.weight = row.weight,
                    r.strength = row.strength,
                    r.strength_sources = row.strength_sources,
                    r.description_sources = row.description_sources,
                    r.descriptions = row.descriptions,
                    r.description = row.description_text,
                    r.description_summary_status = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_status
                    END,
                    r.description_summary_error = CASE WHEN row.descriptions_changed THEN NULL ELSE r.description_summary_error END,
                    r.description_summary_updated = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_updated
                    END,
                    r.description_summary_fallback = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_fallback
                    END,
                    r.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_input_fingerprint
                    END
                """,
                {
                    "graph_name": graph_document.document.graph_name,
                    "relationships": rebuild,
                },
            )
        return written

    def _write_claims(self, claims: list) -> int:
        """Write Claim nodes with SUBJECT_OF and SOURCED_FROM edges."""
        if not claims:
            return 0

        rows = self._claim_rows(claims)
        result = self.graph_store.execute_query(
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
            RETURN count(*) AS written
            """,
            {"claims": rows},
        )
        # Rows that survived both endpoint MATCHes: fully linked claims.
        return _confirmed(result, len(rows))

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def _rebuild_relationship_rows(
        self,
        graph_document: GraphDocument,
        state: _DocumentState,
        retained_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop this document's chunk and description contributions from edges
        that its new version no longer contains."""
        document_id = graph_document.document.id
        retained_keys = {_relationship_key(row) for row in retained_rows}
        rows: list[dict[str, Any]] = []
        for stored in state.relationships:
            if _relationship_key(stored) in retained_keys:
                continue
            stored_descriptions = raw_descriptions(stored)
            entries, descriptions = _merge_contributions(
                _accessible_entries(stored.get("description_sources"), stored_descriptions,
                                    [], document_id, stored["id"]),
                document_id,
                [],
            )
            chunk_ids = _surviving_chunk_ids(
                [str(chunk) for chunk in (stored.get("source_chunk_ids") or [])],
                state.chunk_ids,
                [],
            )
            changed = descriptions != stored_descriptions
            strength_entries, strength_value = _replace_strength(stored, document_id, None)
            if (not changed and chunk_ids == (stored.get("source_chunk_ids") or [])
                    and encode_contributions(entries) == stored.get("description_sources")
                    and encode_contributions(strength_entries) == stored.get("strength_sources")):
                continue
            rows.append(
                {
                    "id": stored.get("id"),
                    "graph_name": graph_document.document.graph_name,
                    "source_chunk_ids": chunk_ids,
                    "observation_count": len(chunk_ids),
                    "weight": float(len(chunk_ids)),
                    "strength": strength_value,
                    "strength_sources": encode_contributions(strength_entries),
                    "description_sources": encode_contributions(entries),
                    "descriptions": descriptions,
                    "description_text": (
                        stored.get("description")
                        if stored.get("description_summary_status") is not None
                        and not changed
                        else _joined(descriptions)
                    ),
                    "descriptions_changed": changed,
                }
            )
        return rows

    def _rebuild_entity_rows(
        self,
        graph_document: GraphDocument,
        state: _DocumentState,
        retained_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop this document's description contribution from entities that its
        new version no longer mentions."""
        document_id = graph_document.document.id
        retained_ids = {row["id"] for row in retained_rows}
        rows: list[dict[str, Any]] = []
        for stored in state.entities.values():
            if stored["id"] in retained_ids:
                continue
            stored_descriptions = raw_descriptions(stored)
            entries, descriptions = _merge_contributions(
                _accessible_entries(stored.get("description_sources"), stored_descriptions,
                                    [], document_id, stored["id"]),
                document_id,
                [],
            )
            changed = descriptions != stored_descriptions
            if not changed and encode_contributions(entries) == stored.get("description_sources"):
                continue
            description_text = (
                stored.get("description")
                if not changed and stored.get("description_summary_status") is not None
                else _joined(descriptions)
            )
            rows.append(
                {
                    "id": stored["id"],
                    "description_sources": encode_contributions(entries),
                    "descriptions": descriptions,
                    "description_text": description_text,
                    "descriptions_changed": changed,
                    "embedding_input_fingerprint": entity_embedding_fingerprint(
                        labels=stored.get("labels"),
                        name=stored.get("name") or stored["id"],
                        description=description_text,
                    ),
                }
            )
        return rows

    def _reconcile(
        self,
        graph_document: GraphDocument,
        state: _DocumentState,
        entity_rows: list[dict[str, Any]],
        relationship_rows: list[dict[str, Any]],
        rebuild_relationships: list[dict[str, Any]],
    ) -> None:
        """Remove what only the document's previous version contributed."""
        graph_name = graph_document.document.graph_name
        keep_chunk_ids = [chunk.id for chunk in graph_document.chunks]
        keep_evidence = {
            (link.chunk_id, link.entity_id)
            for link in graph_document.evidence_links
        }
        keep_claim_ids = {claim.id for claim in graph_document.claims}
        removed_chunk_ids = [
            chunk_id for chunk_id in state.chunk_ids
            if chunk_id not in keep_chunk_ids
        ]

        stale_evidence = [
            row for row in state.evidence
            if (row["chunk_id"], row["entity_id"]) not in keep_evidence
        ]
        if stale_evidence:
            self.graph_store.execute_query(
                """
                UNWIND $links AS row
                MATCH (c:Chunk {id: row.chunk_id, graph_name: $graph_name})
                      -[r:FROM_CHUNK]->
                      (e:__Entity__ {id: row.entity_id, graph_name: $graph_name})
                DELETE r
                """,
                {"graph_name": graph_name, "links": stale_evidence},
            )

        stale_claims = [
            row["id"] for row in state.claims
            if row["id"] not in keep_claim_ids or row.get("chunk_id") in removed_chunk_ids
        ]
        if stale_claims:
            self.graph_store.execute_query(
                """
                UNWIND $claim_ids AS id
                MATCH (c:Claim {id: id, graph_name: $graph_name})
                DETACH DELETE c
                """,
                {"graph_name": graph_name, "claim_ids": sorted(set(stale_claims))},
            )

        if removed_chunk_ids:
            self.graph_store.execute_query(
                """
                UNWIND $chunk_ids AS id
                MATCH (c:Chunk {id: id, graph_name: $graph_name})
                DETACH DELETE c
                """,
                {"graph_name": graph_name, "chunk_ids": removed_chunk_ids},
            )

        empty_relationship_ids = [
            row["id"]
            for row in list(relationship_rows) + list(rebuild_relationships)
            if not row["source_chunk_ids"]
        ]
        if empty_relationship_ids:
            self.graph_store.execute_query(
                """
                UNWIND $rel_ids AS id
                MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
                WHERE r.graph_name = $graph_name AND r.id = id
                  AND size(coalesce(r.source_chunk_ids, [])) = 0
                DELETE r
                """,
                {
                    "graph_name": graph_name,
                    "rel_ids": sorted(set(empty_relationship_ids)),
                },
            )

        remaining_entity_ids = {row["id"] for row in entity_rows}
        rebuild_entities = self._rebuild_entity_rows(
            graph_document, state, entity_rows
        )
        if rebuild_entities:
            self.graph_store.execute_query(
                """
                UNWIND $entities AS row
                MATCH (e:__Entity__ {id: row.id, graph_name: $graph_name})
                SET e.description_sources = row.description_sources,
                    e.descriptions = row.descriptions,
                    e.observation_count = size(row.descriptions),
                    e.description = row.description_text,
                    e.embedding_input_fingerprint = row.embedding_input_fingerprint,
                    e.description_summary_status = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_status
                    END,
                    e.description_summary_error = CASE WHEN row.descriptions_changed THEN NULL ELSE e.description_summary_error END,
                    e.description_summary_updated = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_updated
                    END,
                    e.description_summary_fallback = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_fallback
                    END,
                    e.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_input_fingerprint
                    END
                """,
                {"graph_name": graph_name, "entities": rebuild_entities},
            )

        orphan_candidates = list(
            dict.fromkeys(
                [row["id"] for row in entity_rows]
                + [row["id"] for row in rebuild_entities]
                + [row["entity_id"] for row in state.evidence if row.get("entity_id")]
                + [row["entity_id"] for row in state.claims if row.get("entity_id")]
                + [row["source_id"] for row in state.relationships]
                + [row["target_id"] for row in state.relationships]
            )
        )
        orphan_candidates = [
            entity_id for entity_id in orphan_candidates
            if entity_id not in remaining_entity_ids and entity_id
        ]
        if orphan_candidates:
            self.graph_store.execute_query(
                """
                UNWIND $entity_ids AS id
                MATCH (e:__Entity__ {id: id, graph_name: $graph_name})
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) AS degree
                WHERE degree = 0
                DETACH DELETE e
                """,
                {"graph_name": graph_name, "entity_ids": orphan_candidates},
            )

    # ------------------------------------------------------------------
    # Row shapes
    # ------------------------------------------------------------------
    def _document_rows(self, documents: list) -> list[dict[str, Any]]:
        return [
            {
                "id": doc.id,
                "text_hash": doc.text_hash,
                "graph_name": doc.graph_name,
                "uid": _build_uid(doc.graph_name, doc.id),
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
                "uid": _build_uid(chunk.graph_name, chunk.id),
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

    def _evidence_link_rows(self, links: list) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": link.chunk_id,
                "entity_id": link.entity_id,
                "graph_name": link.graph_name,
            }
            for link in links
        ]

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


def _accessible_entries(
    stored_sources: Any,
    stored_descriptions: list[str],
    new_descriptions: list[str],
    document_id: str,
    record_id: str,
) -> list[list[str]]:
    """Reject unknown legacy ownership even when the replacement text matches."""
    entries = _description_entries(stored_sources)
    if set(stored_descriptions) <= {text for _, text in entries}:
        return entries
    raise ValueError(
        f"Cannot replace record {record_id!r}: its stored descriptions predate "
        "per-document provenance, so the text contributed by document "
        f"{document_id!r} cannot be identified. Rebuild the affected "
        "documents with the current SDK (re-ingest them) before replacing one "
        "of them."
    )


def _relationship_key(rel: Any) -> tuple[str, str, str]:
    """Edge identity as written: endpoints plus relationship type."""
    if isinstance(rel, dict):
        return (
            str(rel.get("source_id") or ""),
            str(rel.get("target_id") or ""),
            str(rel.get("type") or ""),
        )
    return (str(rel.source_id), str(rel.target_id), str(rel.type))


def _confirmed(result: Any, submitted: int) -> int:
    """Rows the database confirmed, or the submitted rows when it reports none."""
    # shortcut: an aggregate always returns one row on the Cypher backends, so an
    # empty result means the store did not report (test double, custom query
    # escape hatch); upgrade to a hard requirement if that ever changes.
    if result and isinstance(result[0], dict) and "written" in result[0]:
        return int(result[0]["written"])
    return submitted


def _replace_strength(
    stored: dict, document_id: str, strength: Any
) -> tuple[list[list[Any]], float | None]:
    entries = contribution_entries(stored.get("strength_sources"))
    if stored.get("strength") is not None and not entries:
        raise ValueError(
            f"Cannot replace record {stored.get('id')!r}: strength ownership is unknown; "
            "re-ingest the affected documents before replacement."
        )
    try:
        entries = [[owner, float(value)] for owner, value in entries]
        entries = [entry for entry in entries if entry[0] != document_id]
        if strength is not None:
            entries.append([document_id, float(strength)])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid strength provenance; re-ingest the affected documents.") from exc
    return entries, max((value for _, value in entries), default=None)
