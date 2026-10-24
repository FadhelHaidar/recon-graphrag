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
from dataclasses import dataclass, field
from typing import Any

from recon_graphrag.entity_text import entity_embedding_fingerprint
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
        "description_sources",
        "embedding_input_fingerprint",
        "embedding_content_fingerprint",
        "embedding_model_fingerprint",
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
    """Normalize stored description provenance to ``[document_id, text]`` pairs.

    A list of pairs survives APOC ``mergeNodes(properties: 'combine')`` as a
    concatenated list, which a nested map property does not.
    """
    entries: list[list[str]] = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            document_id, text = item
            if str(text).strip():
                entries.append([str(document_id), str(text)])
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
        self._reject_reserved_keys(graph_document)
        state = self._read_document_state(graph_document)
        entity_rows = self._entity_rows(graph_document, state)
        relationship_rows = self._relationship_rows(graph_document, state)

        self._write_documents([graph_document.document])
        self._write_chunks(graph_document.chunks)
        self._write_entities(graph_document.entities, entity_rows)
        self._write_evidence_links(graph_document.evidence_links)
        rebuild_relationships = self._rebuild_relationship_rows(
            graph_document, state, relationship_rows
        )
        self._write_relationships(graph_document, relationship_rows, rebuild_relationships)
        self._write_claims(graph_document.claims)
        self._reconcile(
            graph_document,
            state,
            entity_rows,
            relationship_rows,
            rebuild_relationships,
        )

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
                       coalesce(r.descriptions, []) AS descriptions,
                       r.description_sources AS description_sources,
                       r.description AS description,
                       r.description_summary_status AS description_summary_status,
                       r.strength AS strength,
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
                           coalesce(e.descriptions, []) AS descriptions,
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
            stored_descriptions = [str(d) for d in (stored.get("descriptions") or [])]
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
                    "canonical_key": entity.canonical_key,
                    "human_readable_id": entity.human_readable_id,
                    "description_sources": merged_entries,
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
            if rel.strength is not None:
                properties.pop("strength", None)
            description = str(properties.get("description") or "")
            stored = stored_by_key.get(_relationship_key(rel)) or {}
            stored_descriptions = [str(d) for d in (stored.get("descriptions") or [])]
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
                    "strength": _merged_strength(stored.get("strength"), rel.strength),
                    "description_sources": merged_entries,
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

    def _write_entities(
        self, entities: list, rows: list[dict[str, Any]]
    ) -> None:
        if not entities:
            return

        for entity_type, group in self._group_by_type(entities).items():
            label = escape_cypher_identifier(entity_type)
            group_ids = {entity.id for entity in group}
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
                    e.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_input_fingerprint
                    END,
                    e.description_summary_error = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE e.description_summary_error
                    END
                """,
                {"entities": [row for row in rows if row["id"] in group_ids]},
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

    def _write_relationships(
        self,
        graph_document: GraphDocument,
        rows: list[dict[str, Any]],
        rebuild: list[dict[str, Any]],
    ) -> None:
        """Write this document's relationships, plus provenance-only rebuilds."""
        relationships = graph_document.relationships
        by_key = {_relationship_key(row): row for row in rows}
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
                    r.created = coalesce(r.created, timestamp()),
                    r.source_chunk_ids = row.source_chunk_ids,
                    r.observation_count = row.observation_count,
                    r.weight = row.weight,
                    r.strength = row.strength,
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
                    r.description_input_fingerprint = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_input_fingerprint
                    END,
                    r.description_summary_error = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_error
                    END
                """,
                {
                    "relationships": [
                        by_key[_relationship_key(rel)]
                        for rel in group
                        if _relationship_key(rel) in by_key
                    ]
                },
            )

        if rebuild:
            self.graph_store.execute_query(
                """
                UNWIND $relationships AS row
                MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
                WHERE r.graph_name = $graph_name AND r.id = row.id
                SET r.source_chunk_ids = row.source_chunk_ids,
                    r.observation_count = row.observation_count,
                    r.weight = row.weight,
                    r.description_sources = row.description_sources,
                    r.descriptions = row.descriptions,
                    r.description = row.description_text,
                    r.description_summary_status = CASE
                        WHEN row.descriptions_changed THEN NULL
                        ELSE r.description_summary_status
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
            stored_descriptions = [str(d) for d in (stored.get("descriptions") or [])]
            entries, descriptions = _merge_contributions(
                _description_entries(stored.get("description_sources")),
                document_id,
                [],
            )
            chunk_ids = _surviving_chunk_ids(
                [str(chunk) for chunk in (stored.get("source_chunk_ids") or [])],
                state.chunk_ids,
                [],
            )
            changed = descriptions != stored_descriptions
            if not changed and chunk_ids == (stored.get("source_chunk_ids") or []):
                continue
            rows.append(
                {
                    "id": stored.get("id"),
                    "graph_name": graph_document.document.graph_name,
                    "source_chunk_ids": chunk_ids,
                    "observation_count": len(chunk_ids),
                    "weight": float(len(chunk_ids)),
                    "description_sources": entries,
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
            stored_descriptions = [str(d) for d in (stored.get("descriptions") or [])]
            entries, descriptions = _merge_contributions(
                _description_entries(stored.get("description_sources")),
                document_id,
                [],
            )
            changed = descriptions != stored_descriptions
            if not changed:
                continue
            description_text = (
                stored.get("description")
                if stored.get("description_summary_status") is not None
                else _joined(descriptions)
            )
            rows.append(
                {
                    "id": stored["id"],
                    "description_sources": entries,
                    "descriptions": descriptions,
                    "description_text": description_text,
                    "descriptions_changed": True,
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
    """Provenance for an existing record, or a targeted legacy diagnostic.

    Records written before per-document provenance existed hold description
    text that cannot be attributed. An identical rewrite can safely adopt that
    text as this document's contribution; a changed rewrite cannot tell whose
    text it would delete, so it fails and asks for a rebuild instead.
    """
    entries = _description_entries(stored_sources)
    if entries or not stored_descriptions:
        return entries
    if set(stored_descriptions) <= set(new_descriptions):
        return [[document_id, text] for text in stored_descriptions]
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


def _merged_strength(existing: Any, new: Any) -> float | None:
    """Keep the strongest observed contribution.

    shortcut: strength is not tracked per document, so a replaced document's
    stronger value can survive until the edge itself is replaced; track it in
    ``description_sources``-style provenance if per-document strength matters.
    """
    values = [float(value) for value in (existing, new) if value is not None]
    return max(values) if values else None
