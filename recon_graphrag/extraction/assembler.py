"""Assemble validated extractions into a GraphDocument."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    ExtractedClaim,
    ExtractedNode,
    GraphDocument,
    GraphExtraction,
    RelationshipRecord,
)
from recon_graphrag.models.artifacts import (
    ClaimRecord,
    DescriptionObservation,
    SourceReference,
)

# Maximum characters for consolidated entity descriptions.
DESCRIPTION_CONSOLIDATION_CAP = 2000


def _empty(value: Any) -> bool:
    """Missing observations do not conflict; zero and False are populated."""
    return value is None or value == "" or value == [] or value == {}


def _plan_entity_keys(
    document_id: str,
    chunks: list[TextChunk],
    extractions: dict[str, GraphExtraction],
    metadata: dict,
    graph_name: str,
    identity_properties: dict[str, str] | None,
    schema: GraphSchema | None,
    conflict_properties: dict[str, list[str]] | list[str] | None,
) -> tuple[dict[tuple[str, int], str], set[str], list[dict[str, Any]]]:
    """Block ambiguous candidates before resolving any local endpoints."""
    identities = {label: prop for label, prop in (identity_properties or {}).items() if prop != "name"}
    if schema:
        identities.update({
            node.label: node.identity_property for node in schema.node_types
            if node.identity_property != "name"
        })
    groups: dict[tuple[str, str, bool], list[tuple[TextChunk, int, ExtractedNode, str]]] = {}
    labels_by_key = {}
    trust_by_key = {}
    for chunk in chunks:
        extraction = extractions.get(chunk.id)
        if not extraction:
            continue
        for index, node in enumerate(extraction.nodes):
            identity_prop = identities.get(node.label)
            identity = node.properties.get(identity_prop) if identity_prop else None
            explicit = node.properties.get("canonical_key")
            valid_identity = isinstance(identity, (str, int, float, bool)) and not _empty(identity)
            trusted = valid_identity or (bool(explicit) and node.canonical_key_trusted)
            if valid_identity:
                key = f"{node.label}:{identity}"
            elif explicit:
                key = str(explicit)
            else:
                key = f"{document_id}:{chunk.id}:{node.label}:{node.id}"
            # Untrusted LLM observations stay local, but share a review group by name.
            candidate = key if explicit or trusted else str(node.properties.get("name") or key)
            group = groups.setdefault((node.label, candidate, trusted), [])
            group.append((chunk, index, node, key))
            labels_by_key.setdefault(key, set()).add(node.label)
            trust_by_key.setdefault(key, set()).add(trusted)

    if any(len(labels) > 1 and True in trust_by_key[key]
           for key, labels in labels_by_key.items()):
        raise ValueError("Cross-label canonical-key collision involving a trusted identity.")

    reserved_keys = set(labels_by_key)
    assigned_keys: set[str] = set()
    transformed_keys: dict[tuple, str] = {}
    keys = {}
    trusted_keys = set()
    reviews = []
    for (label, _, trusted), observations in groups.items():
        node_type = schema.get_node_type(label) if schema else None
        list_props = {
            prop.name for prop in node_type.properties if prop.type == "LIST"
        } if node_type else set()
        ignored = {"canonical_key", "human_readable_id", "description"}
        if not trusted:
            ignored.update({"name", "title"})
        configured = conflict_properties or []
        if isinstance(configured, dict):
            configured = sum((configured.get(key, []) for key in ("*", "__default__", label)), [])
        names = dict.fromkeys(
            name for _, _, node, _ in observations for name in node.properties
            if name not in ignored and (not trusted or name not in list_props)
        )
        names.update(dict.fromkeys(name for name in configured if name not in list_props))
        conflicts = []
        for name in names:
            values = [
                {"value": node.properties[name], "source": asdict(_build_source_ref(document_id, chunk, metadata))}
                for chunk, _, node, _ in observations
                if not _empty(node.properties.get(name))
            ]
            if values and any(item["value"] != values[0]["value"] for item in values[1:]):
                conflicts.append({"property": name, "observations": values})
        scalar_conflict = any(conflict["property"] not in list_props for conflict in conflicts)
        entity_keys = []
        for chunk, index, node, key in observations:
            original_key = key
            mixed_trust = len(trust_by_key[key]) > 1
            if len(labels_by_key[key]) > 1:
                key = f"{node.label}:{key}"
            if mixed_trust and not trusted:
                key = f"{key}:untrusted"
            if scalar_conflict and not trusted:
                # Occurrence-scoped variants are stable for the same document/chunks.
                key = f"{key}:provisional:{uuid.uuid5(uuid.NAMESPACE_URL, f'{document_id}:{chunk.id}:{index}')}"
            if key != original_key:
                occurrence = (chunk.id, index) if scalar_conflict and not trusted else None
                allocation_identity = (label, original_key, trusted, occurrence)
                if allocation_identity not in transformed_keys:
                    candidate = key
                    suffix = 0
                    # Generated keys must not take any original or allocated identity's key.
                    while candidate in reserved_keys or candidate in assigned_keys:
                        suffix += 1
                        candidate = f"{key}:{suffix}"
                    transformed_keys[allocation_identity] = candidate
                    assigned_keys.add(candidate)
                key = transformed_keys[allocation_identity]
            keys[(chunk.id, index)] = key
            if trusted:
                trusted_keys.add(key)
            if key not in entity_keys:
                entity_keys.append(key)
        if conflicts:
            reviews.append({
                "domain_label": label,
                "reason": "property_conflict",
                "decision": ("identity_retained" if trusted else
                             "blocked" if scalar_conflict else "review_required"),
                "entity_ids": [_build_entity_uuid(graph_name, key) for key in entity_keys],
                "canonical_keys": entity_keys,
                "conflicts": conflicts,
                "scores": {"fuzzy": None, "embedding": None, "llm": None},
            })
    return keys, trusted_keys, reviews


class GraphDocumentAssembler:
    def assemble(
        self,
        document_id: str,
        text_hash: str,
        chunks: list[TextChunk],
        chunk_extractions: dict[str, GraphExtraction],
        metadata: dict,
        graph_name: str,
        chunk_claims: dict[str, list[ExtractedClaim]] | None = None,
        identity_properties: dict[str, str] | None = None,
        schema: GraphSchema | None = None,
        conflict_properties: dict[str, list[str]] | list[str] | None = None,
    ) -> GraphDocument:
        entities_by_id: dict[str, EntityRecord] = {}
        relationships_by_key: dict[tuple, RelationshipRecord] = {}
        evidence_links = []
        claim_records: list[ClaimRecord] = []

        node_keys, trusted_keys, review_groups = _plan_entity_keys(
            document_id, chunks, chunk_extractions, metadata, graph_name,
            identity_properties, schema, conflict_properties,
        )
        list_properties = {
            node.label: {prop.name for prop in node.properties if prop.type == "LIST"}
            for node in schema.node_types
        } if schema else {}

        chunk_records = [
            ChunkRecord(
                id=chunk.id,
                document_id=document_id,
                text=chunk.text,
                index=chunk.index,
                metadata=chunk.metadata,
                graph_name=graph_name,
            )
            for chunk in chunks
        ]

        for chunk in chunks:
            extraction = chunk_extractions.get(chunk.id)
            if not extraction:
                continue

            # F7: extracted local IDs ("p1") are chunk-local references, not
            # identity. Map each to its entity's persistent canonical key so
            # endpoints and claims resolve through the same mapping.
            chunk_keys: dict[str, str] = {}

            for index, node in enumerate(extraction.nodes):
                canonical_key = node_keys[(chunk.id, index)]
                chunk_keys[node.id] = canonical_key

                entity_id = _build_entity_uuid(graph_name, canonical_key)
                if canonical_key not in entities_by_id:
                    properties = dict(node.properties)
                    title = str(
                        properties.get("title")
                        or properties.get("name")
                        or canonical_key
                    )
                    properties.setdefault("name", title)
                    properties.setdefault("title", title)
                    explicit = bool(properties.get("canonical_key"))
                    readable = (
                        str(properties.get("human_readable_id") or canonical_key)
                        if explicit
                        else str(node.id)
                    )
                    properties["canonical_key"] = canonical_key
                    properties["human_readable_id"] = readable
                    entities_by_id[canonical_key] = EntityRecord(
                        id=entity_id,
                        type=node.label,
                        properties=properties,
                        graph_name=graph_name,
                        canonical_key=canonical_key,
                        human_readable_id=readable,
                    )
                else:
                    properties = entities_by_id[canonical_key].properties
                    # Keep the first populated value; later observations fill gaps.
                    for key, value in node.properties.items():
                        if _empty(properties.get(key)) and not _empty(value):
                            properties[key] = value
                        elif (canonical_key in trusted_keys
                              and key in list_properties.get(node.label, set())
                              and isinstance(properties.get(key), list)
                              and isinstance(value, list)):
                            merged = []
                            for item in properties[key] + value:
                                if item not in merged:
                                    merged.append(item)
                            properties[key] = merged

                # Accumulate description observation
                description = node.properties.get("description", "")
                observation = DescriptionObservation(
                    entity_id=entity_id,
                    description="" if description is None else str(description),
                    source=_build_source_ref(document_id, chunk, metadata),
                )
                entities_by_id[canonical_key].description_observations.append(
                    observation
                )

                evidence_links.append(
                    EvidenceLink(
                        chunk_id=chunk.id,
                        entity_id=entity_id,
                        graph_name=graph_name,
                    )
                )

            for rel in extraction.relationships:
                source_key = chunk_keys.get(str(rel.source_id), str(rel.source_id))
                target_key = chunk_keys.get(str(rel.target_id), str(rel.target_id))
                key = (source_key, rel.type, target_key)
                source_id = _build_entity_uuid(graph_name, source_key)
                target_id = _build_entity_uuid(graph_name, target_key)
                relationship_key = f"{source_key}:{rel.type}:{target_key}"

                if key not in relationships_by_key:
                    extracted_weight = rel.properties.get("weight")
                    strength = (
                        float(extracted_weight)
                        if extracted_weight is not None
                        else None
                    )
                    properties = {
                        **rel.properties,
                        "canonical_key": relationship_key,
                        "human_readable_id": relationship_key,
                        "source_canonical_key": source_key,
                        "target_canonical_key": target_key,
                        "source_chunk_ids": [chunk.id],
                        "weight": 1.0,
                    }
                    relationships_by_key[key] = RelationshipRecord(
                        id=_build_relationship_uuid(graph_name, relationship_key),
                        source_id=source_id,
                        target_id=target_id,
                        type=rel.type,
                        properties=properties,
                        graph_name=graph_name,
                        observation_count=1,
                        strength=strength,
                    )
                else:
                    existing = relationships_by_key[key]
                    existing.observation_count += 1
                    # ponytail: strength is first-observation-wins within a doc.
                    # Inert while Leiden weights by `weight` (frequency); switch
                    # to max(existing.strength, this weight) if `strength` ever
                    # becomes the community/retrieval weight property.
                    existing.properties["weight"] = float(existing.observation_count)
                    existing.properties.setdefault("source_chunk_ids", []).append(
                        chunk.id
                    )

            # Convert extracted claims to ClaimRecord
            if chunk_claims:
                for extracted_claim in chunk_claims.get(chunk.id, []):
                    subject_key = chunk_keys.get(
                        str(extracted_claim.subject_entity_id),
                        str(extracted_claim.subject_entity_id),
                    )
                    object_key = (
                        chunk_keys.get(
                            str(extracted_claim.object_entity_id),
                            str(extracted_claim.object_entity_id),
                        )
                        if extracted_claim.object_entity_id
                        else None
                    )
                    claim_id = _build_claim_id(document_id, chunk.id, extracted_claim)
                    claim_records.append(
                        ClaimRecord(
                            id=claim_id,
                            entity_id=_build_entity_uuid(graph_name, subject_key),
                            claim_type=extracted_claim.claim_type,
                            description=extracted_claim.description,
                            source=_build_source_ref(document_id, chunk, metadata),
                            status=extracted_claim.status,
                            graph_name=graph_name,
                            start_date=extracted_claim.start_date,
                            end_date=extracted_claim.end_date,
                            object_entity_id=(
                                _build_entity_uuid(graph_name, object_key)
                                if object_key
                                else None
                            ),
                            source_text=extracted_claim.source_text or chunk.text[:300],
                            text_unit_id=extracted_claim.text_unit_id or chunk.id,
                        )
                    )

        # Consolidate descriptions into properties["description"]
        for entity in entities_by_id.values():
            consolidated = _consolidate_descriptions(entity.description_observations)
            entity.properties["description"] = consolidated

        return GraphDocument(
            document=DocumentRecord(
                id=document_id,
                text_hash=text_hash,
                metadata=metadata,
                graph_name=graph_name,
            ),
            chunks=chunk_records,
            entities=list(entities_by_id.values()),
            relationships=list(relationships_by_key.values()),
            evidence_links=evidence_links,
            claims=claim_records,
            assembly_review_groups=review_groups,
        )


def _build_claim_id(document_id: str, chunk_id: str, claim: ExtractedClaim) -> str:
    """Build a deterministic claim ID from document, chunk, and claim content."""
    import hashlib

    content = f"{claim.subject_entity_id}:{claim.claim_type}:{claim.description}"
    short_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"claim:{document_id}:{chunk_id}:{short_hash}"


def _build_entity_uuid(graph_name: str, canonical_key: str) -> str:
    """Build an idempotent UUID for a graph-scoped entity canonical key."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"recon-graphrag/entity/{graph_name}/{canonical_key}",
        )
    )


def _build_relationship_uuid(graph_name: str, canonical_key: str) -> str:
    """Build an idempotent UUID for a graph-scoped relationship canonical key."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"recon-graphrag/relationship/{graph_name}/{canonical_key}",
        )
    )


def _build_source_ref(
    document_id: str, chunk: TextChunk, metadata: dict
) -> SourceReference:
    """Build a SourceReference from chunk metadata."""
    return SourceReference(
        document_id=document_id,
        chunk_id=chunk.id,
        document_name=metadata.get("title") or metadata.get("source"),
        page_start=chunk.metadata.get("page_start"),
        page_end=chunk.metadata.get("page_end"),
        char_start=chunk.metadata.get("char_start"),
        char_end=chunk.metadata.get("char_end"),
    )


def _consolidate_descriptions(
    observations: list[DescriptionObservation],
    cap: int = DESCRIPTION_CONSOLIDATION_CAP,
) -> str:
    """Deterministically consolidate description observations.

    1. Collect non-empty, unique descriptions.
    2. Sort by source document/chunk for determinism.
    3. Join with "; " within a character cap.
    """
    seen: set[str] = set()
    unique: list[DescriptionObservation] = []
    for obs in observations:
        normalized = obs.description.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(obs)

    unique.sort(key=lambda o: (o.source.document_id, o.source.chunk_id))

    parts = [o.description.strip() for o in unique]
    result = "; ".join(parts)

    if len(result) > cap:
        result = result[:cap] + "..."

    return result
