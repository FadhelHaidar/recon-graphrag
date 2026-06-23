"""Assemble validated extractions into a GraphDocument."""

from __future__ import annotations

from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    ExtractedClaim,
    GraphDocument,
    GraphExtraction,
    RelationshipRecord,
)
from recon_graphrag.models.artifacts import ClaimRecord, DescriptionObservation, SourceReference


# Maximum characters for consolidated entity descriptions.
DESCRIPTION_CONSOLIDATION_CAP = 2000


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
    ) -> GraphDocument:
        entities_by_id: dict[str, EntityRecord] = {}
        relationships_by_key: dict[tuple, RelationshipRecord] = {}
        evidence_links = []
        claim_records: list[ClaimRecord] = []

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

            for node in extraction.nodes:
                if node.id not in entities_by_id:
                    entities_by_id[node.id] = EntityRecord(
                        id=node.id,
                        type=node.label,
                        properties=node.properties,
                        graph_name=graph_name,
                    )

                # Accumulate description observation
                description = node.properties.get("description", "")
                observation = DescriptionObservation(
                    entity_id=node.id,
                    description=description,
                    source=_build_source_ref(document_id, chunk, metadata),
                )
                entities_by_id[node.id].description_observations.append(observation)

                evidence_links.append(
                    EvidenceLink(
                        chunk_id=chunk.id,
                        entity_id=node.id,
                        graph_name=graph_name,
                    )
                )

            for rel in extraction.relationships:
                key = (rel.source_id, rel.type, rel.target_id)

                if key not in relationships_by_key:
                    extracted_weight = rel.properties.get("weight")
                    strength = (
                        float(extracted_weight) if extracted_weight is not None else None
                    )
                    relationships_by_key[key] = RelationshipRecord(
                        id=f"{rel.source_id}:{rel.type}:{rel.target_id}",
                        source_id=rel.source_id,
                        target_id=rel.target_id,
                        type=rel.type,
                        properties={
                            **rel.properties,
                            "source_chunk_ids": [chunk.id],
                            "weight": 1.0,
                        },
                        graph_name=graph_name,
                        observation_count=1,
                        strength=strength,
                    )
                else:
                    existing = relationships_by_key[key]
                    existing.observation_count += 1
                    existing.properties["weight"] = float(existing.observation_count)
                    existing.properties.setdefault("source_chunk_ids", []).append(
                        chunk.id
                    )

            # Convert extracted claims to ClaimRecord
            if chunk_claims:
                for extracted_claim in chunk_claims.get(chunk.id, []):
                    claim_id = _build_claim_id(
                        document_id, chunk.id, extracted_claim
                    )
                    claim_records.append(
                        ClaimRecord(
                            id=claim_id,
                            entity_id=extracted_claim.subject_entity_id,
                            claim_type=extracted_claim.claim_type,
                            description=extracted_claim.description,
                            source=_build_source_ref(document_id, chunk, metadata),
                            status=extracted_claim.status,
                            graph_name=graph_name,
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
        )


def _build_claim_id(
    document_id: str, chunk_id: str, claim: ExtractedClaim
) -> str:
    """Build a deterministic claim ID from document, chunk, and claim content."""
    import hashlib

    content = f"{claim.subject_entity_id}:{claim.claim_type}:{claim.description}"
    short_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"claim:{document_id}:{chunk_id}:{short_hash}"


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
