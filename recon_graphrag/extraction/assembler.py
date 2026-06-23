"""Assemble validated extractions into a GraphDocument."""

from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    GraphExtraction,
    RelationshipRecord,
)


class GraphDocumentAssembler:
    def assemble(
        self,
        document_id: str,
        text_hash: str,
        chunks: list[TextChunk],
        chunk_extractions: dict[str, GraphExtraction],
        metadata: dict,
        graph_name: str,
    ) -> GraphDocument:
        entities_by_id = {}
        relationships_by_key = {}
        evidence_links = []

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
                entities_by_id.setdefault(
                    node.id,
                    EntityRecord(
                        id=node.id,
                        type=node.label,
                        properties=node.properties,
                        graph_name=graph_name,
                    ),
                )
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
        )
