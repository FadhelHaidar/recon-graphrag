"""Backend-neutral graph writer helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore


class BaseGraphWriter(ABC):
    """Shared writer orchestration and row preparation.

    Backend subclasses own the Cypher and dialect-specific escaping.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def write_graph_document(self, graph_document: GraphDocument) -> dict[str, int]:
        self._write_documents([graph_document.document])
        self._write_chunks(graph_document.chunks)
        self._write_entities(graph_document.entities)
        self._write_evidence_links(graph_document.evidence_links)
        self._write_relationships(graph_document.relationships)
        self._write_claims(graph_document.claims)

        return self._stats_for(graph_document)

    @abstractmethod
    def _write_documents(self, documents: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _write_chunks(self, chunks: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _write_entities(self, entities: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _write_evidence_links(self, links: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _write_relationships(self, relationships: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _write_claims(self, claims: list) -> None:
        raise NotImplementedError

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
                "graph_name": claim.graph_name,
            }
            for claim in claims
        ]

    def _group_by_type(self, records: list) -> dict[str, list]:
        grouped: dict[str, list] = defaultdict(list)
        for record in records:
            grouped[record.type].append(record)
        return dict(grouped)
