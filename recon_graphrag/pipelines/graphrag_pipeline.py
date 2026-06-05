"""Knowledge graph construction pipeline.

Ingests text via internal extraction backend, then automatically
runs entity resolution and embedding (steps 1-3 of the full pipeline).

Steps 4-6 (community detection, summarization, community embedding)
are handled separately by the CommunityPipeline — typically on a schedule.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from recon_graphrag.communities.embeddings import CommunityEmbedder
from recon_graphrag.extraction.chunking import TextChunker, PageWindowBuilder
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.validator import SchemaValidator
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore
from recon_graphrag.graph.index_manager import IndexManager
from recon_graphrag.graph.neo4j_writer import Neo4jGraphWriter
from recon_graphrag.graph.writer import GraphWriter
from recon_graphrag.llm.base import BaseLLM


class GraphBuilderPipeline:
    """Build a knowledge graph from text using LLM entity extraction."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        schema: GraphSchema,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        graph_name: str = "entity-graph",
        graph_writer: Optional[GraphWriter] = None,
        perform_entity_resolution: bool = True,
        embed_entities: bool = True,
        fail_on_resolution_error: bool = False,
        fail_on_embedding_error: bool = False,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.schema = schema
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph_name = graph_name
        self.perform_entity_resolution = perform_entity_resolution
        self.embed_entity_nodes = embed_entities
        self.fail_on_resolution_error = fail_on_resolution_error
        self.fail_on_embedding_error = fail_on_embedding_error

        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.extractor = LLMGraphExtractor(llm)
        self.validator = SchemaValidator()
        self.assembler = GraphDocumentAssembler()
        self.graph_writer = graph_writer or Neo4jGraphWriter(graph_store)

    async def build_from_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Build knowledge graph from raw text.

        Uses internal character-level chunking and extraction, then automatically:
          - Step 2: Entity resolution (merge duplicates)
          - Step 3: Entity embedding (for local/DRIFT search)

        Steps 4-6 must be run separately via CommunityPipeline.
        """
        print("Step 1: Extracting entities and relationships...")
        result = await self._extract_and_write_text(text=text, metadata=metadata)

        print("Step 1b: Backfilling missing description properties...")
        self._backfill_descriptions()

        if self.perform_entity_resolution:
            print("Step 2: Resolving duplicate entities...")
            await self._resolve_entities()

        if self.embed_entity_nodes:
            print("Step 3: Embedding entities...")
            await self._embed_entities()

        validation = self._validate_graph_build()

        return {
            "extraction": result,
            "validation": validation,
        }

    async def build_from_pages(
        self,
        pages: list[str],
        metadata: Optional[dict] = None,
        window_size: int = 2,
        window_overlap: int = 1,
    ) -> dict:
        """Build knowledge graph from paginated text using sliding windows."""
        metadata = metadata or {}
        text = "\n\n".join(pages)
        document_id = self._make_document_id(text=text, metadata=metadata)
        text_hash = self._hash_text(text)

        window_builder = PageWindowBuilder(
            window_size=window_size,
            window_overlap=window_overlap,
        )
        chunks = window_builder.build_windows(
            pages=pages,
            document_id=document_id,
            metadata=metadata,
        )

        result = await self._extract_and_write_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
        )

        self._backfill_descriptions()

        if self.perform_entity_resolution:
            await self._resolve_entities()

        if self.embed_entity_nodes:
            await self._embed_entities()

        validation = self._validate_graph_build()

        return {
            "extraction": result,
            "validation": validation,
        }

    async def build_from_documents(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """Build knowledge graph from multiple documents."""
        results = []
        for doc in documents:
            result = await self.build_from_text(
                text=doc["text"],
                metadata=doc.get("metadata"),
            )
            results.append(result)
        return results

    async def _extract_and_write_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        metadata = metadata or {}
        document_id = self._make_document_id(text=text, metadata=metadata)
        text_hash = self._hash_text(text)

        chunks = self.chunker.chunk_text(
            text=text,
            document_id=document_id,
            metadata=metadata,
        )

        return await self._extract_and_write_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
        )

    async def _extract_and_write_chunks(
        self,
        document_id: str,
        text_hash: str,
        chunks: list,
        metadata: dict,
    ) -> dict:
        chunk_extractions = {}

        for chunk in chunks:
            try:
                raw_extraction = await self.extractor.extract(
                    text=chunk.text,
                    schema=self.schema,
                )
                validated = self.validator.validate(raw_extraction, self.schema)
                chunk_extractions[chunk.id] = validated
            except Exception as e:
                print(f"  Warning: extraction failed for chunk {chunk.id}: {e}")

        graph_document = self.assembler.assemble(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            chunk_extractions=chunk_extractions,
            metadata=metadata,
            graph_name=self.graph_name,
        )

        write_stats = self.graph_writer.write_graph_document(graph_document)

        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "write_stats": write_stats,
        }

    def _backfill_descriptions(self):
        """Set description = '' on __Entity__ nodes missing the property.

        Prevents Neo4j warnings when queries reference e.description / node.description
        on nodes whose schema doesn't define a description property.
        """
        self.graph_store.execute_query(
            "MATCH (e:__Entity__) WHERE e.description IS NULL SET e.description = ''"
        )

    async def _resolve_entities(self):
        """Step 2: Merge duplicate entities via SinglePropertyExactMatchResolver."""
        mgr = IndexManager(self.graph_store)
        try:
            await mgr.resolve_entities()
        except Exception as e:
            print(f"  Warning: entity resolution failed (APOC plugin required): {e}")
            if self.fail_on_resolution_error:
                raise

    async def _embed_entities(self):
        """Step 3: Generate vector embeddings for entity nodes."""
        embedder = CommunityEmbedder(self.graph_store, self.embedder)
        try:
            await embedder.embed_entities()
        except Exception as e:
            print(f"  Warning: entity embedding failed: {e}")
            if self.fail_on_embedding_error:
                raise

    def _validate_graph_build(self) -> dict:
        query = """
        CALL {
            MATCH (e:__Entity__)
            RETURN count(e) AS entity_count
        }
        CALL {
            MATCH (c:Chunk)
            RETURN count(c) AS chunk_count
        }
        CALL {
            MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__)
            RETURN count(r) AS evidence_link_count
        }
        CALL {
            MATCH (:__Entity__)-[r]-(:__Entity__)
            RETURN count(r) AS entity_relationship_count
        }
        RETURN entity_count,
               chunk_count,
               evidence_link_count,
               entity_relationship_count
        """
        result = self.graph_store.execute_query(query)
        return result[0] if result else {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_document_id(self, text: str, metadata: dict) -> str:
        source = metadata.get("source") or metadata.get("title")
        if source:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(source).lower()).strip("-")
            return f"doc:{slug}"

        return f"doc:{self._hash_text(text)[:16]}"
