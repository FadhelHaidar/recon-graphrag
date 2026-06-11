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
from recon_graphrag.graphdb.base import GraphStore, GraphWriter
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
        self.graph_writer = graph_writer or graph_store

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
        metadata = metadata or {}
        document_id = self._make_document_id(text=text, metadata=metadata)
        print(f"Starting graph build from text: document_id={document_id} chars={len(text)}")

        result = await self._extract_and_write_text(text=text, metadata=metadata)

        print("Backfilling missing entity descriptions")
        self._backfill_descriptions()

        if self.perform_entity_resolution:
            print("Resolving duplicate entities")
            await self._resolve_entities()

        if self.embed_entity_nodes:
            print("Embedding entity nodes")
            await self._embed_entities()

        print("Validating graph build")
        validation = self._validate_graph_build()

        print(f"Graph build complete: document_id={document_id}")
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

        print(
            f"Starting graph build from pages: document_id={document_id} "
            f"pages={len(pages)} chars={len(text)} window_size={window_size} window_overlap={window_overlap}"
        )

        window_builder = PageWindowBuilder(
            window_size=window_size,
            window_overlap=window_overlap,
        )
        chunks = window_builder.build_windows(
            pages=pages,
            document_id=document_id,
            metadata=metadata,
        )

        if chunks:
            print(
                f"Built page windows: document_id={document_id} chunks={len(chunks)} "
                f"first_page={chunks[0].metadata.get('page_start')} last_page={chunks[-1].metadata.get('page_end')}"
            )
        else:
            print(f"Built page windows: document_id={document_id} chunks=0")

        result = await self._extract_and_write_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
        )

        print("Backfilling missing entity descriptions")
        self._backfill_descriptions()

        if self.perform_entity_resolution:
            print("Resolving duplicate entities")
            await self._resolve_entities()

        if self.embed_entity_nodes:
            print("Embedding entity nodes")
            await self._embed_entities()

        print("Validating graph build")
        validation = self._validate_graph_build()

        print(f"Graph build complete: document_id={document_id}")
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
        extraction_errors = {}
        total = len(chunks)

        print(f"Starting extraction: document_id={document_id} chunks={total}")

        for i, chunk in enumerate(chunks, start=1):
            print(f"[{i}/{total}] Extracting chunk {chunk.id} ...")
            try:
                raw_extraction = await self.extractor.extract(
                    text=chunk.text,
                    schema=self.schema,
                )
                validated = self.validator.validate(raw_extraction, self.schema)
                chunk_extractions[chunk.id] = validated
                node_count = len(validated.nodes)
                rel_count = len(validated.relationships)
                print(f"  [{i}/{total}] ✓ chunk {chunk.id} extracted ({node_count} nodes, {rel_count} rels)")
            except Exception as e:
                extraction_errors[chunk.id] = e
                print(f"  [{i}/{total}] ✗ chunk {chunk.id} failed")

        print(
            f"Extraction complete: {len(chunk_extractions)}/{total} succeeded, "
            f"{len(extraction_errors)}/{total} failed."
        )

        if chunks and not chunk_extractions:
            first_chunk_id, first_error = next(iter(extraction_errors.items()))
            raise RuntimeError(
                f"Extraction failed for all {len(chunks)} chunk(s). "
                f"First failure was for {first_chunk_id}: {first_error}"
            ) from first_error

        graph_document = self.assembler.assemble(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            chunk_extractions=chunk_extractions,
            metadata=metadata,
            graph_name=self.graph_name,
        )

        write_stats = self.graph_writer.write_graph_document(graph_document)
        print(
            f"Writing graph document to {self._graph_store_name()} "
            f"({write_stats.get('entities')} entities, {write_stats.get('relationships')} relationships) ..."
        )
        print(f"Write complete: {write_stats}")

        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "write_stats": write_stats,
        }

    def _chunk_log_extra(self, chunk) -> str:
        """Return a formatted string with chunk metadata for printing."""
        parts = []
        if "page_start" in chunk.metadata:
            parts.append(f" page_start={chunk.metadata['page_start']}")
        if "page_end" in chunk.metadata:
            parts.append(f" page_end={chunk.metadata['page_end']}")
        if "char_start" in chunk.metadata:
            parts.append(f" char_start={chunk.metadata['char_start']}")
        if "char_end" in chunk.metadata:
            parts.append(f" char_end={chunk.metadata['char_end']}")
        return "".join(parts)

    def _backfill_descriptions(self):
        """Set description = '' on __Entity__ nodes missing the property."""
        self.graph_store.backfill_descriptions()

    async def _resolve_entities(self):
        """Step 2: Merge duplicate entities with the internal resolver."""
        try:
            result = await self.graph_store.resolve_entities()
            if isinstance(result, dict) and result.get("skipped"):
                print(f"Entity resolution skipped: reason={result.get('reason')}")
        except Exception as e:
            print(f"Entity resolution failed: {e}")
            if self.fail_on_resolution_error:
                raise

    async def _embed_entities(self):
        """Step 3: Generate vector embeddings for entity nodes."""
        embedder = CommunityEmbedder(self.graph_store, self.embedder)
        try:
            await embedder.embed_entities()
        except Exception as e:
            print(f"Entity embedding failed: {e}")
            if self.fail_on_embedding_error:
                raise

    def _validate_graph_build(self) -> dict:
        return self.graph_store.validate_graph_build()

    def _graph_store_name(self) -> str:
        """Return a human-readable name derived from the graph store class."""
        class_name = type(self.graph_store).__name__
        suffix = "GraphStore"
        if class_name.endswith(suffix) and len(class_name) > len(suffix):
            return class_name[: -len(suffix)]
        return class_name

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_document_id(self, text: str, metadata: dict) -> str:
        source = metadata.get("source") or metadata.get("title")
        if source:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(source).lower()).strip("-")
            return f"doc:{slug}"

        return f"doc:{self._hash_text(text)[:16]}"
