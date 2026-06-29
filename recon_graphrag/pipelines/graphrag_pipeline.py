"""Knowledge graph construction pipeline.

Ingests text via internal extraction backend, then automatically
runs entity resolution and embedding (steps 1-3 of the full pipeline).

Steps 4-5 (community detection and summarization) are handled separately by the
CommunityPipeline, typically on a schedule.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Optional

from recon_graphrag.embeddings import EntityEmbedder
from recon_graphrag.extraction.chunking import (
    PageWindowBuilder,
    TextChunker,
    _page_text,
)
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.validator import SchemaValidator
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore, GraphWriter
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.utils.tokens import TokenCounter, TiktokenTokenCounter


class GraphBuilderPipeline:
    """Build a knowledge graph from text using LLM entity extraction."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        schema: GraphSchema,
        graph_name: str = "entity-graph",
        graph_writer: Optional[GraphWriter] = None,
        extraction_concurrency: int = 5,
        max_gleanings: int = 1,
        extract_claims: bool = False,
        perform_entity_resolution: bool = True,
        entity_resolution_strategy: str = "normalized",
        entity_resolution_aliases: Optional[dict] = None,
        entity_resolution_llm_guidance: Optional[str] = None,
        entity_resolution_context_properties: Optional[
            dict[str, list[str]] | list[str]
        ] = None,
        entity_resolution_conflict_properties: Optional[
            dict[str, list[str]] | list[str]
        ] = None,
        entity_resolution_context_mode: str = "safe_defaults",
        allow_ai_auto_merge: bool = False,
        embed_entities: bool = True,
        fail_on_resolution_error: bool = False,
        fail_on_embedding_error: bool = False,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.schema = schema
        self.graph_name = graph_name
        self.extraction_concurrency = extraction_concurrency
        self.max_gleanings = max_gleanings
        self.extract_claims = extract_claims
        self.perform_entity_resolution = perform_entity_resolution
        self.entity_resolution_strategy = entity_resolution_strategy
        self.entity_resolution_aliases = entity_resolution_aliases
        self.entity_resolution_llm_guidance = entity_resolution_llm_guidance
        self.entity_resolution_context_properties = (
            entity_resolution_context_properties
        )
        self.entity_resolution_conflict_properties = (
            entity_resolution_conflict_properties
        )
        self.entity_resolution_context_mode = entity_resolution_context_mode
        self.allow_ai_auto_merge = allow_ai_auto_merge
        self.embed_entity_nodes = embed_entities
        self.fail_on_resolution_error = fail_on_resolution_error
        self.fail_on_embedding_error = fail_on_embedding_error

        self.extractor = LLMGraphExtractor(llm)
        self.validator = SchemaValidator()
        self.assembler = GraphDocumentAssembler()
        self.graph_writer = graph_writer or graph_store

    async def build_from_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        chunk_unit: str = "characters",
        token_counter: TokenCounter | None = None,
        token_encoding: str = "cl100k_base",
    ) -> dict:
        """Build knowledge graph from raw text.

        Uses internal character-level chunking and extraction, then automatically:
          - Step 2: Entity resolution (merge duplicates)
          - Step 3: Entity embedding (for local/DRIFT search)

        Steps 4-5 must be run separately via CommunityPipeline.
        """
        metadata = metadata or {}
        document_id = self._make_document_id(text=text, metadata=metadata)
        text_hash = self._hash_text(text)
        print(f"Starting graph build from text: document_id={document_id} chars={len(text)}")

        chunker = self._make_text_chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_unit=chunk_unit,
            token_counter=token_counter,
            token_encoding=token_encoding,
        )
        chunks = chunker.chunk_text(
            text=text,
            document_id=document_id,
            metadata=metadata,
        )

        result = await self._build_from_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
        )

        print(f"Graph build complete: document_id={document_id}")
        return result

    async def build_from_documents(
        self,
        documents: list[dict],
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        chunk_unit: str = "characters",
        token_counter: TokenCounter | None = None,
        token_encoding: str = "cl100k_base",
        window_size: int = 2,
        window_overlap: int = 1,
    ) -> list[dict]:
        """Build knowledge graph from multiple document envelopes.

        Each envelope must contain exactly one of:
          - ``text``: a raw text string.
          - ``pages``: a list of page strings or page dicts with ``text``.

        Envelopes may optionally include ``metadata``.

        Returns one result dict per input envelope.
        """
        self._validate_document_envelopes(documents)

        results = []
        for envelope in documents:
            metadata = envelope.get("metadata") or {}
            if "text" in envelope:
                result = await self.build_from_text(
                    text=envelope["text"],
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_unit=chunk_unit,
                    token_counter=token_counter,
                    token_encoding=token_encoding,
                )
            else:
                result = await self._build_from_pages_envelope(
                    pages=envelope["pages"],
                    metadata=metadata,
                    window_size=window_size,
                    window_overlap=window_overlap,
                )
            results.append(result)
        return results

    async def _build_from_pages_envelope(
        self,
        pages: list[str | dict],
        metadata: dict,
        window_size: int,
        window_overlap: int,
    ) -> dict:
        text = "\n\n".join(_page_text(page) for page in pages)
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

        result = await self._build_from_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
        )

        print(f"Graph build complete: document_id={document_id}")
        return result

    def _validate_document_envelopes(self, documents: list[dict]) -> None:
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")

        for envelope in documents:
            if not isinstance(envelope, dict):
                raise ValueError("each document envelope must be a dict")

            has_text = "text" in envelope
            has_pages = "pages" in envelope

            if has_text and has_pages:
                raise ValueError("document envelope must not contain both 'text' and 'pages'")
            if not has_text and not has_pages:
                raise ValueError("document envelope must contain either 'text' or 'pages'")

            if has_text and not isinstance(envelope["text"], str):
                raise ValueError("document envelope 'text' must be a string")
            if has_pages and not isinstance(envelope["pages"], list):
                raise ValueError("document envelope 'pages' must be a list")

            metadata = envelope.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("document envelope 'metadata' must be a dict")

    def _make_text_chunker(
        self,
        chunk_size: int,
        chunk_overlap: int,
        chunk_unit: str,
        token_counter: TokenCounter | None,
        token_encoding: str,
    ) -> TextChunker:
        if chunk_unit == "tokens" and token_counter is None:
            token_counter = TiktokenTokenCounter(model=token_encoding)

        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            unit=chunk_unit,
            token_counter=token_counter,
        )

    async def _build_from_chunks(
        self,
        document_id: str,
        text_hash: str,
        chunks: list,
        metadata: dict,
    ) -> dict:
        extraction = await self._extract_and_write_chunks(
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

        return {
            "extraction": extraction,
            "validation": validation,
        }

    async def _extract_and_write_chunks(
        self,
        document_id: str,
        text_hash: str,
        chunks: list,
        metadata: dict,
    ) -> dict:
        chunk_extractions = {}
        chunk_claims = {}
        extraction_errors = {}
        total = len(chunks)

        print(
            f"Starting extraction: document_id={document_id} chunks={total} "
            f"concurrency={self.extraction_concurrency}"
        )

        semaphore = asyncio.Semaphore(self.extraction_concurrency)

        async def _extract_one(i: int, chunk):
            async with semaphore:
                print(f"[{i}/{total}] Extracting chunk {chunk.id} ...")
                try:
                    raw_extraction = await self.extractor.extract(
                        text=chunk.text,
                        schema=self.schema,
                        max_gleanings=self.max_gleanings,
                    )
                    validated = self.validator.validate(raw_extraction, self.schema)
                    node_count = len(validated.nodes)
                    rel_count = len(validated.relationships)
                    print(
                        f"  [{i}/{total}] OK chunk {chunk.id} extracted "
                        f"({node_count} nodes, {rel_count} rels)"
                    )

                    # Optionally extract claims
                    claims = []
                    if self.extract_claims and validated.nodes:
                        entity_ids = [n.id for n in validated.nodes]
                        try:
                            claims = await self.extractor.extract_claims(
                                text=chunk.text,
                                entity_ids=entity_ids,
                            )
                            print(
                                f"  [{i}/{total}] Claims chunk {chunk.id}: "
                                f"{len(claims)} claims"
                            )
                        except Exception as ce:
                            print(
                                f"  [{i}/{total}] Claims chunk {chunk.id} failed: {ce}"
                            )

                    return chunk.id, validated, claims, None
                except Exception as e:
                    print(f"  [{i}/{total}] FAIL chunk {chunk.id} failed")
                    return chunk.id, None, [], e

        tasks = [
            asyncio.create_task(_extract_one(i, chunk))
            for i, chunk in enumerate(chunks, start=1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in results:
            if isinstance(item, Exception):
                raise item
            chunk_id, validated, claims, error = item
            if error is not None:
                extraction_errors[chunk_id] = error
            else:
                chunk_extractions[chunk_id] = validated
                if claims:
                    chunk_claims[chunk_id] = claims

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

        total_claims = sum(len(c) for c in chunk_claims.values())
        if total_claims:
            print(f"Total claims extracted: {total_claims}")

        graph_document = self.assembler.assemble(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            chunk_extractions=chunk_extractions,
            metadata=metadata,
            graph_name=self.graph_name,
            chunk_claims=chunk_claims if chunk_claims else None,
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

    def _backfill_descriptions(self):
        """Set description = '' on __Entity__ nodes missing the property."""
        self.graph_store.backfill_descriptions()

    async def _resolve_entities(self):
        """Step 2: Merge duplicate entities with the internal resolver."""
        try:
            kwargs = {
                "graph_name": self.graph_name,
                "strategy": self.entity_resolution_strategy,
            }
            if self.entity_resolution_strategy == "hybrid":
                kwargs.update(
                    {
                        "embedder": self.embedder,
                        "llm": self.llm,
                        "aliases": self.entity_resolution_aliases,
                        "llm_guidance": self.entity_resolution_llm_guidance,
                        "allow_ai_auto_merge": self.allow_ai_auto_merge,
                        "context_properties": (
                            self.entity_resolution_context_properties
                        ),
                        "conflict_properties": (
                            self.entity_resolution_conflict_properties
                        ),
                        "context_mode": self.entity_resolution_context_mode,
                    }
                )
            result = await self.graph_store.resolve_entities(
                **kwargs,
            )
            if isinstance(result, dict) and result.get("skipped"):
                print(f"Entity resolution skipped: reason={result.get('reason')}")
        except Exception as e:
            print(f"Entity resolution failed: {e}")
            if self.fail_on_resolution_error:
                raise

    async def _embed_entities(self):
        """Step 3: Generate vector embeddings for entity nodes."""
        embedder = EntityEmbedder(self.graph_store, self.embedder)
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
