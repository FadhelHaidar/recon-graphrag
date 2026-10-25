"""Knowledge graph construction pipeline.

Ingests text via internal extraction backend, then automatically
runs entity resolution and embedding (steps 1-3 of the full pipeline).

Steps 4-5 (community detection and summarization) are handled separately by the
CommunityPipeline, typically on a schedule.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings import ChunkEmbedder, EntityEmbedder
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import (
    PageWindowBuilder,
    TextChunker,
    _page_text,
)
from recon_graphrag.extraction.description_summarizer import DescriptionSummarizer
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.parser import _PropertyValidationError
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.structured import (
    RowMapping,
    iter_csv,
    iter_excel,
    iter_sql,
    row_text,
    rows_to_chunks_and_extractions,
)
from recon_graphrag.extraction.types import (
    ExtractedClaim,
    ExtractedRelationship,
    GraphExtraction,
)
from recon_graphrag.extraction.validator import SchemaValidator
from recon_graphrag.graphdb.base import GraphStore, GraphWriter
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.observability import (
    render_usage_table,
    run_scope,
    track_usage,
    usage_delta,
    usage_snapshot,
)
from recon_graphrag.utils.tokens import TiktokenTokenCounter, TokenCounter

logger = logging.getLogger(__name__)
# Default tiktoken encoding for token-based chunking (not a secret).
_DEFAULT_CHUNK_ENCODING = "cl100k_base"


class GraphBuilderPipeline:
    """Build a knowledge graph from text (LLM extraction) or structured rows.

    The constructor holds infrastructure (store, llm, embedder) and finalize
    configuration (entity resolution, summarization, embedding), which apply
    to every build method. Per-ingest knobs (schema, extraction concurrency,
    gleanings, chunking, mappings) are parameters of the build methods.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        graph_name: str = "entity-graph",
        graph_writer: GraphWriter | None = None,
        perform_entity_resolution: bool = True,
        entity_resolution_strategy: str = "normalized",
        entity_resolution_aliases: dict | None = None,
        entity_resolution_llm_guidance: str | None = None,
        entity_resolution_context_properties: dict[str, list[str]]
        | list[str]
        | None = None,
        entity_resolution_conflict_properties: dict[str, list[str]]
        | list[str]
        | None = None,
        entity_resolution_context_mode: str = "safe_defaults",
        allow_ai_auto_merge: bool = False,
        embed_entities: bool = True,
        embed_chunks: bool = False,
        summarize_descriptions: bool = True,
        summarization_concurrency: int = 5,
        summarization_limit: int = 500,
        fail_on_resolution_error: bool = False,
        fail_on_embedding_error: bool = False,
        extraction_prompt: str | None = None,
        assessment_prompt: str | None = None,
        continuation_prompt: str | None = None,
        claim_prompt: str | None = None,
        entity_summary_prompt: str | None = None,
        relationship_summary_prompt: str | None = None,
        track_token_usage: bool = True,
    ):
        self.graph_store = graph_store
        if track_token_usage:
            llm = track_usage(llm)
        self.llm = llm
        self.embedder = embedder
        self.graph_name = graph_name
        self.perform_entity_resolution = perform_entity_resolution
        self.entity_resolution_strategy = entity_resolution_strategy
        self.entity_resolution_aliases = entity_resolution_aliases
        self.entity_resolution_llm_guidance = entity_resolution_llm_guidance
        self.entity_resolution_context_properties = entity_resolution_context_properties
        self.entity_resolution_conflict_properties = (
            entity_resolution_conflict_properties
        )
        self.entity_resolution_context_mode = entity_resolution_context_mode
        self.allow_ai_auto_merge = allow_ai_auto_merge
        self.embed_entity_nodes = embed_entities
        # Off by default: chunk vectors cost one embedding call per chunk and
        # only enable relevance-ranked source selection, which degrades to the
        # previous chunk-id order when vectors are absent.
        self.embed_chunk_nodes = embed_chunks
        self.summarize_descriptions = summarize_descriptions
        self.summarization_concurrency = summarization_concurrency
        self.summarization_limit = summarization_limit
        self.fail_on_resolution_error = fail_on_resolution_error
        self.fail_on_embedding_error = fail_on_embedding_error

        # All-None strings produce the default prompts unchanged.
        self.prompt_builder = SchemaPromptBuilder(
            extraction_prompt=extraction_prompt,
            assessment_prompt=assessment_prompt,
            continuation_prompt=continuation_prompt,
            claim_prompt=claim_prompt,
            entity_summary_prompt=entity_summary_prompt,
            relationship_summary_prompt=relationship_summary_prompt,
        )

        self.extractor = LLMGraphExtractor(llm, prompt_builder=self.prompt_builder)
        self.validator = SchemaValidator()
        self.assembler = GraphDocumentAssembler()
        self.graph_writer = graph_writer or graph_store

    async def build_from_text(
        self,
        text: str,
        metadata: dict | None = None,
        *,
        schema: GraphSchema,
        extraction_concurrency: int = 5,
        max_gleanings: int = 1,
        extract_claims: bool = False,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        chunk_unit: str = "tokens",
        token_counter: TokenCounter | None = None,
        token_encoding: str = _DEFAULT_CHUNK_ENCODING,
    ) -> dict:
        """Build knowledge graph from raw text.

        ``schema`` is required; to auto-generate one from sample text, call
        ``analyze_schema``/``aanalyze_schema`` explicitly and pass the result.

        Uses internal token-based chunking and extraction, then automatically:
          - Step 2: Entity resolution (merge duplicates)
          - Step 3: Entity embedding (for local/DRIFT search)

        Steps 4-5 must be run separately via CommunityPipeline.
        """
        self._require_schema(schema)
        self._require_concurrency(extraction_concurrency)
        with run_scope():
            result = await self._ingest_text(
                text=text,
                metadata=metadata or {},
                schema=schema,
                extraction_concurrency=extraction_concurrency,
                max_gleanings=max_gleanings,
                extract_claims=extract_claims,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_unit=chunk_unit,
                token_counter=token_counter,
                token_encoding=token_encoding,
                finalize=True,
            )
            token_usage = usage_snapshot(self.llm)
            if token_usage is not None:
                result["token_usage"] = token_usage
            return result

    async def _ingest_text(
        self,
        text: str,
        metadata: dict,
        *,
        schema: GraphSchema,
        extraction_concurrency: int,
        max_gleanings: int,
        extract_claims: bool,
        chunk_size: int,
        chunk_overlap: int,
        chunk_unit: str,
        token_counter: TokenCounter | None,
        token_encoding: str,
        finalize: bool,
        batch_abort: asyncio.Event | None = None,
    ) -> dict:
        document_id = self._make_document_id(text=text, metadata=metadata)
        text_hash = self._hash_text(text)
        logger.info(
            "graph build start: document_id=%s chars=%s", document_id, len(text)
        )

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
            schema=schema,
            extraction_concurrency=extraction_concurrency,
            max_gleanings=max_gleanings,
            extract_claims=extract_claims,
            finalize=finalize,
            batch_abort=batch_abort,
        )

        logger.info("graph build complete: document_id=%s", document_id)
        return result

    async def build_from_documents(
        self,
        documents: list[dict],
        *,
        schema: GraphSchema,
        extraction_concurrency: int = 5,
        document_concurrency: int = 1,
        max_gleanings: int = 1,
        extract_claims: bool = False,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        chunk_unit: str = "tokens",
        token_counter: TokenCounter | None = None,
        token_encoding: str = _DEFAULT_CHUNK_ENCODING,
        window_size: int = 2,
        window_overlap: int = 1,
    ) -> list[dict]:
        """Build knowledge graph from multiple document envelopes.

        ``schema`` is required; to auto-generate one from sample text, call
        ``analyze_schema``/``aanalyze_schema`` explicitly and pass the result.

        Each envelope must contain exactly one of:
          - ``text``: a raw text string.
          - ``pages``: a list of page strings or page dicts with ``text``.

        Envelopes may optionally include ``metadata``.

        Returns one result dict per input envelope.
        """
        self._require_schema(schema)
        self._require_concurrency(extraction_concurrency, document_concurrency)
        self._validate_document_envelopes(documents)

        with run_scope():
            batch_abort = asyncio.Event()

            # Extract + write each document concurrently. Whole-graph finalization
            # (resolution, summarization, embedding) runs once afterward instead of
            # once per document, which would rescan the whole graph N times and race
            # when document_concurrency > 1.
            #
            # ponytail: before/after usage_delta attributes tokens by wall-clock
            # window, not call-site, so with document_concurrency > 1 concurrent
            # documents can bleed into each other's counts. Exact when
            # document_concurrency=1 (the default).
            async def _build_one(envelope: dict) -> dict | None:
                if batch_abort.is_set():
                    return None
                try:
                    return await _ingest_one(envelope)
                except BaseException:
                    # Nested extraction also closes this gate before draining.
                    batch_abort.set()
                    raise

            async def _ingest_one(envelope: dict) -> dict:
                metadata = envelope.get("metadata") or {}
                before = usage_snapshot(self.llm)
                if "text" in envelope:
                    result = await self._ingest_text(
                        text=envelope["text"],
                        metadata=metadata,
                        schema=schema,
                        extraction_concurrency=extraction_concurrency,
                        max_gleanings=max_gleanings,
                        extract_claims=extract_claims,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_unit=chunk_unit,
                        token_counter=token_counter,
                        token_encoding=token_encoding,
                        finalize=False,
                        batch_abort=batch_abort,
                    )
                else:
                    result = await self._build_from_pages_envelope(
                        pages=envelope["pages"],
                        metadata=metadata,
                        schema=schema,
                        extraction_concurrency=extraction_concurrency,
                        max_gleanings=max_gleanings,
                        extract_claims=extract_claims,
                        window_size=window_size,
                        window_overlap=window_overlap,
                        finalize=False,
                        batch_abort=batch_abort,
                    )
                after = usage_snapshot(self.llm)
                if after is not None:
                    result["token_usage"] = usage_delta(before, after)
                return result

            # Fixed workers over an iterator: only ``document_concurrency`` tasks
            # exist no matter how many documents were passed.
            pending = iter(list(enumerate(documents)))
            results: list[dict | None] = [None] * len(documents)

            async def _document_worker() -> None:
                for position, envelope in pending:
                    results[position] = await _build_one(envelope)

            workers = [
                asyncio.create_task(_document_worker())
                for _ in range(max(int(document_concurrency), 1))
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

            completed: list[dict] = []
            for result in results:
                if result is None:
                    raise RuntimeError(
                        "Document batch aborted before all documents completed"
                    )
                completed.append(result)
            validation = await self._finalize_graph()
            for result in completed:
                result["validation"] = validation
            return completed

    async def _build_from_pages_envelope(
        self,
        pages: list[str | dict],
        metadata: dict,
        *,
        schema: GraphSchema,
        extraction_concurrency: int,
        max_gleanings: int,
        extract_claims: bool,
        window_size: int,
        window_overlap: int,
        finalize: bool = True,
        batch_abort: asyncio.Event | None = None,
    ) -> dict:
        text = "\n\n".join(_page_text(page) for page in pages)
        document_id = self._make_document_id(text=text, metadata=metadata)
        text_hash = self._hash_text(text)

        logger.info(
            "graph build start from pages: document_id=%s pages=%s chars=%s "
            "window_size=%s window_overlap=%s",
            document_id,
            len(pages),
            len(text),
            window_size,
            window_overlap,
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
            logger.info(
                "built page windows: document_id=%s chunks=%s first_page=%s last_page=%s",
                document_id,
                len(chunks),
                chunks[0].metadata.get("page_start"),
                chunks[-1].metadata.get("page_end"),
            )
        else:
            logger.info("built page windows: document_id=%s chunks=0", document_id)

        result = await self._build_from_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
            schema=schema,
            extraction_concurrency=extraction_concurrency,
            max_gleanings=max_gleanings,
            extract_claims=extract_claims,
            finalize=finalize,
            batch_abort=batch_abort,
        )

        logger.info("graph build complete: document_id=%s", document_id)
        return result

    async def build_from_rows(
        self,
        rows: Iterable[dict],
        mapping: RowMapping,
        metadata: dict | None = None,
        *,
        extraction_schema: GraphSchema | None = None,
        extraction_concurrency: int = 5,
        max_gleanings: int = 1,
        finalize: bool = True,
    ) -> dict:
        """Build knowledge graph from structured rows (dicts) via a RowMapping.

        Columns map deterministically to entities/relationships (no LLM).
        Rows with a non-empty ``TextColumn`` additionally get an LLM extraction
        pass using ``extraction_schema`` (defaults to ``mapping.to_schema()``);
        extracted entities are linked to the row's anchor entity and merge with
        same-label/same-name direct entities during finalize entity resolution.
        """
        self._require_concurrency(extraction_concurrency)
        mapping.validate()
        rows = [dict(row) for row in rows]
        metadata = metadata or {}
        llm_schema = extraction_schema or mapping.to_schema()

        with run_scope():
            text = "\n\n".join(row_text(row) for row in rows)
            document_id = self._make_document_id(text=text, metadata=metadata)
            text_hash = self._hash_text(text)
            logger.info(
                "graph build start from rows: document_id=%s rows=%s",
                document_id,
                len(rows),
            )

            chunks, extractions, llm_rows = rows_to_chunks_and_extractions(
                rows=rows, mapping=mapping, document_id=document_id
            )

            if llm_rows:
                await self._extract_text_columns(
                    document_id=document_id,
                    chunks=chunks,
                    extractions=extractions,
                    llm_rows=llm_rows,
                    schema=llm_schema,
                    extraction_concurrency=extraction_concurrency,
                    max_gleanings=max_gleanings,
                )

            graph_document = self.assembler.assemble(
                document_id=document_id,
                text_hash=text_hash,
                chunks=chunks,
                chunk_extractions=extractions,
                metadata=metadata,
                graph_name=self.graph_name,
                identity_properties=self._identity_properties(llm_schema),
            )
            write_stats = await asyncio.to_thread(
                self.graph_writer.write_graph_document, graph_document
            )
            logger.info(
                "write complete to %s: %s entities, %s relationships",
                self._graph_store_name(),
                write_stats.get("entities"),
                write_stats.get("relationships"),
            )

            result = {
                "extraction": {
                    "document_id": document_id,
                    "chunks": len(chunks),
                    "llm_rows": len(llm_rows),
                    "write_stats": write_stats,
                }
            }
            if finalize:
                result["validation"] = await self._finalize_graph()
            token_usage = usage_snapshot(self.llm)
            if token_usage is not None:
                result["token_usage"] = token_usage
            logger.info("graph build complete: document_id=%s", document_id)
            return result

    async def build_from_csv(
        self,
        path: str | Path,
        mapping: RowMapping,
        metadata: dict | None = None,
        *,
        encoding: str = "utf-8-sig",
        **row_kwargs: Any,
    ) -> dict:
        """Build knowledge graph from a CSV file via a RowMapping."""
        metadata = {"source": Path(path).name, **(metadata or {})}
        return await self.build_from_rows(
            iter_csv(path, encoding=encoding), mapping, metadata, **row_kwargs
        )

    async def build_from_excel(
        self,
        path: str | Path,
        mapping: RowMapping,
        metadata: dict | None = None,
        *,
        sheet: str | None = None,
        **row_kwargs: Any,
    ) -> dict:
        """Build knowledge graph from an Excel worksheet (requires openpyxl)."""
        metadata = {"source": Path(path).name, **(metadata or {})}
        return await self.build_from_rows(
            iter_excel(path, sheet=sheet), mapping, metadata, **row_kwargs
        )

    async def build_from_sql(
        self,
        connection: Any,
        query: str,
        mapping: RowMapping,
        metadata: dict | None = None,
        *,
        params: Any = None,
        **row_kwargs: Any,
    ) -> dict:
        """Build knowledge graph from a DB-API 2.0 query result via a RowMapping."""
        return await self.build_from_rows(
            iter_sql(connection, query, params=params), mapping, metadata, **row_kwargs
        )

    async def _extract_text_columns(
        self,
        chunks: list,
        extractions: dict,
        llm_rows: dict,
        *,
        document_id: str,
        schema: GraphSchema,
        extraction_concurrency: int,
        max_gleanings: int,
    ) -> None:
        """LLM-extract rows with text columns, merging into their extractions.

        Extraction runs on the full row chunk text so FROM_CHUNK evidence
        points at exactly what the LLM saw.
        """
        pending_rows = [chunk for chunk in chunks if chunk.id in llm_rows]
        stopped = False
        # Fixed workers over an iterator: pending tasks are bounded by
        # extraction_concurrency instead of one task per row chunk.
        row_source = iter(pending_rows)

        async def _extract_worker(bar) -> None:
            nonlocal stopped
            for chunk in row_source:
                if stopped:
                    return
                try:
                    validated, _ = await self._extract_chunk(
                        chunk, document_id, schema, max_gleanings, False
                    )
                except BaseException:
                    stopped = True
                    raise
                target = extractions[chunk.id]
                target.nodes.extend(validated.nodes)
                target.relationships.extend(validated.relationships)
                for anchor_key, anchor_identity, rel_type in llm_rows[chunk.id]:
                    if not anchor_key:
                        continue
                    for node in validated.nodes:
                        # Skip self-links: resolution later merges the LLM
                        # node with the direct anchor when names match.
                        if node.id.strip().lower() == anchor_identity.lower():
                            continue
                        target.relationships.append(
                            ExtractedRelationship(
                                source_id=anchor_key,
                                target_id=node.id,
                                type=rel_type,
                            )
                        )
                bar.update(1)

        with tqdm_asyncio(
            total=len(pending_rows), desc="Extracting rows", disable=None
        ) as bar:
            workers = [
                asyncio.create_task(_extract_worker(bar))
                for _ in range(max(int(extraction_concurrency), 1))
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

    async def _extract_chunk(
        self,
        chunk,
        document_id: str,
        schema: GraphSchema,
        max_gleanings: int,
        extract_claims: bool,
    ) -> tuple[GraphExtraction, list[ExtractedClaim]]:
        # Public IDs may contain source/title metadata. Use an opaque document
        # fingerprint and chunk ordinal in errors, never text or provider errors.
        context = (
            f"document={self._hash_text(document_id)[:16]} chunk={chunk.index + 1}"
        )
        try:
            raw = await self.extractor.extract(
                text=chunk.text, schema=schema, max_gleanings=max_gleanings
            )
        except _PropertyValidationError as error:
            # Only parser-created metadata is trusted across extraction; provider
            # ValueError strings must never supply field context.
            raise ValueError(
                f"Validation failed ({context} field={error.path}: {error.reason})"
            ) from None
        except Exception:
            raise RuntimeError(f"Extraction failed ({context})") from None

        try:
            validated = self.validator.validate(raw, schema)
        except ValueError as error:
            # Only inspect validator errors, not arbitrary provider messages.
            # C02 emits schema field paths and fixed reasons, never values.
            path = re.fullmatch(
                r"((?:nodes|relationships)\[\d+\]\.properties(?:\.[^\r\n]+)?): "
                r"(expected (?:an object|a finite number|STRING|INTEGER|FLOAT|"
                r"BOOLEAN|DATE|DATETIME|LIST)|required property is missing or null)",
                str(error),
            )
            field = f" field={path[1]}: {path[2]}" if path else ""
            raise ValueError(f"Validation failed ({context}{field})") from None
        except Exception:
            raise RuntimeError(f"Validation failed ({context})") from None

        claims = []
        if extract_claims and validated.nodes:
            try:
                claims = await self.extractor.extract_claims(
                    text=chunk.text,
                    entity_ids=[node.id for node in validated.nodes],
                    text_unit_id=chunk.id,
                )
            except Exception:
                raise RuntimeError(f"Claims extraction failed ({context})") from None
        return validated, claims

    @staticmethod
    def _require_schema(schema: GraphSchema) -> None:
        if not isinstance(schema, GraphSchema):
            raise ValueError(
                "schema must be a GraphSchema; to auto-generate one from sample "
                "text, call analyze_schema/aanalyze_schema and pass the result"
            )

    @staticmethod
    def _require_concurrency(
        extraction_concurrency: int, document_concurrency: int | None = None
    ) -> None:
        if extraction_concurrency < 1:
            raise ValueError("extraction_concurrency must be at least 1")
        if document_concurrency is not None and document_concurrency < 1:
            raise ValueError("document_concurrency must be at least 1")

    @staticmethod
    def _identity_properties(schema: GraphSchema) -> dict[str, str] | None:
        """Map node label to a non-name identity property for entity keys."""
        identity_properties = {
            node_type.label: node_type.identity_property
            for node_type in schema.node_types
            if node_type.identity_property and node_type.identity_property != "name"
        }
        return identity_properties or None

    def _validate_document_envelopes(self, documents: list[dict]) -> None:
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")

        for envelope in documents:
            if not isinstance(envelope, dict):
                raise ValueError("each document envelope must be a dict")

            has_text = "text" in envelope
            has_pages = "pages" in envelope

            if has_text and has_pages:
                raise ValueError(
                    "document envelope must not contain both 'text' and 'pages'"
                )
            if not has_text and not has_pages:
                raise ValueError(
                    "document envelope must contain either 'text' or 'pages'"
                )

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
        *,
        schema: GraphSchema,
        extraction_concurrency: int,
        max_gleanings: int,
        extract_claims: bool,
        finalize: bool = True,
        batch_abort: asyncio.Event | None = None,
    ) -> dict:
        extraction = await self._extract_and_write_chunks(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            metadata=metadata,
            schema=schema,
            extraction_concurrency=extraction_concurrency,
            max_gleanings=max_gleanings,
            extract_claims=extract_claims,
            batch_abort=batch_abort,
        )

        result = {"extraction": extraction}
        if finalize:
            result["validation"] = await self._finalize_graph()
        return result

    async def _finalize_graph(self) -> dict:
        """Run whole-graph post-processing after all documents are written.

        Operates over the entire graph (scoped by graph_name), so it must run
        once per build batch, not once per document.
        """
        logger.info("backfilling missing entity descriptions")
        await self._backfill_descriptions()

        resolution = None
        if self.perform_entity_resolution:
            logger.info("resolving duplicate entities")
            resolution = await self._resolve_entities()

        if self.summarize_descriptions:
            summarizer = DescriptionSummarizer(
                self.llm,
                self.graph_store,
                self.graph_name,
                concurrency=self.summarization_concurrency,
                prompt_builder=self.prompt_builder,
            )
            await summarizer.summarize_entities(limit=self.summarization_limit)
            await summarizer.summarize_relationships(limit=self.summarization_limit)

        if self.embed_entity_nodes:
            await self._embed_entities()

        if self.embed_chunk_nodes:
            await self._embed_chunks()

        logger.info("validating graph build")
        validation = await self._validate_graph_build()
        token_usage = usage_snapshot(self.llm)
        self._write_finalize_summary(resolution, validation, token_usage)
        return validation

    def _write_finalize_summary(
        self, resolution: dict | None, validation: dict, token_usage: dict | None = None
    ) -> None:
        """Print the post-resolution graph summary (always, via tqdm.write)."""
        lines = [
            "graph build summary:",
            f"  final graph: {validation.get('entity_count')} entities, "
            f"{validation.get('entity_relationship_count')} relationships",
        ]
        if resolution and resolution.get("skipped"):
            lines.append(f"  entity resolution: skipped ({resolution.get('reason')})")
        elif resolution:
            merged_groups = resolution.get("merged_groups", 0)
            llm_merged = len(resolution.get("ai_merged_review_groups", []))
            deterministic = merged_groups - llm_merged
            lines.append(
                f"  entity resolution ({resolution.get('strategy')}): "
                f"{resolution.get('merged_nodes', 0)} nodes merged "
                f"across {merged_groups} groups "
                f"(deterministic={deterministic}, llm={llm_merged}); "
                f"{len(resolution.get('review_groups', []))} flagged for review"
            )
        if token_usage:
            lines.append("")
            lines.append(render_usage_table(token_usage))
        tqdm_asyncio.write("\n".join(lines))

    async def _extract_and_write_chunks(
        self,
        document_id: str,
        text_hash: str,
        chunks: list,
        metadata: dict,
        *,
        schema: GraphSchema,
        extraction_concurrency: int,
        max_gleanings: int,
        extract_claims: bool,
        batch_abort: asyncio.Event | None = None,
    ) -> dict:
        chunk_extractions = {}
        chunk_claims = {}
        total = len(chunks)

        logger.info(
            "extraction start: document_id=%s chunks=%s concurrency=%s",
            document_id,
            total,
            extraction_concurrency,
        )

        stopped = False
        node_total = rel_total = 0
        # Fixed workers over an iterator: pending tasks are bounded by
        # extraction_concurrency instead of one task per chunk.
        pending = iter(chunks)

        async def _extract_worker(bar) -> None:
            nonlocal stopped, node_total, rel_total
            for chunk in pending:
                if stopped:
                    return
                try:
                    validated, claims = await self._extract_chunk(
                        chunk, document_id, schema, max_gleanings, extract_claims
                    )
                except BaseException:
                    stopped = True
                    # Close the document gate immediately, not after sibling
                    # cancellation cleanup (which may await arbitrary work).
                    if batch_abort is not None:
                        batch_abort.set()
                    raise
                chunk_extractions[chunk.id] = validated
                node_total += len(validated.nodes)
                rel_total += len(validated.relationships)
                if claims:
                    chunk_claims[chunk.id] = claims
                bar.set_postfix(entities=node_total, relationships=rel_total)
                bar.update(1)

        with tqdm_asyncio(total=total, desc="Extracting", disable=None) as bar:
            workers = [
                asyncio.create_task(_extract_worker(bar))
                for _ in range(max(int(extraction_concurrency), 1))
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        tqdm_asyncio.write(
            f"extraction complete: {len(chunk_extractions)}/{total} succeeded, "
            f"0/{total} failed "
            f"({node_total} entities, {rel_total} relationships extracted)"
        )

        total_claims = sum(len(c) for c in chunk_claims.values())
        if total_claims:
            logger.info("total claims extracted: %s", total_claims)

        graph_document = self.assembler.assemble(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            chunk_extractions=chunk_extractions,
            metadata=metadata,
            graph_name=self.graph_name,
            identity_properties=self._identity_properties(schema),
            chunk_claims=chunk_claims if chunk_claims else None,
        )

        write_stats = await asyncio.to_thread(
            self.graph_writer.write_graph_document, graph_document
        )
        logger.info(
            "write complete to %s: %s entities, %s relationships, %s claims",
            self._graph_store_name(),
            write_stats.get("entities"),
            write_stats.get("relationships"),
            write_stats.get("claims", 0),
        )

        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "write_stats": write_stats,
        }

    async def _backfill_descriptions(self):
        """Set description = '' on __Entity__ nodes missing the property."""
        await asyncio.to_thread(self.graph_store.backfill_descriptions)

    async def _resolve_entities(self):
        """Step 2: Merge duplicate entities with the internal resolver."""
        try:
            strategy = self.entity_resolution_strategy
            if strategy == "exact":
                result = await self.graph_store.resolve_entities_exact(
                    graph_name=self.graph_name,
                )
            elif strategy == "normalized":
                result = await self.graph_store.resolve_entities_normalized(
                    graph_name=self.graph_name,
                    conflict_properties=self.entity_resolution_conflict_properties,
                )
            elif strategy == "fuzzy":
                result = await self.graph_store.resolve_entities_fuzzy(
                    graph_name=self.graph_name,
                )
            elif strategy == "hybrid":
                result = await self.graph_store.resolve_entities_hybrid(
                    graph_name=self.graph_name,
                    embedder=self.embedder,
                    llm=self.llm,
                    aliases=self.entity_resolution_aliases,
                    llm_guidance=self.entity_resolution_llm_guidance,
                    allow_ai_auto_merge=self.allow_ai_auto_merge,
                    context_properties=self.entity_resolution_context_properties,
                    conflict_properties=self.entity_resolution_conflict_properties,
                    context_mode=self.entity_resolution_context_mode,
                )
            else:
                raise ValueError(f"Unknown entity resolution strategy: {strategy}")
            if isinstance(result, dict) and not result.get("skipped"):
                for rg in result.get("review_groups", []):
                    logger.debug(
                        "review group: names=%s scores=%s decision=%s",
                        rg.get("names"),
                        rg.get("scores"),
                        rg.get("decision"),
                    )
            return result if isinstance(result, dict) else None
        except Exception as e:
            logger.error("entity resolution failed: %s", e)
            if self.fail_on_resolution_error:
                raise
            return None

    async def _embed_entities(self, *, require_complete: bool = False):
        """Step 3: Generate vector embeddings for entity nodes.

        With ``require_complete`` (or ``fail_on_embedding_error``), verifies
        that no entity in this graph still lacks an embedding after the
        embedding pass and raises if any remain.
        """
        embedder = EntityEmbedder(
            self.graph_store, self.embedder, graph_name=self.graph_name
        )
        strict = require_complete or self.fail_on_embedding_error
        try:
            await embedder.embed_entities()
            if strict and (
                await asyncio.to_thread(
                    self.graph_store.get_unembedded_entities,
                    graph_name=self.graph_name,
                    limit=1,
                    model_fingerprint=embedder.model_fingerprint,
                )
            ):
                raise RuntimeError(
                    "Entity embedding incomplete: unembedded entities remain "
                    "in the configured graph."
                )
        except Exception as e:
            logger.error("entity embedding failed: %s", e)
            if strict:
                raise

    async def _embed_chunks(self):
        """Generate vector embeddings for source chunks.

        Optional stage: chunk vectors only enable relevance-ranked source
        selection in local/DRIFT retrieval. A failure is logged and never fails
        the build, because ranking falls back to chunk-id order without them.
        """
        embedder = ChunkEmbedder(
            self.graph_store, self.embedder, graph_name=self.graph_name
        )
        try:
            await embedder.embed_chunks()
        except Exception as e:
            logger.error("chunk embedding failed: %s", e)

    async def complete_embeddings(self) -> None:
        """Embed every entity in this graph that still lacks an embedding.

        Recovery after a failed embedding stage: uses the configured graph
        store, embedder, and graph name to fill only missing entity vectors,
        then verifies completion and raises if any remain. Existing vectors
        are not replaced; extraction, writes, resolution, and summarization
        are not rerun. A graph without missing vectors does not prove that
        all intended documents were ingested.
        """
        await self._embed_entities(require_complete=True)

    async def _validate_graph_build(self) -> dict:
        return await asyncio.to_thread(self.graph_store.validate_graph_build)

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
        explicit = metadata.get("document_id")
        if explicit:
            return str(explicit)
        source = metadata.get("source") or metadata.get("title")
        if source:
            # Slipugged slugs collided ("A/B" → "a-b" == "A-B"); hash instead.
            return f"doc:{self._hash_text(str(source))[:16]}"

        return f"doc:{self._hash_text(text)[:16]}"
