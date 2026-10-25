"""LLM-based graph extraction from text."""

from __future__ import annotations

import asyncio
from typing import Any

from recon_graphrag.extraction.parser import AssessmentParser, ClaimParser, GraphExtractionParser
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import ExtractedClaim, GraphExtraction
from recon_graphrag.observability import token_stage


class LLMGraphExtractor:
    def __init__(self, llm, prompt_builder: SchemaPromptBuilder | None = None):
        self.llm = llm
        self.prompt_builder = prompt_builder or SchemaPromptBuilder()
        self.parser = GraphExtractionParser()
        self.assessment_parser = AssessmentParser()
        self.claim_parser = ClaimParser()

    async def extract(
        self,
        text: str,
        schema: GraphSchema,
        max_gleanings: int = 0,
    ) -> GraphExtraction:
        """Extract entities and relationships, with optional gleaning.

        Args:
            text: Source text to extract from.
            schema: Graph schema defining allowed types.
            max_gleanings: Maximum gleaning iterations after initial extraction.
                0 = single extraction call (current behavior).

        Returns:
            Merged GraphExtraction from initial + all gleaning iterations.
        """
        current = await self._single_extract(text, schema)

        for _iteration in range(max_gleanings):
            # Assessment: did we miss anything?
            missed = await self._assess(text, schema, current)
            if not missed:
                break

            # Continuation: extract missed items
            continuation = await self._continue(text, schema, current)
            if not continuation.nodes and not continuation.relationships:
                break

            # Merge: add only genuinely new items
            existing_ids = {n.id for n in current.nodes}
            existing_rels = {
                (r.source_id, r.type, r.target_id)
                for r in current.relationships
            }

            new_nodes = [n for n in continuation.nodes if n.id not in existing_ids]
            new_rels = [
                r for r in continuation.relationships
                if (r.source_id, r.type, r.target_id) not in existing_rels
            ]

            if not new_nodes and not new_rels:
                break  # No new observations

            current.nodes.extend(new_nodes)
            current.relationships.extend(new_rels)

        return current

    async def extract_claims(
        self,
        text: str,
        entity_ids: list[str],
        text_unit_id: str | None = None,
    ) -> list[ExtractedClaim]:
        """Extract claims/covariates about known entities.

        This is a separate LLM call that runs after entity extraction.
        Claims reference entity IDs already extracted from the same text.

        Args:
            text: Source text to extract claims from.
            entity_ids: IDs of entities extracted from this text.

        Returns:
            List of validated ExtractedClaim instances.
        """
        if not entity_ids:
            return []

        prompt = self.prompt_builder.build_claim_prompt(
            text=text,
            entity_ids=entity_ids,
        )
        with token_stage("construction.claims"):
            response = await self.llm.ainvoke(prompt)
        return self.claim_parser.parse(
            response.content,
            valid_entity_ids=set(entity_ids),
            source_text=text[:300],
            text_unit_id=text_unit_id,
        )

    async def _single_extract(self, text: str, schema: GraphSchema) -> GraphExtraction:
        """Run a single extraction call."""
        prompt = self.prompt_builder.build_prompt(text=text, schema=schema)
        with token_stage("construction.extract"):
            response = await self.llm.ainvoke(prompt)
        return self.parser.parse(response.content)

    async def _assess(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> bool:
        """Ask the LLM if it missed any entities."""
        prompt = self.prompt_builder.build_assessment_prompt(
            text=text, schema=schema, current=current
        )
        with token_stage("construction.gleaning_assess"):
            response = await self.llm.ainvoke(prompt)
        return self.assessment_parser.parse(response.content)

    async def _continue(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> GraphExtraction:
        """Ask the LLM to extract only missed items."""
        prompt = self.prompt_builder.build_continuation_prompt(
            text=text, schema=schema, current=current
        )
        with token_stage("construction.gleaning_continue"):
            response = await self.llm.ainvoke(prompt)
        return self.parser.parse(response.content)

    async def extract_all(
        self,
        chunks: list[Any],
        schema: GraphSchema,
        *,
        max_gleanings: int = 0,
        extract_claims: bool = False,
        concurrency: int = 5,
    ) -> list[tuple[str, GraphExtraction, list[ExtractedClaim]]]:
        """Extract entities/relationships from many chunks concurrently.

        Args:
            chunks: Chunk-like objects with ``id`` and ``text`` attributes, or
                dictionaries with ``"id"`` and ``"text"`` keys.
            schema: Graph schema defining allowed types.
            max_gleanings: Maximum gleaning iterations per chunk.
            extract_claims: If True, also extract claims for each chunk after
                entity extraction.
            concurrency: Maximum number of chunks to process concurrently.

        Returns:
            One tuple ``(chunk_id, extraction, claims)`` per input chunk, in the
            same order as ``chunks``.
        """
        semaphore = asyncio.Semaphore(max(int(concurrency), 1))
        stopped = False

        def _chunk_id_and_text(chunk: Any) -> tuple[str, str]:
            if isinstance(chunk, dict):
                return str(chunk["id"]), str(chunk["text"])
            chunk_id = getattr(chunk, "id", chunk)
            text = getattr(chunk, "text", chunk)
            return str(chunk_id), str(text)

        async def _extract_one(
            chunk: Any,
        ) -> tuple[str, GraphExtraction, list[ExtractedClaim]] | None:
            nonlocal stopped
            async with semaphore:
                if stopped:
                    return None
                try:
                    chunk_id, text = _chunk_id_and_text(chunk)
                    extraction = await self.extract(
                        text=text,
                        schema=schema,
                        max_gleanings=max_gleanings,
                    )
                    claims: list[ExtractedClaim] = []
                    if extract_claims and extraction.nodes:
                        entity_ids = [n.id for n in extraction.nodes]
                        claims = await self.extract_claims(
                            text=text,
                            entity_ids=entity_ids,
                            text_unit_id=chunk_id,
                        )
                    return chunk_id, extraction, claims
                except BaseException:
                    stopped = True
                    raise

        tasks = [asyncio.create_task(_extract_one(chunk)) for chunk in chunks]
        try:
            results = await asyncio.gather(*tasks)
            completed: list[tuple[str, GraphExtraction, list[ExtractedClaim]]] = []
            for result in results:
                # A stopped gate always accompanies a task that raises.
                if result is None:
                    raise RuntimeError("Extraction aborted before all chunks completed")
                completed.append(result)
            return completed
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
