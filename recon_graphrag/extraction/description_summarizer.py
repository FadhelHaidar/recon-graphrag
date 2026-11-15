"""Mandatory entity and relationship description summarization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from tqdm import tqdm

from recon_graphrag.description_text import (
    entity_summary_fingerprint,
    summary_context_properties,
)
from recon_graphrag.extraction.parser import DescriptionSummaryParser
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.observability import token_stage
from recon_graphrag.utils.tokens import ApproximateTokenCounter, TokenCounter

logger = logging.getLogger(__name__)


class _InvalidSummaryOutput(ValueError):
    """Report exhausted output retries without exposing provider text."""


class DescriptionSummarizationError(RuntimeError):
    """Raised when mandatory description summarization fails."""


class DescriptionSummarizer:
    def __init__(
        self,
        llm: BaseLLM,
        graph_store: GraphStore,
        graph_name: str,
        concurrency: int = 5,
        prompt_builder: SchemaPromptBuilder | None = None,
        *,
        summary_token_budget: int = 8192,
        summary_output_tokens: int = 1024,
        token_counter: TokenCounter | None = None,
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.graph_name = graph_name
        self.concurrency = max(int(concurrency), 1)
        self.parser = DescriptionSummaryParser()
        self.prompts = prompt_builder or SchemaPromptBuilder()
        if summary_output_tokens <= 0 or summary_token_budget <= summary_output_tokens:
            raise ValueError("Summary token budget must exceed the positive output reserve")
        self.summary_token_budget = summary_token_budget
        self.summary_output_tokens = summary_output_tokens
        self.token_counter = token_counter or ApproximateTokenCounter()

    async def summarize_entities(self, limit: int = 500) -> dict:
        return await self._summarize_loop(
            fetch=self.graph_store.get_entities_needing_summary,
            persist=self.graph_store.persist_entity_summaries,
            prompt=self._build_entity_summary_prompt,
            fingerprint=self._entity_fingerprint,
            label="entity",
            limit=limit,
        )

    async def summarize_relationships(self, limit: int = 500) -> dict:
        return await self._summarize_loop(
            fetch=self.graph_store.get_relationships_needing_summary,
            persist=self.graph_store.persist_relationship_summaries,
            prompt=self._build_relationship_summary_prompt,
            fingerprint=self._relationship_fingerprint,
            label="relationship",
            limit=limit,
        )

    async def _summarize_loop(
        self,
        fetch,
        persist,
        prompt,
        fingerprint,
        label: str,
        limit: int,
    ) -> dict:
        total = 0
        batches = 0
        while True:
            candidates = await asyncio.to_thread(fetch, self.graph_name, limit=limit)
            if not candidates:
                return {"summarized": total, "batches": batches}

            fingerprints = {item["id"]: fingerprint(item) for item in candidates}
            summaries, failures = await self._summary_payload(
                candidates, fingerprints, prompt, label
            )
            try:
                await asyncio.to_thread(persist, self.graph_name, summaries)
            except Exception as exc:
                raise DescriptionSummarizationError(
                    f"Could not persist {label} summary batch ({len(summaries)} items); "
                    "check database availability before retrying."
                ) from exc

            if failures:
                first_failure = next(
                    row for row in summaries
                    if row["description_summary_status"] == "failed"
                )
                raise DescriptionSummarizationError(
                    f"Failed to summarize {label} descriptions: "
                    f"{len(failures)} failed, {len(summaries) - len(failures)} succeeded; "
                    f"first failed ID={str(first_failure['id'])[:100]!r}. "
                    f"{first_failure['description_summary_error']} "
                    "Batch outcomes persisted; fix the provider/output and rerun "
                    "to retry failed items."
                ) from failures[0]

            total += len(summaries)
            batches += 1

    async def _summary_payload(
        self,
        items: list[dict],
        fingerprints: dict[str, str],
        prompt_builder,
        label: str = "item",
    ) -> tuple[list[dict], list[Exception]]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _summarize(item: dict) -> tuple[dict, Exception | None]:
            summary = None
            failure = None
            error_message = None
            try:
                summary = await self._summarize_item(item, prompt_builder, semaphore, label)
            except _InvalidSummaryOutput as exc:
                failure = exc
                error_message = str(exc)
            except Exception as exc:
                failure = exc
                if error_message is None:
                    error_message = (
                        f"Description summarization failed ({type(exc).__name__[:100]}); "
                        "check provider/output and retry."
                    )

            progress.update()
            return {
                "id": item["id"],
                "description": summary,
                "descriptions": self._clean_descriptions(item),
                "description_summary_status": "failed" if failure else "success",
                "description_input_fingerprint": fingerprints[item["id"]],
                "description_summary_updated": int(time.time() * 1000),
                "description_summary_error": error_message,
            }, failure

        progress = tqdm(
            total=len(items), desc=f"Summarizing {label} descriptions", disable=None
        )
        tasks = [asyncio.create_task(_summarize(item)) for item in items]
        try:
            outcomes = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            progress.close()
        return (
            [row for row, _ in outcomes],
            [failure for _, failure in outcomes if failure is not None],
        )

    def _budgeted_prompt(self, item: dict, descriptions: list[str], prompt_builder) -> str:
        prompt = prompt_builder({**item, "descriptions": descriptions})
        return prompt + f"\n\nReturn at most {self.summary_output_tokens} tokens. Preserve supported facts and qualifications."

    def _partition_descriptions(self, item: dict, descriptions: list[str], prompt_builder) -> list[list[str]]:
        """Partition complete source text without dropping an oversized observation."""
        available = self.summary_token_budget - self.summary_output_tokens

        def fits(texts: list[str]) -> bool:
            return self.token_counter.count(self._budgeted_prompt(item, texts, prompt_builder)) <= available

        if not fits([]):
            raise ValueError("Fixed summary context exceeds the input budget")
        batches: list[list[str]] = []
        batch: list[str] = []
        for text in descriptions:
            remaining = text
            while remaining:
                if fits([*batch, remaining]):
                    batch.append(remaining)
                    break
                # Find a source-string boundary; token truncation can lose Unicode bytes.
                low, high = 0, len(remaining)
                while low < high:
                    middle = (low + high + 1) // 2
                    if fits([*batch, remaining[:middle]]):
                        low = middle
                    else:
                        high = middle - 1
                if low:
                    batch.append(remaining[:low])
                    remaining = remaining[low:]
                elif not batch:
                    raise ValueError("Summary budget cannot fit one source character")
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return batches

    async def _request_summary(self, prompt: str, semaphore, label: str) -> str:
        stage = "construction.describe_entities" if label == "entity" else "construction.describe_relationships"
        for attempt in range(3):
            async with semaphore:
                with token_stage(stage):
                    response = await self.llm.ainvoke(prompt)
            invalid_reason = "empty output"
            try:
                summary = self._parse_summary(response.content)
                reason = getattr(response, "finish_reason", None)
                if isinstance(reason, str) and reason not in {"stop", "end_turn"}:
                    invalid_reason = "incomplete output"
                    raise ValueError("Incomplete summary output")
                if self.token_counter.count(summary) > self.summary_output_tokens:
                    invalid_reason = "oversized output"
                    raise ValueError("Summary exceeds the output reserve")
                return summary
            except ValueError:
                if attempt == 2:
                    raise _InvalidSummaryOutput(
                        f"Invalid description summary after 3 attempts ({invalid_reason})."
                    ) from None
                await asyncio.sleep(attempt + 1)
        raise AssertionError("unreachable")

    async def _summarize_item(self, item: dict, prompt_builder, semaphore, label: str) -> str:
        batches = self._partition_descriptions(item, self._clean_descriptions(item), prompt_builder)
        requests = 0
        while batches:
            generated = []
            for batch in batches:
                prompt = self._budgeted_prompt(item, batch, prompt_builder)
                generated.append(await self._request_summary(prompt, semaphore, label))
                requests += 1
            if len(generated) == 1:
                logger.info("Completed %s summary with %d evidence/consolidation requests", label, requests)
                return generated[0]
            next_batches = self._partition_descriptions(item, generated, prompt_builder)
            if len(next_batches) >= len(batches):
                raise ValueError("Summary consolidation cannot make progress within the budget")
            batches = next_batches
        raise ValueError("Summary requires nonempty source evidence")

    def _build_entity_summary_prompt(self, entity: dict) -> str:
        return self.prompts.build_entity_summary_prompt(
            descriptions=self._clean_descriptions(entity),
            entity_name=str(entity.get("name") or entity.get("entity_id") or ""),
            entity_type=str(entity.get("type") or "Entity"),
            properties=self._context_properties(entity),
        )

    def _build_relationship_summary_prompt(self, rel: dict) -> str:
        return self.prompts.build_relationship_summary_prompt(
            descriptions=self._clean_descriptions(rel),
            source=str(rel.get("source_id") or ""),
            target=str(rel.get("target_id") or ""),
            rel_type=str(rel.get("type") or "RELATED_TO"),
        )

    def _parse_summary(self, content: str) -> str:
        return self.parser.parse(content)

    def _entity_fingerprint(self, entity: dict) -> str:
        return self._hash_payload({"inputs": entity_summary_fingerprint(entity), **self._batching_policy()})

    def _relationship_fingerprint(self, rel: dict) -> str:
        payload = {
            "id": rel.get("rel_id") or rel.get("id"),
            "source": rel.get("source_id"),
            "target": rel.get("target_id"),
            "type": rel.get("type"),
            "descriptions": sorted(self._clean_descriptions(rel)),
        }
        return self._hash_payload({**payload, **self._batching_policy()})

    def _batching_policy(self) -> dict:
        return {"batching_version": 1, "token_budget": self.summary_token_budget,
                "output_reserve": self.summary_output_tokens,
                "token_counter": type(self.token_counter).__qualname__,
                "approximate_ratio": getattr(self.token_counter, "_ratio", None),
                "encoding": getattr(getattr(self.token_counter, "_encoding", None), "name", None)}

    def _context_properties(self, item: dict) -> dict:
        return summary_context_properties(item)

    def _clean_descriptions(self, item: dict) -> list[str]:
        return [
            str(description).strip()
            for description in (item.get("descriptions") or [])
            if str(description).strip()
        ]

    def _hash_payload(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
