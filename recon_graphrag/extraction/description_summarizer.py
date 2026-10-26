"""Mandatory entity and relationship description summarization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from tqdm import tqdm

from recon_graphrag.extraction.parser import DescriptionSummaryParser
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.entity_resolution_context import (
    INTERNAL_CONTEXT_FIELDS,
    STANDARD_CONTEXT_FIELDS,
)
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.observability import token_stage

# Node properties that are pipeline bookkeeping rather than domain facts;
# everything else is user-schema data worth showing to the summary LLM.
_NON_CONTEXT_FIELDS = (
    INTERNAL_CONTEXT_FIELDS
    | set(STANDARD_CONTEXT_FIELDS)
    | {
        "descriptions",
        "description_summary_status",
        "description_input_fingerprint",
        "description_summary_updated",
        "description_summary_error",
        "observation_count",
        "type",
        "weight",
    }
)


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
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.graph_name = graph_name
        self.concurrency = max(int(concurrency), 1)
        self.parser = DescriptionSummaryParser()
        self.prompts = prompt_builder or SchemaPromptBuilder()

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
                prompt = prompt_builder(item)
                stage = (
                    "construction.describe_entities"
                    if label == "entity"
                    else "construction.describe_relationships"
                )
                for attempt in range(3):
                    async with semaphore:
                        with token_stage(stage):
                            response = await self.llm.ainvoke(prompt)
                    try:
                        summary = self._parse_summary(response.content)
                        break
                    except ValueError:
                        if attempt == 2:
                            error_message = "Invalid description summary after 3 attempts (empty output)."
                            raise
                        await asyncio.sleep(attempt + 1)
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
        payload = {
            "id": entity.get("entity_id") or entity.get("id"),
            "name": entity.get("name"),
            "descriptions": sorted(self._clean_descriptions(entity)),
        }
        properties = self._context_properties(entity)
        if properties:
            payload["properties"] = {
                key: str(value) for key, value in properties.items()
            }
        return self._hash_payload(payload)

    def _relationship_fingerprint(self, rel: dict) -> str:
        payload = {
            "id": rel.get("rel_id") or rel.get("id"),
            "source": rel.get("source_id"),
            "target": rel.get("target_id"),
            "type": rel.get("type"),
            "descriptions": sorted(self._clean_descriptions(rel)),
        }
        return self._hash_payload(payload)

    def _context_properties(self, item: dict) -> dict:
        props = item.get("props") or {}
        return {
            key: value
            for key, value in props.items()
            if key not in _NON_CONTEXT_FIELDS and value not in (None, "", [], {})
        }

    def _clean_descriptions(self, item: dict) -> list[str]:
        return [
            str(description).strip()
            for description in (item.get("descriptions") or [])
            if str(description).strip()
        ]

    def _hash_payload(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
