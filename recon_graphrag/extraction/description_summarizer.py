"""Mandatory entity and relationship description summarization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from recon_graphrag.extraction.parser import DescriptionSummaryParser
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM


class DescriptionSummarizationError(RuntimeError):
    """Raised when mandatory description summarization fails."""


class DescriptionSummarizer:
    def __init__(
        self,
        llm: BaseLLM,
        graph_store: GraphStore,
        graph_name: str,
        concurrency: int = 5,
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.graph_name = graph_name
        self.concurrency = max(int(concurrency), 1)
        self.parser = DescriptionSummaryParser()
        self.prompts = SchemaPromptBuilder()

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
            candidates = fetch(self.graph_name, limit=limit)
            if not candidates:
                return {"summarized": total, "batches": batches}

            fingerprints = {item["id"]: fingerprint(item) for item in candidates}
            try:
                summaries = await self._summary_payload(
                    candidates, fingerprints, prompt
                )
                persist(self.graph_name, summaries)
            except Exception as exc:
                error_rows = [
                    {
                        "id": item["id"],
                        "description": None,
                        "descriptions": self._clean_descriptions(item),
                        "description_summary_status": "failed",
                        "description_input_fingerprint": fingerprints[item["id"]],
                        "description_summary_updated": int(time.time() * 1000),
                        "description_summary_error": str(exc),
                    }
                    for item in candidates
                ]
                persist(self.graph_name, error_rows)
                raise DescriptionSummarizationError(
                    f"Failed to summarize {label} descriptions: {exc}"
                ) from exc

            total += len(summaries)
            batches += 1

    async def _summary_payload(
        self, items: list[dict], fingerprints: dict[str, str], prompt_builder
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _summarize(item: dict) -> dict:
            async with semaphore:
                response = await self.llm.ainvoke(prompt_builder(item))
                summary = self._parse_summary(response.content)
                return {
                    "id": item["id"],
                    "description": summary,
                    "descriptions": self._clean_descriptions(item),
                    "description_summary_status": "success",
                    "description_input_fingerprint": fingerprints[item["id"]],
                    "description_summary_updated": int(time.time() * 1000),
                    "description_summary_error": None,
                }

        return await asyncio.gather(*[_summarize(item) for item in items])

    def _build_entity_summary_prompt(self, entity: dict) -> str:
        return self.prompts.build_entity_summary_prompt(
            descriptions=self._clean_descriptions(entity),
            entity_name=str(entity.get("name") or entity.get("entity_id") or ""),
            entity_type=str(entity.get("type") or "Entity"),
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

    def _clean_descriptions(self, item: dict) -> list[str]:
        return [
            str(description).strip()
            for description in (item.get("descriptions") or [])
            if str(description).strip()
        ]

    def _hash_payload(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
