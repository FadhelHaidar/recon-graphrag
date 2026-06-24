"""Global search: community-summaries-based scored map-reduce retrieval.

Reads community reports at one level, shuffles for unbiased ordering,
packs into token-budgeted batches, runs parallel map calls that score
helpfulness, filters and sorts by score, and reduces into a final answer.
"""

from __future__ import annotations

import asyncio
import json
import random as _random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.citations import resolve_reference_citations
from recon_graphrag.retrieval.community_levels import (
    CommunityLevelSelector,
    resolve_community_level,
)
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEFAULT_MAP_PROMPT = """\
You are answering a question using community reports from a knowledge graph.

## Question
{query}

## Community Reports (sorted by helpfulness for this question)
{batch_text}

## Instructions
1. Answer the question using ONLY information from the reports above.
2. Rate how helpful these reports are for answering the question (0-100).
   - 0 = reports contain no relevant information
   - 100 = reports directly and completely answer the question
3. Cite report IDs in your answer using [Report:id] format.
4. If the report text includes stable evidence IDs you used, include them in
   references. Use target_type "entity", "relationship", or "claim".
5. Return valid JSON only.

JSON format:
{{
  "answer": "Your answer based on the reports...",
  "helpfulness": 75,
  "report_ids": ["report:1:0", "report:2:0"],
  "references": [
    {{"target_id": "entity-or-claim-id", "target_type": "entity"}}
  ]
}}

If the reports contain no relevant information, set helpfulness to 0 and
answer to "No relevant information found."
"""

DEFAULT_REDUCE_PROMPT = """\
You are synthesizing partial answers into a comprehensive final answer.

## Question
{query}

## Partial Answers (sorted by helpfulness)
{partial_text}

## Instructions
1. Combine these partial answers into one coherent, comprehensive answer.
2. Remove redundancy and resolve contradictions.
3. Preserve specific details and cite report IDs where appropriate.
4. If no partial answers contain relevant information, say so.

Final Answer:"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MapBatch:
    """A prepared batch of reports for the map phase."""

    batch_id: str
    report_ids: list[str]
    text: str
    token_count: int


@dataclass
class PartialAnswer:
    """A scored partial answer from one map batch."""

    batch_id: str
    answer: str
    helpfulness: int
    report_ids: list[str]
    references: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class GlobalSearchDiagnostics:
    """Diagnostics for global search."""

    selected_level: int | None = None
    random_seed: int | None = None
    reports_available: int = 0
    reports_used: int = 0
    map_batches: int = 0
    map_succeeded: int = 0
    map_failed: int = 0
    map_filtered_zero: int = 0
    reduce_partials_used: int = 0
    elapsed_ms: int = 0


# ---------------------------------------------------------------------------
# Global search retriever
# ---------------------------------------------------------------------------


def _diag_to_dict(diag: GlobalSearchDiagnostics) -> dict:
    """Convert diagnostics to a plain dict for SearchResult.metadata."""
    return {
        "selected_level": diag.selected_level,
        "random_seed": diag.random_seed,
        "reports_available": diag.reports_available,
        "reports_used": diag.reports_used,
        "map_batches": diag.map_batches,
        "map_succeeded": diag.map_succeeded,
        "map_failed": diag.map_failed,
        "map_filtered_zero": diag.map_filtered_zero,
        "reduce_partials_used": diag.reduce_partials_used,
        "elapsed_ms": diag.elapsed_ms,
    }


class GlobalSearchRetriever(BaseRetriever):
    """Global search: read all reports at a level, scored map-reduce.

    Reads every community report at one hierarchy level, shuffles for
    unbiased ordering, packs into token-budgeted batches, runs parallel
    map calls that score helpfulness, filters zero-score batches, sorts
    by score, and reduces into a final answer with citations.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        map_prompt: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
        graph_name: str = "entity-graph",
        token_counter: TokenCounter | None = None,
        map_budget_tokens: int = 12000,
        reduce_budget_tokens: int = 12000,
        map_concurrency: int = 5,
        max_map_calls: int | None = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.map_prompt = map_prompt or DEFAULT_MAP_PROMPT
        self.reduce_prompt = reduce_prompt or DEFAULT_REDUCE_PROMPT
        self.graph_name = graph_name
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.map_budget_tokens = map_budget_tokens
        self.reduce_budget_tokens = reduce_budget_tokens
        self.map_concurrency = map_concurrency
        self.max_map_calls = max_map_calls

    async def search(
        self,
        query: str,
        level: CommunityLevelSelector = None,
        community_level: CommunityLevelSelector = None,
        random_seed: int | None = 42,
    ) -> SearchResult:
        """Run global search over community reports.

        Args:
            query: User question.
            level/community_level: Community hierarchy level.
            random_seed: Seed for reproducible report shuffling.
        """
        start = time.monotonic()
        diag = GlobalSearchDiagnostics(random_seed=random_seed)

        selected_level = community_level if community_level is not None else level
        resolved_level = resolve_community_level(
            self.graph_store,
            self.graph_name,
            selected_level,
        )

        if resolved_level is None:
            return SearchResult(
                query=query,
                mode="global",
                answer="Global search requires an explicit community level.",
            )

        diag.selected_level = resolved_level

        # 1. Read all reports at this level
        reports = self._read_reports(resolved_level)
        diag.reports_available = len(reports)

        if not reports:
            return SearchResult(
                query=query,
                mode="global",
                answer="No community reports found at the specified level.",
            )

        # 2. Filter failed/empty reports
        reports = [r for r in reports if r.get("summary", "").strip()]
        diag.reports_used = len(reports)

        if not reports:
            return SearchResult(
                query=query,
                mode="global",
                answer="All community reports at this level are empty.",
            )

        # 3. Shuffle
        reports = self._shuffle(reports, random_seed)

        # 4. Batch
        batches = self._create_batches(query, reports)
        diag.map_batches = len(batches)

        # Limit map calls if configured
        if self.max_map_calls and len(batches) > self.max_map_calls:
            batches = batches[: self.max_map_calls]
            diag.map_batches = len(batches)

        # 5. Map phase
        partials = await self._map_phase(query, batches)
        diag.map_succeeded = sum(1 for p in partials if p.error is None)
        diag.map_failed = sum(1 for p in partials if p.error is not None)

        # 6. Filter score=0
        scored = [p for p in partials if p.error is None and p.helpfulness > 0]
        diag.map_filtered_zero = diag.map_succeeded - len(scored)

        if not scored:
            diag.elapsed_ms = int((time.monotonic() - start) * 1000)
            return SearchResult(
                query=query,
                mode="global",
                answer="No relevant information found in community reports.",
                metadata=_diag_to_dict(diag),
            )

        # 7. Sort by helpfulness DESC
        scored.sort(key=lambda p: (-p.helpfulness, p.batch_id))

        # 8. Reduce phase
        answer = await self._reduce_phase(query, scored)
        diag.reduce_partials_used = len(scored)
        diag.elapsed_ms = int((time.monotonic() - start) * 1000)

        # Build context from used reports
        context_parts = []
        for p in scored:
            context_parts.append(
                f"[Batch {p.batch_id}] (score: {p.helpfulness})\n{p.answer}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Resolve citations from validated references returned by map.
        references = []
        seen_refs: set[tuple[str, str]] = set()
        for partial in scored:
            for ref in partial.references:
                key = (
                    str(ref.get("target_type", "")),
                    str(ref.get("target_id", "")),
                )
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                references.append(ref)

        citations = []
        try:
            citations = resolve_reference_citations(
                self.graph_store, self.graph_name, references
            )
        except Exception:
            pass  # Citations are non-fatal

        return SearchResult(
            query=query,
            mode="global",
            answer=answer,
            context=context,
            metadata=_diag_to_dict(diag),
            citations=citations,
        )

    # ------------------------------------------------------------------
    # Report reading
    # ------------------------------------------------------------------

    def _read_reports(self, level: int) -> list[dict]:
        """Read all community reports at a given level."""
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE coalesce(c.report_status, 'success') <> 'failed'
          AND coalesce(c.report_text, c.summary, '') <> ''
        RETURN c.id AS id,
               c.level AS level,
               coalesce(c.report_text, c.summary) AS summary
        ORDER BY c.id
        """
        return self.graph_store.execute_query(
            query, {"graph_name": self.graph_name, "level": level}
        )

    # ------------------------------------------------------------------
    # Shuffling and batching
    # ------------------------------------------------------------------

    @staticmethod
    def _shuffle(reports: list[dict], seed: int | None) -> list[dict]:
        """Shuffle reports with a reproducible seed."""
        shuffled = list(reports)
        rng = _random.Random(seed)
        rng.shuffle(shuffled)
        return shuffled

    def _create_batches(self, query: str, reports: list[dict]) -> list[MapBatch]:
        """Pack reports into token-budgeted batches."""
        counter = self.token_counter

        # Reserve tokens for prompt template and query
        prompt_overhead = counter.count(
            self.map_prompt.format(query=query, batch_text="")
        )
        available = self.map_budget_tokens - prompt_overhead
        if available <= 0:
            available = 1000  # fallback

        batches: list[MapBatch] = []
        current_texts: list[str] = []
        current_ids: list[str] = []
        current_tokens = 0
        batch_idx = 0

        for report in reports:
            rid = str(report.get("id", ""))
            summary = report.get("summary", "")
            text = f"Report {rid}:\n{summary}"
            tokens = counter.count(text)

            if current_tokens + tokens > available and current_texts:
                batches.append(
                    MapBatch(
                        batch_id=str(batch_idx),
                        report_ids=list(current_ids),
                        text="\n\n".join(current_texts),
                        token_count=current_tokens,
                    )
                )
                batch_idx += 1
                current_texts = []
                current_ids = []
                current_tokens = 0

            current_texts.append(text)
            current_ids.append(rid)
            current_tokens += tokens

        if current_texts:
            batches.append(
                MapBatch(
                    batch_id=str(batch_idx),
                    report_ids=list(current_ids),
                    text="\n\n".join(current_texts),
                    token_count=current_tokens,
                )
            )

        return batches

    # ------------------------------------------------------------------
    # Map phase
    # ------------------------------------------------------------------

    async def _map_phase(
        self, query: str, batches: list[MapBatch]
    ) -> list[PartialAnswer]:
        """Run map phase with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.map_concurrency)

        async def _process_batch(batch: MapBatch) -> PartialAnswer:
            async with semaphore:
                prompt = self.map_prompt.format(
                    query=query, batch_text=batch.text
                )
                try:
                    response = await self.llm.ainvoke(prompt)
                    data = self._parse_map_response(response.content)
                    return PartialAnswer(
                        batch_id=batch.batch_id,
                        answer=data.get("answer", ""),
                        helpfulness=int(data.get("helpfulness", 0)),
                        report_ids=data.get("report_ids", batch.report_ids),
                        references=data.get("references", []),
                    )
                except Exception as e:
                    return PartialAnswer(
                        batch_id=batch.batch_id,
                        answer="",
                        helpfulness=0,
                        report_ids=batch.report_ids,
                        error=str(e),
                    )

        tasks = [_process_batch(b) for b in batches]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _parse_map_response(self, content: str) -> dict:
        """Parse map response JSON, stripping fences."""
        content = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = self._extract_json_object(content)
        if not isinstance(data, dict):
            raise ValueError("Map response must be a JSON object")

        # Validate helpfulness range
        score = data.get("helpfulness", 0)
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            data["helpfulness"] = 0
        else:
            data["helpfulness"] = int(score)

        allowed_types = {"entity", "relationship", "claim"}
        references = []
        for ref in data.get("references", []):
            if not isinstance(ref, dict):
                continue
            target_id = str(ref.get("target_id", "")).strip()
            target_type = str(ref.get("target_type", "")).strip()
            if target_id and target_type in allowed_types:
                references.append(
                    {"target_id": target_id, "target_type": target_type}
                )
        data["references"] = references

        return data

    @staticmethod
    def _extract_json_object(content: str) -> dict:
        """Extract the first JSON object from a response with surrounding prose."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise ValueError("Map response did not contain a JSON object")

    # ------------------------------------------------------------------
    # Reduce phase
    # ------------------------------------------------------------------

    async def _reduce_phase(
        self, query: str, partials: list[PartialAnswer]
    ) -> str:
        """Synthesize partial answers into final answer."""
        counter = self.token_counter

        # Build partial answer text, respecting token budget
        parts: list[str] = []
        used_tokens = 0
        prompt_overhead = counter.count(
            self.reduce_prompt.format(query=query, partial_text="")
        )
        available = self.reduce_budget_tokens - prompt_overhead

        for p in partials:
            refs = ", ".join(
                f"{r['target_type']}:{r['target_id']}" for r in p.references
            )
            text = f"[Score {p.helpfulness}] {p.answer}"
            if refs:
                text += f"\nReferences: {refs}"
            tokens = counter.count(text)
            if used_tokens + tokens > available and parts:
                break
            parts.append(text)
            used_tokens += tokens

        partial_text = "\n\n".join(parts)
        prompt = self.reduce_prompt.format(
            query=query, partial_text=partial_text
        )
        response = await self.llm.ainvoke(prompt)
        return response.content
