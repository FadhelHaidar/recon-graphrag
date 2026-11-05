"""Global search: community-report-based scored map-reduce retrieval.

Reads community reports at one level, shuffles for unbiased ordering,
packs into token-budgeted batches, runs parallel map calls that score
helpfulness, filters and sorts by score, and reduces into a final answer.

The optional ``map_strategy="scored_points"`` replaces batch-level answers
with strict-schema scored points selected into one budgeted context envelope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random as _random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.artifacts import Citation
from recon_graphrag.models.types import SearchResult
from recon_graphrag.observability import run_scope, token_stage, track_usage, usage_snapshot

from recon_graphrag.retrieval.citations import resolve_reference_citations
from recon_graphrag.retrieval.community_levels import (
    CommunityLevelSelector,
    resolve_community_level,
)
from recon_graphrag.retrieval.graph_evidence import GraphEvidenceBuilder
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
)

logger = logging.getLogger(__name__)


# Prompts
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

DEFAULT_POINT_MAP_PROMPT = """\
You are extracting scored evidence points from community reports.

## Question
{query}

## Community Reports
{batch_text}

## Instructions
1. Extract every distinct point the reports support that helps answer the
   question. Each point must stand on its own as a short statement.
2. Score each point 0-100 for how much it helps answer the question.
   - 0 = not relevant
   - 100 = directly and completely answers the question
3. List the report IDs that support the point, using the "Report {{id}}:"
   labels above. Use only IDs from these reports.
4. Return valid JSON only, with no prose before or after.

JSON format:
{{
  "points": [
    {{
      "description": "A self-contained statement from the reports.",
      "score": 75,
      "report_ids": ["report:1:0", "report:2:0"]
    }}
  ]
}}

Return an empty "points" array when the reports contain nothing relevant.
"""

# Strict provider schema for the scored-point map strategy.
GLOBAL_POINT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "global_map_points",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "score": {"type": "integer"},
                            "report_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["description", "score", "report_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    },
}

# Context notices are SDK-controlled: no prompt text, provider text, or IDs.
POINT_OMITTED_NOTICE = (
    "Budgeted point selection: lower-scored points are omitted from this context."
)

MAP_STRATEGIES = ("batch", "scored_points")


class _MapPointsError(ValueError):
    """A point-mode map failure carrying a fixed, publishable reason code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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

GENERAL_KNOWLEDGE_INSTRUCTION = (
    "\n\nNote: The available data does not contain enough information to "
    "answer this question. You are allowed to supplement the answer with "
    "general knowledge, but clearly indicate which parts come from general "
    "knowledge versus the provided data."
)


# Data types
# Cypher LIMIT value used when no report limit is configured.
_UNBOUNDED_ROWS = 2**31 - 1


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
    batch_report_ids: list[str] = field(default_factory=list)
    error: str | None = None
    # Scored-point strategy only: validated points and dropped report IDs.
    points: list[dict] = field(default_factory=list)
    rejected_references: int = 0


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
    source_report_ids_used: int = 0
    source_references_extracted: int = 0
    source_citations_resolved: int = 0
    source_resolution_errors: list[str] = field(default_factory=list)
    elapsed_ms: int = 0


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
        "source_report_ids_used": diag.source_report_ids_used,
        "source_references_extracted": diag.source_references_extracted,
        "source_citations_resolved": diag.source_citations_resolved,
        "source_resolution_errors": list(diag.source_resolution_errors),
        "elapsed_ms": diag.elapsed_ms,
    }


class GlobalSearchRetriever:
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
        max_reports: int | None = None,
        max_map_tokens: int | None = None,
        allow_general_knowledge: bool = False,
        track_token_usage: bool = True,
        *,
        map_strategy: str = "batch",
    ):
        if map_strategy not in MAP_STRATEGIES:
            raise ValueError(
                f"map_strategy must be one of {MAP_STRATEGIES}, got {map_strategy!r}"
            )
        self.graph_store = graph_store
        self.llm = track_usage(llm) if track_token_usage else llm
        self.map_strategy = map_strategy
        default_prompt = (
            DEFAULT_POINT_MAP_PROMPT
            if map_strategy == "scored_points"
            else DEFAULT_MAP_PROMPT
        )
        self.map_prompt = map_prompt or default_prompt
        self.reduce_prompt = reduce_prompt or DEFAULT_REDUCE_PROMPT
        self.graph_name = graph_name
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.map_budget_tokens = map_budget_tokens
        self.reduce_budget_tokens = reduce_budget_tokens
        self.map_concurrency = map_concurrency
        self.max_map_calls = max_map_calls
        self.max_reports = max_reports
        self.max_map_tokens = max_map_tokens
        self.allow_general_knowledge = allow_general_knowledge

    async def search(
        self,
        query: str,
        community_level: CommunityLevelSelector,
        random_seed: int | None = 42,
        synthesize_response: bool = True,
        include_graph_evidence: bool = False,
    ) -> SearchResult:
        """Run global search over community reports.

        Args:
            query: User question.
            community_level: Community hierarchy level.
            random_seed: Seed for reproducible report shuffling.
            synthesize_response: If False, skip the final reduce synthesis and
                return scored partial answers in context with ``answer=""``.
                Map LLM calls still run for relevance scoring and reference
                extraction. Required to be False with
                ``map_strategy="scored_points"``.
            include_graph_evidence: When True, attach the member entities and
                relationships of the community reports behind the partial
                answers that reached the final reduce prompt (extra database
                reads). Reports only mapped or read are not expanded.
        """
        point_mode = self.map_strategy == "scored_points"
        if point_mode:
            if synthesize_response:
                raise ValueError(
                    "map_strategy='scored_points' returns evidence context and "
                    "requires synthesize_response=False"
                )
            if getattr(self.llm, "supports_structured_output", None) is False:
                raise ValueError(
                    "map_strategy='scored_points' requires an LLM adapter with "
                    "structured output support"
                )

        with run_scope():
            start = time.monotonic()
            diag = GlobalSearchDiagnostics(random_seed=random_seed)

            resolved_level = resolve_community_level(
                self.graph_store,
                self.graph_name,
                community_level,
            )

            if resolved_level is None:
                diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                metadata = {}
                token_usage = usage_snapshot(self.llm)
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                return SearchResult(
                    query=query,
                    mode="global",
                    answer="Global search requires an explicit community level.",
                    metadata=metadata,
                )

            diag.selected_level = resolved_level

            phase_start = time.monotonic()
            reports = self._read_reports(resolved_level, self.max_reports)
            read_ms = int((time.monotonic() - phase_start) * 1000)
            diag.reports_available = len(reports)
            logger.debug(
                "global phase=read ms=%d reports=%d", read_ms, len(reports)
            )

            if not reports:
                diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                metadata = {}
                token_usage = usage_snapshot(self.llm)
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                return SearchResult(
                    query=query,
                    mode="global",
                    answer="No community reports found at the specified level.",
                    metadata=metadata,
                )

            reports = [r for r in reports if r.get("report_text", "").strip()]
            diag.reports_used = len(reports)

            if not reports:
                diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                metadata = {}
                token_usage = usage_snapshot(self.llm)
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                return SearchResult(
                    query=query,
                    mode="global",
                    answer="All community reports at this level are empty.",
                    metadata=metadata,
                )

            reports = self._shuffle(reports, random_seed)

            batches = self._create_batches(
                query,
                reports,
                max_batches=self.max_map_calls,
                max_tokens=self.max_map_tokens,
            )
            diag.map_batches = len(batches)

            phase_start = time.monotonic()
            partials = await self._map_phase(query, batches)
            diag.map_succeeded = sum(1 for p in partials if p.error is None)
            diag.map_failed = sum(1 for p in partials if p.error is not None)
            logger.debug(
                "global phase=map ms=%d batches=%d failed=%d concurrency=%d",
                int((time.monotonic() - phase_start) * 1000),
                len(batches),
                diag.map_failed,
                self.map_concurrency,
            )

            scored = [p for p in partials if p.error is None and p.helpfulness > 0]
            diag.map_filtered_zero = diag.map_succeeded - len(scored)

            # Point mode selects its context from every successful point, so the
            # envelope is built before the no-evidence returns below.
            point_meta: dict = {}
            point_context = ""
            if point_mode:
                point_context, point_meta = self._point_context(
                    query, partials, diag
                )

            if point_mode and point_meta["context_failed"]:
                diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                metadata = {**_diag_to_dict(diag), **point_meta}
                metadata["synthesize_response"] = False
                metadata["response_synthesis_skipped"] = True
                token_usage = usage_snapshot(self.llm)
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                return SearchResult(
                    query=query,
                    mode="global",
                    answer="",
                    context="",
                    metadata=metadata,
                )

            use_general_knowledge = False
            if not scored:
                if not self.allow_general_knowledge:
                    diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                    metadata = {**_diag_to_dict(diag), **point_meta}
                    token_usage = usage_snapshot(self.llm)
                    if token_usage is not None:
                        metadata["token_usage"] = token_usage
                    return SearchResult(
                        query=query,
                        mode="global",
                        answer="No relevant information found in community reports.",
                        metadata=metadata,
                    )
                use_general_knowledge = True
                scored = [p for p in partials if p.error is None]

            scored.sort(key=lambda p: (-p.helpfulness, p.batch_id))

            if point_mode:
                context = point_context
            else:
                # Build context from scored reports
                context_parts = []
                for p in scored:
                    context_parts.append(
                        f"[Batch {p.batch_id}] (score: {p.helpfulness})\n{p.answer}"
                    )
                context = "\n\n---\n\n".join(context_parts)

            # Resolve citations from explicit map references and report-derived refs.
            phase_start = time.monotonic()
            citations = self._resolve_source_citations(diag, scored, resolved_level)
            logger.debug(
                "global phase=citations ms=%d refs=%d citations=%d",
                int((time.monotonic() - phase_start) * 1000),
                diag.source_references_extracted,
                diag.source_citations_resolved,
            )

            if not synthesize_response:
                diag.reduce_partials_used = 0
                diag.elapsed_ms = int((time.monotonic() - start) * 1000)
                metadata = {**_diag_to_dict(diag), **point_meta}
                metadata["synthesize_response"] = False
                metadata["response_synthesis_skipped"] = True
                token_usage = usage_snapshot(self.llm)
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                graph = (
                    await self._graph_evidence(
                        scored, resolved_level, synthesis_included=False
                    )
                    if include_graph_evidence
                    else {}
                )
                return SearchResult(
                    query=query,
                    mode="global",
                    answer="",
                    context=context,
                    metadata=metadata,
                    citations=citations,
                    graph=graph,
                )

            phase_start = time.monotonic()
            answer, retained_ids = await self._reduce_phase(
                query, scored, use_general_knowledge=use_general_knowledge
            )
            diag.reduce_partials_used = len(scored)
            logger.debug(
                "global phase=reduce ms=%d partials=%d",
                int((time.monotonic() - phase_start) * 1000),
                len(scored),
            )
            phase_start = time.monotonic()
            graph = (
                await self._graph_evidence(
                    [p for p in scored if p.batch_id in retained_ids],
                    resolved_level,
                    synthesis_included=True,
                )
                if include_graph_evidence
                else {}
            )

            logger.debug(
                "global phase=evidence ms=%d enabled=%s",
                int((time.monotonic() - phase_start) * 1000),
                include_graph_evidence,
            )

            diag.elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                "global total ms=%d level=%s reports=%d batches=%d",
                diag.elapsed_ms,
                resolved_level,
                diag.reports_used,
                diag.map_batches,
            )

            metadata = _diag_to_dict(diag)
            token_usage = usage_snapshot(self.llm)
            if token_usage is not None:
                metadata["token_usage"] = token_usage
            return SearchResult(
                query=query,
                mode="global",
                answer=answer,
                context=context,
                metadata=metadata,
                citations=citations,
                graph=graph,
            )

    async def _graph_evidence(
        self,
        partials: list[PartialAnswer],
        level: int,
        *,
        synthesis_included: bool,
    ) -> dict:
        """Expand the reports behind ``partials`` into entities and edges."""
        try:
            report_ids = self._resolve_used_report_ids(partials)
        except Exception:  # noqa: BLE001 - report selection is best effort
            report_ids = []
        builder = GraphEvidenceBuilder(
            graph_name=self.graph_name, mode="global", graph_store=self.graph_store
        )
        await builder.expand_communities(
            [{"community_id": rid, "level": level} for rid in report_ids],
            origin="report",
        )
        builder.set_provenance(
            retained_batches=sorted(p.batch_id for p in partials),
            contributing_reports=report_ids,
            community_level=level,
            report_selection="map_reported_else_batch",
            synthesis_included=synthesis_included,
        )
        return builder.build()


    def _read_reports(
        self, level: int, max_reports: int | None = None
    ) -> list[dict]:
        """Read community reports at a level, optionally bounded at the DB."""
        report_limit = _UNBOUNDED_ROWS if max_reports is None else max_reports
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE coalesce(c.report_status, 'success') <> 'failed'
          AND c.report_text <> ''
        RETURN c.id AS id,
               c.level AS level,
               c.report_text AS report_text
        ORDER BY c.id
        LIMIT $report_limit
        """
        return self.graph_store.execute_query(
            query,
            {
                "graph_name": self.graph_name,
                "level": level,
                "report_limit": report_limit,
            },
        )

    # Source citation resolution
    def _resolve_source_citations(
        self,
        diag: GlobalSearchDiagnostics,
        scored: list[PartialAnswer],
        resolved_level: int,
    ) -> list[Citation]:
        """Resolve citations from explicit map refs and used community reports."""
        allowed_types = {"entity", "relationship", "claim"}
        references: list[dict] = []
        seen_refs: set[tuple[str, str]] = set()
        errors: list[str] = []

        for partial in scored:
            for ref in partial.references:
                if not isinstance(ref, dict):
                    continue
                target_type = str(ref.get("target_type", "")).strip()
                target_id = str(ref.get("target_id", "")).strip()
                if target_type not in allowed_types or not target_id:
                    continue
                key = (target_type, target_id)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                references.append({"target_type": target_type, "target_id": target_id})

        used_report_ids: list[str] = []
        try:
            used_report_ids = self._resolve_used_report_ids(scored)
        except Exception as e:
            errors.append(f"report-id resolution failed: {e}")

        diag.source_report_ids_used = len(used_report_ids)

        extracted_refs: list[dict] = []
        try:
            reports = self._read_report_references(resolved_level, used_report_ids)
        except Exception as e:
            errors.append(f"report reference lookup failed: {e}")
            reports = []

        for report in reports:
            try:
                json_refs = self._extract_report_json_refs(report.get("report_json"))
                if json_refs:
                    extracted_refs.extend(json_refs)
                else:
                    text_refs = self._extract_report_text_refs(
                        report.get("report_text", "")
                    )
                    extracted_refs.extend(text_refs)
            except Exception as e:
                errors.append(
                    f"reference extraction failed for {report.get('id')}: {e}"
                )

        for ref in extracted_refs:
            target_type = str(ref.get("target_type", "")).strip()
            target_id = str(ref.get("target_id", "")).strip()
            if target_type not in allowed_types or not target_id:
                continue
            key = (target_type, target_id)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            references.append({"target_type": target_type, "target_id": target_id})

        diag.source_references_extracted = len(references)

        citations: list[Citation] = []
        try:
            citations = resolve_reference_citations(
                self.graph_store, self.graph_name, references
            )
        except Exception as e:
            errors.append(f"citation resolution failed: {e}")

        diag.source_citations_resolved = len(citations)
        diag.source_resolution_errors = errors
        return citations

    @staticmethod
    def _resolve_used_report_ids(partials: list[PartialAnswer]) -> list[str]:
        """Select validated community report IDs used by helpful partials."""
        used: list[str] = []
        seen: set[str] = set()
        for partial in partials:
            batch_ids = set(partial.batch_report_ids)
            returned = [str(r) for r in (partial.report_ids or [])]
            valid = [r for r in returned if r in batch_ids]
            if not valid:
                valid = list(partial.batch_report_ids)
            for rid in valid:
                if rid and rid not in seen:
                    seen.add(rid)
                    used.append(rid)
        return used

    def _read_report_references(
        self, level: int, report_ids: list[str]
    ) -> list[dict]:
        """Fetch report JSON and rendered text for selected report IDs."""
        if not report_ids:
            return []
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE c.id IN $report_ids
        RETURN c.id AS id,
               c.report_json AS report_json,
               c.report_text AS report_text
        ORDER BY c.id
        """
        return self.graph_store.execute_query(
            query,
            {
                "graph_name": self.graph_name,
                "level": level,
                "report_ids": report_ids,
            },
        )

    @staticmethod
    def _extract_report_json_refs(report_json: str | None) -> list[dict]:
        """Extract typed references from structured report JSON.

        Raises:
            ValueError: If the report JSON is malformed or not a JSON object.
        """
        if not report_json:
            return []
        try:
            data = json.loads(report_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"report JSON is not valid: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("report JSON is not an object")
        allowed_types = {"entity", "relationship", "claim"}
        refs: list[dict] = []
        findings = data.get("findings", [])
        if not isinstance(findings, list):
            return refs
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            references = finding.get("references", [])
            if not isinstance(references, list):
                continue
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                target_id = str(ref.get("target_id", "")).strip()
                target_type = str(ref.get("target_type", "")).strip()
                if target_id and target_type in allowed_types:
                    refs.append({"target_id": target_id, "target_type": target_type})
        return refs

    @staticmethod
    def _extract_report_text_refs(report_text: str) -> list[dict]:
        """Parse typed references from rendered report text [refs: ...] blocks."""
        refs: list[dict] = []
        if not report_text:
            return refs
        allowed_types = {"entity", "relationship", "claim"}
        for match in re.finditer(r"\[refs:\s*([^\]]+)\]", report_text):
            entries = match.group(1).split(",")
            for entry in entries:
                entry = entry.strip()
                if ":" not in entry:
                    continue
                target_type, target_id = entry.split(":", 1)
                target_type = target_type.strip()
                target_id = target_id.strip()
                if target_id and target_type in allowed_types:
                    refs.append({"target_id": target_id, "target_type": target_type})
        return refs


    # Shuffling and batching
    @staticmethod
    def _shuffle(reports: list[dict], seed: int | None) -> list[dict]:
        """Shuffle reports with a reproducible seed."""
        shuffled = list(reports)
        rng = _random.Random(seed)
        rng.shuffle(shuffled)
        return shuffled

    def _create_batches(
        self,
        query: str,
        reports: list[dict],
        max_batches: int | None = None,
        max_tokens: int | None = None,
    ) -> list[MapBatch]:
        """Pack reports into token-budgeted batches.

        ``max_tokens`` caps the sum of complete formatted map prompts, using
        the configured token counter. Stop before the first batch that cannot
        fit; do not partially send it. Both caps are off by default.
        """
        counter = self.token_counter

        # Reserve tokens for prompt template and query
        prompt_overhead = counter.count(
            self.map_prompt.format(query=query, batch_text="")
        )
        available = self.map_budget_tokens - prompt_overhead
        if available <= 0:
            raise ValueError("map_budget_tokens is too small for the map prompt")

        batches: list[MapBatch] = []
        current_texts: list[str] = []
        current_ids: list[str] = []
        current_tokens = 0
        total_tokens = 0

        def append_batch() -> bool:
            nonlocal total_tokens
            text = "\n\n".join(current_texts)
            prompt_tokens = counter.count(
                self.map_prompt.format(query=query, batch_text=text)
            )
            if max_tokens is not None and total_tokens + prompt_tokens > max_tokens:
                return False
            batches.append(
                MapBatch(
                    batch_id=str(len(batches)),
                    report_ids=list(current_ids),
                    text=text,
                    token_count=current_tokens,
                )
            )
            total_tokens += prompt_tokens
            return True

        for report in reports:
            if max_batches is not None and len(batches) >= max_batches:
                break
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            rid = str(report.get("id", ""))
            report_text = report.get("report_text", "")
            text = f"Report {rid}:\n{report_text}"
            tokens = counter.count(text)
            if tokens > available:
                text = counter.truncate(text, available)
                tokens = counter.count(text)

            if current_tokens + tokens > available and current_texts:
                if not append_batch():
                    return batches
                if max_batches is not None and len(batches) >= max_batches:
                    return batches
                current_texts = []
                current_ids = []
                current_tokens = 0

            current_texts.append(text)
            current_ids.append(rid)
            current_tokens += tokens

        if current_texts and (max_batches is None or len(batches) < max_batches):
            append_batch()

        return batches

    # Map phase
    def _point_context(
        self,
        query: str,
        partials: list[PartialAnswer],
        diag: GlobalSearchDiagnostics,
    ) -> tuple[str, dict]:
        """Select scored points into one JSON context envelope within budget."""
        counter = self.token_counter
        budget = self.reduce_budget_tokens

        blocks: list[str] = []
        ordered = sorted(
            (
                (partial.batch_id, index, point)
                for partial in partials
                if partial.error is None
                for index, point in enumerate(partial.points)
                if point["score"] > 0
            ),
            key=lambda item: (-item[2]["score"], item[0], item[1]),
        )
        for _, _, point in ordered:
            text = f"[Score {point['score']}] {point['description']}"
            if point["report_ids"]:
                text += "\nReports: " + ", ".join(point["report_ids"])
            blocks.append(text)

        envelope = {
            "query": query,
            "mode": "global",
            "answer": "",
            "context": "",
            "citations": [],
        }

        def over_budget(kept: list[str], notice: str | None) -> bool:
            envelope["context"] = "\n\n".join(([notice] if notice else []) + kept)
            return counter.count(json.dumps(envelope, ensure_ascii=False)) > budget

        if over_budget([], None):
            del envelope["query"]  # Already present in the caller's history.
        if over_budget([], None):
            raise ValueError(
                "reduce_budget_tokens is too small for the global context envelope"
            )

        kept: list[str] = []

        def pack(notice: str | None) -> list[str]:
            # Skip an oversized point and keep packing: no text slicing.
            packed: list[str] = []
            for block in blocks:
                if not over_budget([*packed, block], notice):
                    packed.append(block)
            return packed

        kept = pack(None)
        notice = POINT_OMITTED_NOTICE if len(kept) < len(blocks) else None
        if notice:
            kept = pack(notice)
            if over_budget(kept, notice):
                raise ValueError(
                    "reduce_budget_tokens is too small for the global context envelope"
                )

        dropped = len(blocks) - len(kept)
        context = "\n\n".join(([notice] if notice else []) + kept)

        all_failed = diag.map_batches > 0 and diag.map_failed == diag.map_batches
        warnings: list[str] = []
        if all_failed:
            warnings.append(
                f"All {diag.map_batches} map batches failed; no evidence was returned."
            )
        elif diag.map_failed:
            warnings.append(
                f"{diag.map_failed} of {diag.map_batches} map batches failed; "
                "context may be incomplete."
            )
        rejected = sum(p.rejected_references for p in partials)
        if rejected:
            warnings.append(
                f"Dropped {rejected} report references that did not belong to "
                "their mapped batch."
            )
        if dropped:
            warnings.append(
                f"Omitted {dropped} of {len(blocks)} scored points to fit the "
                "context budget."
            )

        metadata = {
            "context_budget_tokens": budget,
            "context_tokens": counter.count(context),
            "context_parts_available": len(blocks),
            "context_parts_kept": len(kept),
            "context_parts_dropped": dropped,
            "context_warnings": warnings,
            "context_failed": all_failed,
        }
        return context, metadata

    def _point_partial(self, batch: MapBatch, response: Any) -> PartialAnswer:
        """Validate a batch's points and scope their report IDs to the batch."""
        points = self._parse_point_response(response)
        allowed = set(batch.report_ids)
        rejected = 0
        for point in points:
            claimed = list(dict.fromkeys(point["report_ids"]))
            valid = [rid for rid in claimed if rid in allowed]
            rejected += len(claimed) - len(valid)
            point["report_ids"] = valid

        positive = [p for p in points if p["score"] > 0]
        report_ids: list[str] = []
        for point in positive:
            for rid in point["report_ids"]:
                if rid not in report_ids:
                    report_ids.append(rid)

        return PartialAnswer(
            batch_id=batch.batch_id,
            answer="",
            helpfulness=max((p["score"] for p in positive), default=0),
            report_ids=report_ids,
            batch_report_ids=batch.report_ids,
            points=points,
            rejected_references=rejected,
        )

    async def _map_phase(
        self, query: str, batches: list[MapBatch]
    ) -> list[PartialAnswer]:
        """Run map phase with bounded concurrency and bounded pending work."""

        point_mode = self.map_strategy == "scored_points"

        async def _process_batch(batch: MapBatch) -> PartialAnswer:
            prompt = self.map_prompt.format(query=query, batch_text=batch.text)
            if point_mode:
                try:
                    with token_stage("retrieval.global_map"):
                        response = await self.llm.ainvoke(
                            prompt, response_format=GLOBAL_POINT_RESPONSE_FORMAT
                        )
                    return self._point_partial(batch, response)
                except Exception as e:
                    # Fixed reason code only: provider text never reaches logs.
                    reason = (
                        e.reason
                        if isinstance(e, _MapPointsError)
                        else "provider_error"
                    )
                    logger.warning(
                        "global map batch %s failed: %s", batch.batch_id, reason
                    )
                    return PartialAnswer(
                        batch_id=batch.batch_id,
                        answer="",
                        helpfulness=0,
                        report_ids=batch.report_ids,
                        batch_report_ids=batch.report_ids,
                        error=reason,
                    )
            try:
                with token_stage("retrieval.global_map"):
                    response = await self.llm.ainvoke(prompt)
                data = self._parse_map_response(response.content)
                return PartialAnswer(
                    batch_id=batch.batch_id,
                    answer=data.get("answer", ""),
                    helpfulness=int(data.get("helpfulness", 0)),
                    report_ids=data.get("report_ids", batch.report_ids),
                    references=data.get("references", []),
                    batch_report_ids=batch.report_ids,
                )
            except Exception as e:
                return PartialAnswer(
                    batch_id=batch.batch_id,
                    answer="",
                    helpfulness=0,
                    report_ids=batch.report_ids,
                    batch_report_ids=batch.report_ids,
                    error=str(e),
                )
        # Fixed workers bound pending work; positional writes preserve order.
        partials: list[PartialAnswer] = []
        pending = iter(list(enumerate(batches)))

        async def _worker() -> None:
            for position, batch in pending:
                partials_by_position[position] = await _process_batch(batch)

        partials_by_position: list[PartialAnswer | None] = [None] * len(batches)
        workers = [
            asyncio.create_task(_worker())
            for _ in range(max(int(self.map_concurrency), 1))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        partials = [p for p in partials_by_position if p is not None]
        return partials

    def _parse_map_response(self, content: str) -> dict:
        """Parse map response JSON, stripping fences."""
        data = self._json_object(content)

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

    def _parse_point_response(self, response: Any) -> list[dict]:
        """Parse a scored-point map response strictly under the point schema."""
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason in ("length", "max_tokens"):
            raise _MapPointsError("truncated")
        if finish_reason in ("refusal", "content_filter"):
            raise _MapPointsError("refused")
        try:
            data = self._json_object(response.content)
        except ValueError:
            raise _MapPointsError("invalid_schema") from None

        raw_points = data.get("points")
        if not isinstance(raw_points, list):
            raise _MapPointsError("invalid_schema")
        points: list[dict] = []
        for item in raw_points:
            if not isinstance(item, dict):
                raise _MapPointsError("invalid_schema")
            description = item.get("description")
            score = item.get("score")
            report_ids = item.get("report_ids")
            if not isinstance(description, str) or not description.strip():
                raise _MapPointsError("invalid_schema")
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= 100
            ):
                raise _MapPointsError("invalid_schema")
            if not isinstance(report_ids, list) or not all(
                isinstance(rid, str) and rid.strip() for rid in report_ids
            ):
                raise _MapPointsError("invalid_schema")
            points.append(
                {
                    "description": description.strip(),
                    "score": score,
                    "report_ids": [rid.strip() for rid in report_ids],
                }
            )
        return points

    @staticmethod
    def _json_object(content: str) -> dict:
        """Parse the JSON object in a response body, stripping code fences."""
        content = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = GlobalSearchRetriever._extract_json_object(content)
        if not isinstance(data, dict):
            raise ValueError("Map response must be a JSON object")
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

    # Reduce phase
    async def _reduce_phase(
        self, query: str, partials: list[PartialAnswer], *, use_general_knowledge: bool = False,
    ) -> tuple[str, set[str]]:
        """Synthesize partial answers into final answer.

        Returns the answer and the batch ids actually packed into the reduce
        prompt; partials past the token budget never reach synthesis.
        """
        counter = self.token_counter

        # Build partial answer text, respecting token budget
        parts: list[str] = []
        retained_ids: set[str] = set()
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
            retained_ids.add(p.batch_id)
            used_tokens += tokens

        partial_text = "\n\n".join(parts)
        reduce_prompt = self.reduce_prompt
        if use_general_knowledge:
            reduce_prompt = reduce_prompt + GENERAL_KNOWLEDGE_INSTRUCTION
        prompt = reduce_prompt.format(
            query=query, partial_text=partial_text
        )
        with token_stage("retrieval.global_reduce"):
            response = await self.llm.ainvoke(prompt)
        return response.content, retained_ids
