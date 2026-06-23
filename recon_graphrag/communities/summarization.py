"""LLM-based community summarization.

For each community, collect its entities and relationships, format as structured
text, and generate a summary via LLM. This enables global-level retrieval over
high-level community insights instead of individual nodes.

When ``use_reports=True``, generates structured CommunityReport objects with
validated findings and references instead of plain-text summaries.

Supports concurrent generation within a level and fingerprint-based resume
to skip communities whose context has not changed.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from recon_graphrag.communities.context import (
    CommunityContext,
    enrich_context_with_claims,
    build_reference_ids,
    parse_community_context,
    render_community_context,
)
from recon_graphrag.communities.reports import (
    ReportParser,
    ReportRubric,
    ReportValidationError,
    build_repair_prompt,
    build_report_prompt,
)
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM, LLMResponse
from recon_graphrag.models.artifacts import CommunityReport, report_to_text


DEFAULT_SUMMARY_PROMPT = """Summarize the following cluster of related entities and their connections.

Entities and relationships:
{context}

Generate a concise but comprehensive summary (2-4 paragraphs) that:
1. Identifies the main theme or area covered
2. Describes the key entities involved
3. Highlights important patterns and connections
4. Notes any notable insights or implications

Write in plain, clear language. Do not mention communities, graphs, nodes, or edges.

Summary:"""


@dataclass
class BuildStats:
    """Per-level build statistics."""

    level: int = 0
    attempted: int = 0
    skipped: int = 0
    succeeded: int = 0
    repaired: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0


class CommunitySummarizer:
    """Generate LLM summaries for each community in the knowledge graph."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
        graph_name: str = "entity-graph",
        use_reports: bool = False,
        report_rubric: ReportRubric | None = None,
        concurrency: int = 1,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_SUMMARY_PROMPT
        self.graph_name = graph_name
        self.use_reports = use_reports
        self.report_rubric = report_rubric
        self.concurrency = concurrency
        self._report_parser = ReportParser()

    async def summarize_all(
        self, level: int = 0, skip_existing: bool = False
    ) -> tuple[list[dict], BuildStats]:
        """Summarize all communities at a given hierarchy level.

        Args:
            level: Community hierarchy level to summarize.
            skip_existing: If True, skip communities that already have a
                summary (fingerprint-based resume).

        Returns:
            Tuple of (results list, build stats).
        """
        communities = self.graph_store.get_communities(self.graph_name, level=level)
        if not communities:
            print(f"  No communities found at level {level}")
            return [], BuildStats(level=level)

        start = time.monotonic()
        stats = BuildStats(level=level)
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[dict] = []

        async def _process_one(comm: dict) -> dict | None:
            cid = comm["id"]
            async with semaphore:
                stats.attempted += 1

                # Fingerprint-based resume
                if skip_existing and self._has_existing_summary(cid, level):
                    stats.skipped += 1
                    print(f"  Skipping community {cid} (already summarized)")
                    return None

                entity_count = comm.get("entity_count", 0)
                print(f"  Summarizing community {cid} ({entity_count} entities)...")
                try:
                    if self.use_reports:
                        report = await self.generate_report(cid, level)
                        summary_text = report_to_text(report)
                        self.graph_store.store_community_summary(
                            cid, level, summary_text, self.graph_name
                        )
                        stats.succeeded += 1
                        return {
                            "id": cid,
                            "level": level,
                            "summary": summary_text,
                            "report": report,
                        }
                    else:
                        summary = await self.summarize_community(cid, level)
                        if not summary.strip():
                            stats.failed += 1
                            return None
                        self.graph_store.store_community_summary(
                            cid, level, summary, self.graph_name
                        )
                        stats.succeeded += 1
                        return {"id": cid, "level": level, "summary": summary}
                except Exception as e:
                    stats.failed += 1
                    print(f"  Error summarizing community {cid}: {e}")
                    return None

        tasks = [_process_one(comm) for comm in communities]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                stats.failed += 1
                print(f"  Unexpected error: {outcome}")
            elif outcome is not None:
                results.append(outcome)

        stats.elapsed_seconds = time.monotonic() - start
        return results, stats

    def _has_existing_summary(self, community_id: str, level: int) -> bool:
        """Check if a community already has a stored summary."""
        try:
            rows = self.graph_store.get_community_summaries_by_keys(
                graph_name=self.graph_name,
                keys=[{"id": community_id, "level": level}],
                top_k=1,
            )
            if rows and rows[0].get("summary", "").strip():
                return True
        except Exception:
            pass
        return False

    async def summarize_community(self, community_id: str, level: int = 0) -> str:
        """Summarize a single community by collecting its context."""
        context = self._fetch_community_context(community_id, level)
        if not context.strip():
            return ""

        prompt = self.prompt_template.format(context=context)
        response: LLMResponse = await self.llm.ainvoke(prompt)
        return response.content

    async def generate_report(
        self, community_id: str, level: int = 0
    ) -> CommunityReport:
        """Generate a structured community report with validated references.

        Fetches context, builds a structured prompt, parses the LLM response,
        validates references, and attempts one repair on failure.
        """
        # Fetch context with claims
        context = self._fetch_community_context_obj(community_id, level)
        if not context.edges and not context.entities:
            return CommunityReport(
                id=f"report:{community_id}:{level}",
                community_id=community_id,
                level=level,
                title="Empty community",
                summary="No entities or relationships found.",
            )

        # Build reference allowlist
        reference_ids = build_reference_ids(context)
        valid_ids = set(reference_ids)

        # Render context text
        context_text = render_community_context(context)

        # Build prompt
        prompt = build_report_prompt(
            community_id=community_id,
            level=level,
            context=context_text,
            reference_ids=reference_ids,
            rubric=self.report_rubric,
        )

        # First attempt
        response: LLMResponse = await self.llm.ainvoke(prompt)
        try:
            return self._report_parser.parse(
                response.content,
                community_id=community_id,
                level=level,
                valid_ids=valid_ids,
            )
        except ReportValidationError as e:
            # One repair attempt
            print(f"  Report validation failed for {community_id}, attempting repair...")
            repair_prompt = build_repair_prompt(
                raw_content=e.raw_content,
                errors=e.errors,
                valid_ids=reference_ids,
                rubric=self.report_rubric,
            )
            repair_response = await self.llm.ainvoke(repair_prompt)
            try:
                return self._report_parser.parse(
                    repair_response.content,
                    community_id=community_id,
                    level=level,
                    valid_ids=valid_ids,
                )
            except ReportValidationError as e2:
                print(f"  Repair failed for {community_id}: {e2}")
                # Return a minimal report with error info
                return CommunityReport(
                    id=f"report:{community_id}:{level}",
                    community_id=community_id,
                    level=level,
                    title="Generation failed",
                    summary=response.content[:500] if response.content else "",
                )

    def _fetch_community_context(self, community_id: str, level: int = 0) -> str:
        """Fetch context for a community as rendered text.

        Level 0: degree-ranked entities and intra-community relationships.
        Level > 0: child community summaries first, then entity context fallback.
        """
        if level == 0:
            return self._fetch_ranked_entity_context(community_id, level)

        child_context = self._fetch_child_summary_context(community_id, level)
        if child_context.strip():
            return child_context

        return self._fetch_ranked_entity_context(community_id, level)

    def _fetch_community_context_obj(
        self, community_id: str, level: int = 0
    ) -> CommunityContext:
        """Fetch context as a typed CommunityContext with claims.

        Used by report generation to get structured context for reference IDs.
        """
        rows = self.graph_store.get_community_ranked_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
        )
        context = parse_community_context(community_id, level, rows)

        # Enrich with claims
        entity_ids = [e.id for e in context.entities]
        for edge in context.edges:
            if edge.source.id not in entity_ids:
                entity_ids.append(edge.source.id)
            if edge.target.id not in entity_ids:
                entity_ids.append(edge.target.id)

        if entity_ids:
            try:
                claim_rows = self.graph_store.get_claims_for_entities(
                    graph_name=self.graph_name,
                    entity_ids=entity_ids,
                )
                if claim_rows:
                    context = enrich_context_with_claims(context, claim_rows)
            except Exception:
                pass  # Claims are optional; don't fail report generation

        return context

    def _fetch_ranked_entity_context(self, community_id: str, level: int = 0) -> str:
        """Fetch degree-ranked entity and relationship context."""
        rows = self.graph_store.get_community_ranked_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
        )
        context = parse_community_context(community_id, level, rows)
        return render_community_context(context)

    def _fetch_entity_context(self, community_id: str, level: int = 0) -> str:
        """Fetch all entities and intra-community relationships as text.

        Legacy method using raw backend node objects. Kept for backward
        compatibility during transition.
        """
        results = self.graph_store.get_community_entity_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
        )
        lines = []
        seen_entities = set()
        for record in results:
            entity = record["e"]
            non_entity_labels = self._domain_labels(entity)
            label = list(non_entity_labels)[0] if non_entity_labels else "Entity"
            name = self._node_property(entity, "name") or self._node_property(
                entity, "description"
            )
            key = f"{label}:{name}"
            if key not in seen_entities:
                lines.append(f"- [{label}] {name}")
                seen_entities.add(key)

            other = record["other"]
            if other and record["rel_type"]:
                other_name = self._node_property(other, "name") or self._node_property(
                    other, "description"
                )
                lines.append(f"  {name} --[{record['rel_type']}]--> {other_name}")

        return "\n".join(lines)

    @staticmethod
    def _domain_labels(entity) -> list[str]:
        """Return labels excluding the internal entity marker."""
        labels = getattr(entity, "labels", [])
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, Iterable):
            labels = []

        return [label for label in labels if label != "__Entity__"]

    @staticmethod
    def _node_property(entity, key: str, default: str = ""):
        """Read a node property from dict-like or backend node objects."""
        if entity is None:
            return default

        if hasattr(entity, "get"):
            return entity.get(key, default)

        properties = getattr(entity, "properties", None)
        if isinstance(properties, dict):
            return properties.get(key, default)

        try:
            return entity[key]
        except (KeyError, TypeError, AttributeError):
            return default

    def _fetch_child_summary_context(self, community_id: str, level: int) -> str:
        """Fetch child community summaries for higher-level communities."""
        results = self.graph_store.get_community_child_summary_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
            child_level=level - 1,
        )
        if not results:
            return ""

        lines = []
        for record in results:
            lines.append(f"--- Sub-community {record['id']} (level {record['level']}) ---")
            lines.append(record["summary"])
            lines.append("")
        return "\n".join(lines)
