"""LLM-based community report generation.

For each community, collect its entities and relationships, format as structured
text, and generate a structured report via LLM. This enables global retrieval over
high-level community insights instead of individual nodes.

Generates structured CommunityReport objects with validated findings and
references.

Supports concurrent generation within a level and fingerprint-based resume
to skip communities whose context has not changed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass

from recon_graphrag.communities.context import (
    CommunityContext,
    enrich_context_with_claims,
    build_reference_ids,
    build_packed_reference_ids,
    pack_community_context,
    parse_community_context,
    render_community_context,
)
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    PackItem,
    TokenCounter,
    pack_items,
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
    """Generate LLM reports for each community in the knowledge graph."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        graph_name: str = "entity-graph",
        report_rubric: ReportRubric | None = None,
        concurrency: int = 1,
        max_context_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.graph_name = graph_name
        self.report_rubric = report_rubric
        self.concurrency = concurrency
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter
        self._report_parser = ReportParser()

    async def generate_all(
        self, level: int = 0, skip_existing: bool = False
    ) -> tuple[list[dict], BuildStats]:
        """Generate reports for all communities at a hierarchy level.

        Args:
            level: Community hierarchy level to report.
            skip_existing: If True, skip communities that already have a
                report (fingerprint-based resume).

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
                report_context: CommunityContext | None = None
                input_fingerprint: str | None = None
                if skip_existing:
                    report_context = self._fetch_community_context_obj(cid, level)
                    input_fingerprint = self._report_input_fingerprint(
                        report_context, cid, level
                    )
                    if self._has_existing_report(cid, level, input_fingerprint):
                        stats.skipped += 1
                        print(f"  Skipping community {cid} (report unchanged)")
                        return None

                entity_count = comm.get("entity_count", 0)
                print(f"  Reporting community {cid} ({entity_count} entities)...")
                try:
                    report = await self.generate_report(
                        cid,
                        level,
                        context=report_context,
                        input_fingerprint=input_fingerprint,
                    )
                    report_text = report_to_text(report)
                    self.graph_store.store_community_report(report, self.graph_name)
                    stats.succeeded += 1
                    return {
                        "id": cid,
                        "level": level,
                        "report_text": report_text,
                        "report": report,
                    }
                except Exception as e:
                    stats.failed += 1
                    try:
                        self.graph_store.mark_community_report_failed(
                            self.graph_name,
                            cid,
                            level,
                            str(e),
                        )
                    except Exception:
                        pass
                    print(f"  Error reporting community {cid}: {e}")
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

    def _has_existing_report(
        self,
        community_id: str,
        level: int,
        input_fingerprint: str | None = None,
    ) -> bool:
        """Check if a community already has a current stored report."""
        try:
            rows = self.graph_store.get_community_reports_by_keys(
                graph_name=self.graph_name,
                keys=[{"id": community_id, "level": level}],
                top_k=1,
            )
            if not rows or not rows[0].get("report_text", "").strip():
                return False
            if input_fingerprint is None:
                return True
            return rows[0].get("input_fingerprint") == input_fingerprint
        except Exception:
            pass
        return False

    async def generate_report(
        self,
        community_id: str,
        level: int = 0,
        context: CommunityContext | None = None,
        input_fingerprint: str | None = None,
    ) -> CommunityReport:
        """Generate a structured community report with validated references.

        Fetches context, builds a structured prompt, parses the LLM response,
        validates references, and attempts one repair on failure.
        """
        # Fetch context with claims
        context = context or self._fetch_community_context_obj(community_id, level)
        input_fingerprint = input_fingerprint or self._report_input_fingerprint(
            context, community_id, level
        )
        if not context.edges and not context.entities:
            report = CommunityReport(
                id=f"report:{community_id}:{level}",
                community_id=community_id,
                level=level,
                title="Empty community",
                summary="No entities or relationships found.",
            )
            report.version.input_fingerprint = input_fingerprint
            return report

        # Build reference allowlist
        reference_ids = build_reference_ids(context)
        valid_ids = set(reference_ids)

        # Render context text (packed if budget is set)
        context_tokens_used: int | None = None
        context_truncated = False
        if self.max_context_tokens is not None:
            packed = pack_community_context(
                context,
                max_tokens=self.max_context_tokens,
                counter=self.token_counter,
            )
            context_text = packed.text
            context_tokens_used = packed.used_tokens
            context_truncated = packed.truncated
            # Use allowlist from packed context so findings can only
            # reference items the LLM actually saw.
            reference_ids = build_packed_reference_ids(context, packed)
            valid_ids = set(reference_ids)
            if packed.truncated:
                child_rows = self._fetch_child_report_rows(community_id, level)
                if child_rows:
                    counter = self.token_counter or ApproximateTokenCounter()
                    child_budget = int(self.max_context_tokens * 0.6)
                    child_items = [
                        PackItem(
                            id=str(row["id"]),
                            text=(
                                f"--- Sub-community {row['id']} "
                                f"(level {row['level']}) ---\n{row['report_text']}"
                            ),
                            priority=float(row.get("context_tokens_used", 0)),
                        )
                        for row in child_rows
                        if str(row.get("report_text", "")).strip()
                    ]
                    packed_children = pack_items(
                        child_items, child_budget, counter, truncate_oversized=True
                    )
                    remaining = max(
                        self.max_context_tokens - packed_children.used_tokens, 0
                    )
                    packed_direct = pack_community_context(
                        context, max_tokens=remaining, counter=counter
                    )
                    context_text = "\n\n".join(
                        part
                        for part in (
                            "\n\n".join(
                                item.text for item in packed_children.included
                            ),
                            packed_direct.text,
                        )
                        if part
                    )
                    context_tokens_used = (
                        packed_children.used_tokens + packed_direct.used_tokens
                    )
                    context_truncated = True
                    reference_ids = build_packed_reference_ids(
                        context, packed_direct
                    )
                    valid_ids = set(reference_ids)
        else:
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
            report = self._report_parser.parse(
                response.content,
                community_id=community_id,
                level=level,
                valid_ids=valid_ids,
            )
            report.version.input_fingerprint = input_fingerprint
            report.context_tokens_used = context_tokens_used
            report.context_truncated = context_truncated
            return report
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
                report = self._report_parser.parse(
                    repair_response.content,
                    community_id=community_id,
                    level=level,
                    valid_ids=valid_ids,
                )
                report.version.input_fingerprint = input_fingerprint
                report.context_tokens_used = context_tokens_used
                report.context_truncated = context_truncated
                return report
            except ReportValidationError as e2:
                print(f"  Repair failed for {community_id}: {e2}")
                raise e2

    def _context_fingerprint(self, context: CommunityContext) -> str:
        """Return a stable fingerprint for community report inputs."""
        entities = {
            entity.id: {
                "id": entity.id,
                "name": entity.name,
                "description": entity.description,
                "labels": sorted(entity.labels),
            }
            for entity in context.entities
        }
        for edge in context.edges:
            for entity in (edge.source, edge.target):
                entities.setdefault(
                    entity.id,
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "description": entity.description,
                        "labels": sorted(entity.labels),
                    },
                )

        payload = {
            "community_id": context.community_id,
            "level": context.level,
            "entities": sorted(entities.values(), key=lambda item: item["id"]),
            "edges": sorted(
                [
                    {
                        "source": edge.source.id,
                        "target": edge.target.id,
                        "type": edge.relationship_type,
                        "description": edge.description,
                        "observation_count": edge.observation_count,
                    }
                    for edge in context.edges
                ],
                key=lambda item: (item["source"], item["target"], item["type"]),
            ),
            "claims": sorted(
                [
                    {
                        "id": claim.id,
                        "entity_id": claim.entity_id,
                        "claim_type": claim.claim_type,
                        "description": claim.description,
                        "status": claim.status,
                    }
                    for claim in context.claims
                ],
                key=lambda item: item["id"],
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _report_input_fingerprint(
        self,
        context: CommunityContext,
        community_id: str,
        level: int,
    ) -> str:
        """Fingerprint direct context plus the child reports a parent consumes."""
        children = self._fetch_child_report_rows(community_id, level)
        if not children:
            return self._context_fingerprint(context)
        payload = {
            "context": self._context_fingerprint(context),
            "children": [
                {
                    "id": row.get("id"),
                    "input_fingerprint": row.get("input_fingerprint"),
                    "report_text": row.get("report_text", ""),
                }
                for row in children
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _fetch_child_report_rows(
        self, community_id: str, level: int
    ) -> list[dict]:
        rows = self.graph_store.get_child_community_reports(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
            child_level=level + 1,
        )
        return sorted(
            rows or [],
            key=lambda row: (
                -int(row.get("context_tokens_used", 0) or 0),
                str(row.get("id", "")),
            ),
        )

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
