"""Community pipeline: detect → report → embed.

Convenience wrapper that chains the two community steps into a single
build() call. Typically run on a schedule (e.g. weekly) after new entities
have been ingested.
"""

from __future__ import annotations

import logging
from typing import Optional

from recon_graphrag.communities.reports import ReportRubric, validate_report_prompt_template
from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.observability import run_scope, track_usage, usage_snapshot
from recon_graphrag.utils.tokens import TokenCounter

logger = logging.getLogger(__name__)


class CommunityPipeline:
    """Run the full community pipeline: detect → report → embed."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        relationship_types: Optional[list[str]] = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 1e-4,
        graph_name: str = "entity-graph",
        relationship_weight_property: str = "weight",
        random_seed: Optional[int] = 42,
        report_rubric: ReportRubric | None = None,
        report_prompt: str | None = None,
        summarize_concurrency: int = 10,
        skip_existing: bool = False,
        max_context_tokens: int = 8000,
        token_counter: TokenCounter | None = None,
        max_report_words: int | None = 2000,
        embedder: BaseEmbedder | None = None,
        embed_community_reports: bool = True,
        track_token_usage: bool = True,
        include_error_details: bool = False,
    ):
        """Initialize the community pipeline.

        Args:
            graph_store: Store that provides community detection and persistence.
            llm: LLM used to generate structured community reports.
            relationship_types: Relationship types to include in the detection graph.
            max_levels: Maximum number of community hierarchy levels to detect.
            gamma: Leiden resolution parameter.
            theta: Leiden theta parameter.
            tolerance: Leiden tolerance parameter.
            graph_name: Graph scope to detect communities within.
            relationship_weight_property: Name of the numeric relationship property
                to use as the Leiden edge weight, e.g. "weight". Neo4j runs
                unweighted when this is None; Memgraph defaults to "weight".
            random_seed: Random seed for deterministic Neo4j community detection.
            report_rubric: Rating rubric for structured reports.
            report_prompt: Custom report prompt. A simple instruction string
                gets the structured body (context, allowlist, rubric, JSON
                format) appended by the backend; a full template must include
                all required placeholders (see
                ``recon_graphrag.communities.reports.REQUIRED_REPORT_PLACEHOLDERS``).
            summarize_concurrency: Max concurrent LLM calls per level.
            skip_existing: Skip communities whose report input is unchanged.
            max_context_tokens: Maximum tokens for community context passed to the
                LLM. When set, degree-ranked context is greedily packed to fit
                this budget. When None, all context is included.
            token_counter: Token counter for context packing. Defaults to
                ApproximateTokenCounter when max_context_tokens is set.
            max_report_words: Soft word limit stated in the report prompt.
                None omits the instruction.
            embedder: Embedder for community report vector embeddings.
                When provided and embed_community_reports=True, generates
                report embeddings after generation.
            embed_community_reports: Whether to embed community reports
                after report generation. Requires embedder to be set.
            track_token_usage: Whether to auto-wrap ``llm`` with token-usage
                tracking (no-op if it's already wrapped).
            include_error_details: Include original exception messages and types in
                maintenance warnings. For trusted/private debugging only: messages
                may contain credentials, provider details, or source content.
        """
        self.graph_store = graph_store
        self.llm = track_usage(llm) if track_token_usage else llm
        self.relationship_types = relationship_types
        self.max_levels = max_levels
        self.gamma = gamma
        self.theta = theta
        self.tolerance = tolerance
        self.graph_name = graph_name
        self.relationship_weight_property = relationship_weight_property
        self.random_seed = random_seed
        self.report_rubric = report_rubric
        self.report_prompt = report_prompt
        validate_report_prompt_template(report_prompt)
        self.summarize_concurrency = summarize_concurrency
        self.skip_existing = skip_existing
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter
        self.max_report_words = max_report_words
        self.embedder = embedder
        self.embed_community_reports = embed_community_reports
        self.include_error_details = include_error_details

    async def build(self, level: Optional[int] = None) -> dict:
        """Detect communities, generate reports, and embed them.

        Processes levels finest-to-coarsest. Within each level, runs up to
        ``summarize_concurrency`` report generations in parallel.

        Args:
            level: Coarsest community hierarchy level to report.
                If None, processes all detected levels. If provided, lower
                finer levels are processed first so parent reports can use
                child reports.

        Returns:
            Dict with stats from each step, including per-level build stats.
        """
        with run_scope():
            logger.info("detecting communities")
            existing_reports = self.graph_store.count_communities_with_reports(
                self.graph_name
            )
            community_stats = self.graph_store.detect_communities(
                graph_name=self.graph_name,
                relationship_types=self.relationship_types,
                max_levels=self.max_levels,
                gamma=self.gamma,
                theta=self.theta,
                tolerance=self.tolerance,
                relationship_weight_property=self.relationship_weight_property,
                random_seed=self.random_seed,
            )
            logger.info("found %s communities", len(community_stats))
            if existing_reports:
                logger.warning(
                    "community detection replaced %s community(ies) that already "
                    "had reports in graph %r; their reports are regenerated on "
                    "this run.",
                    existing_reports,
                    self.graph_name,
                )

            detected_levels = sorted({s["level"] for s in community_stats}, reverse=True)
            levels = (
                [lvl for lvl in detected_levels if lvl >= level]
                if level is not None
                else detected_levels
            )
            total_reports = 0
            level_stats: list[dict] = []
            maintenance_warnings: list[dict] = []

            summarizer = CommunitySummarizer(
                self.graph_store,
                self.llm,
                graph_name=self.graph_name,
                report_rubric=self.report_rubric,
                concurrency=self.summarize_concurrency,
                max_context_tokens=self.max_context_tokens,
                token_counter=self.token_counter,
                max_report_words=self.max_report_words,
                report_prompt=self.report_prompt,
                include_error_details=self.include_error_details,
            )

            for lvl in levels:
                reports, stats = await summarizer.generate_all(
                    level=lvl, skip_existing=self.skip_existing
                )
                logger.info(
                    "level %s: %s succeeded, %s skipped, %s failed (%.1fs)",
                    lvl, stats.succeeded, stats.skipped, stats.failed,
                    stats.elapsed_seconds,
                )

                if stats.failed:
                    maintenance_warnings.append({
                        "code": "community_report_generation_failed",
                        "level": lvl,
                        "failed": stats.failed,
                        "errors": stats.errors,
                    })
                total_reports += len(reports)
                level_stats.append({
                    "level": lvl,
                    "attempted": stats.attempted,
                    "skipped": stats.skipped,
                    "succeeded": stats.succeeded,
                    "failed": stats.failed,
                    "elapsed_seconds": round(stats.elapsed_seconds, 2),
                    "llm_calls": stats.llm_calls,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                })

            # Step 6: Embed community reports
            embedded_count = 0
            if self.embedder and self.embed_community_reports:
                from recon_graphrag.embeddings.community_reports import (
                    CommunityReportEmbedder,
                )

                report_embedder = CommunityReportEmbedder(
                    graph_store=self.graph_store,
                    embedder=self.embedder,
                    graph_name=self.graph_name,
                )
                embedded_count = await report_embedder.embed_reports()

            result = {
                "status": "degraded" if maintenance_warnings else "completed",
                "maintenance_warnings": maintenance_warnings,
                "communities": len(community_stats),
                "reports": total_reports,
                "levels": levels,
                "level_stats": level_stats,
                "embedded_reports": embedded_count,
            }
            token_usage = usage_snapshot(self.llm)
            if token_usage is not None:
                result["token_usage"] = token_usage
            return result
