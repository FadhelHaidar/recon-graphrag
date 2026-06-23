"""Community pipeline: detect → summarize → embed (steps 4-6).

Convenience wrapper that chains the three community steps into a single
build() call. Typically run on a schedule (e.g. weekly) after new entities
have been ingested.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.communities.embeddings import CommunityEmbedder
from recon_graphrag.communities.reports import ReportRubric
from recon_graphrag.communities.summarization import BuildStats, CommunitySummarizer
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM


class CommunityPipeline:
    """Run the full community pipeline: detect → summarize → embed."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        relationship_types: Optional[list[str]] = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 1e-4,
        graph_name: str = "entity-graph",
        relationship_weight_property: Optional[str] = None,
        random_seed: Optional[int] = 42,
        summary_prompt: Optional[str] = None,
        use_reports: bool = False,
        report_rubric: ReportRubric | None = None,
        summarize_concurrency: int = 1,
        skip_existing: bool = False,
    ):
        """Initialize the community pipeline.

        Args:
            graph_store: Store that provides community detection and persistence.
            llm: LLM used to summarize detected communities.
            embedder: Embedder used for community summary embeddings.
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
            summary_prompt: Optional custom prompt for community summaries.
            use_reports: Generate structured reports instead of plain summaries.
            report_rubric: Rating rubric for structured reports.
            summarize_concurrency: Max concurrent LLM calls per level.
            skip_existing: Skip communities that already have a summary.
        """
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.relationship_types = relationship_types
        self.max_levels = max_levels
        self.gamma = gamma
        self.theta = theta
        self.tolerance = tolerance
        self.graph_name = graph_name
        self.relationship_weight_property = relationship_weight_property
        self.random_seed = random_seed
        self.summary_prompt = summary_prompt
        self.use_reports = use_reports
        self.report_rubric = report_rubric
        self.summarize_concurrency = summarize_concurrency
        self.skip_existing = skip_existing

    async def build(self, level: Optional[int] = None) -> dict:
        """Run steps 4-6: detect communities, summarize, and embed.

        Processes levels bottom-up. Within each level, runs up to
        ``summarize_concurrency`` summaries in parallel.

        Args:
            level: Highest community hierarchy level to summarize and embed.
                If None, processes all detected levels. If provided, lower
                levels are also processed first so parent summaries can use
                child summaries.

        Returns:
            Dict with stats from each step, including per-level build stats.
        """
        print("Step 4: Detecting communities...")
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
        print(f"  Found {len(community_stats)} communities")

        detected_levels = sorted({s["level"] for s in community_stats})
        levels = (
            [lvl for lvl in detected_levels if lvl <= level]
            if level is not None
            else detected_levels
        )
        total_summaries = 0
        level_stats: list[dict] = []

        summarizer = CommunitySummarizer(
            self.graph_store,
            self.llm,
            prompt_template=self.summary_prompt,
            graph_name=self.graph_name,
            use_reports=self.use_reports,
            report_rubric=self.report_rubric,
            concurrency=self.summarize_concurrency,
        )
        community_embedder = CommunityEmbedder(
            self.graph_store,
            self.embedder,
            graph_name=self.graph_name,
        )

        for lvl in levels:
            print(f"Step 5: Summarizing communities (level {lvl})...")
            summaries, stats = await summarizer.summarize_all(
                level=lvl, skip_existing=self.skip_existing
            )
            print(
                f"  Level {lvl}: {stats.succeeded} succeeded, "
                f"{stats.skipped} skipped, {stats.failed} failed "
                f"({stats.elapsed_seconds:.1f}s)"
            )

            print(f"Step 6: Embedding community summaries (level {lvl})...")
            await community_embedder.embed_communities(level=lvl)

            total_summaries += len(summaries)
            level_stats.append({
                "level": lvl,
                "attempted": stats.attempted,
                "skipped": stats.skipped,
                "succeeded": stats.succeeded,
                "failed": stats.failed,
                "elapsed_seconds": round(stats.elapsed_seconds, 2),
            })

        return {
            "communities": len(community_stats),
            "summaries": total_summaries,
            "levels": levels,
            "level_stats": level_stats,
        }
