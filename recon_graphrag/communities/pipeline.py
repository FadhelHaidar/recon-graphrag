"""Community pipeline: detect → summarize → embed (steps 4-6).

Convenience wrapper that chains the three community steps into a single
build() call. Typically run on a schedule (e.g. weekly) after new entities
have been ingested.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.communities.detection import DEFAULT_GRAPH_NAME, CommunityDetector
from recon_graphrag.communities.embeddings import CommunityEmbedder
from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore
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
        graph_name: str = DEFAULT_GRAPH_NAME,
        relationship_weight_property: Optional[str] = None,
        random_seed: Optional[int] = 42,
        summary_prompt: Optional[str] = None,
    ):
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

    async def build(self, level: Optional[int] = None) -> dict:
        """Run steps 4-6: detect communities, summarize, and embed.

        Args:
            level: Highest community hierarchy level to summarize and embed.
                If None, processes all detected levels. If provided, lower
                levels are also processed first so parent summaries can use
                child summaries.

        Returns:
            Dict with stats from each step.
        """
        print("Step 4: Detecting communities...")
        detector = CommunityDetector(
            self.graph_store,
            relationship_types=self.relationship_types,
            max_levels=self.max_levels,
            gamma=self.gamma,
            theta=self.theta,
            tolerance=self.tolerance,
            graph_name=self.graph_name,
            relationship_weight_property=self.relationship_weight_property,
            random_seed=self.random_seed,
        )
        community_stats = detector.detect()
        print(f"  Found {len(community_stats)} communities")

        detected_levels = sorted({s["level"] for s in community_stats})
        levels = (
            [lvl for lvl in detected_levels if lvl <= level]
            if level is not None
            else detected_levels
        )
        total_summaries = 0

        summarizer = CommunitySummarizer(
            self.graph_store,
            self.llm,
            prompt_template=self.summary_prompt,
            graph_name=self.graph_name,
        )
        community_embedder = CommunityEmbedder(
            self.graph_store,
            self.embedder,
            graph_name=self.graph_name,
        )

        for lvl in levels:
            print(f"Step 5: Summarizing communities (level {lvl})...")
            summaries = await summarizer.summarize_all(level=lvl)
            print(f"  Summarized {len(summaries)} communities at level {lvl}")

            print(f"Step 6: Embedding community summaries (level {lvl})...")
            await community_embedder.embed_communities(level=lvl)

            total_summaries += len(summaries)

        return {
            "communities": len(community_stats),
            "summaries": total_summaries,
            "levels": levels,
        }
