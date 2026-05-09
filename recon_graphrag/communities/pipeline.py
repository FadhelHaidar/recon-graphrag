"""Community pipeline: detect → summarize → embed (steps 4-6).

Convenience wrapper that chains the three community steps into a single
build() call. Typically run on a schedule (e.g. weekly) after new entities
have been ingested.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder

from recon_graphrag.graph.base import GraphStore
from recon_graphrag.communities.detection import CommunityDetector
from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.communities.embeddings import CommunityEmbedder


class CommunityPipeline:
    """Run the full community pipeline: detect → summarize → embed."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        embedder: Embedder,
        relationship_types: Optional[list[str]] = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        summary_prompt: Optional[str] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.relationship_types = relationship_types or []
        self.max_levels = max_levels
        self.gamma = gamma
        self.theta = theta
        self.summary_prompt = summary_prompt

    async def build(self, level: Optional[int] = None) -> dict:
        """Run steps 4-6: detect communities, summarize, and embed.

        Args:
            level: Specific community hierarchy level to summarize and embed.
                If None, processes all levels.

        Returns:
            Dict with stats from each step.
        """
        # Step 4: Detect communities
        print("Step 4: Detecting communities...")
        detector = CommunityDetector(
            self.graph_store,
            relationship_types=self.relationship_types,
            max_levels=self.max_levels,
            gamma=self.gamma,
            theta=self.theta,
        )
        community_stats = detector.detect()
        print(f"  Found {len(community_stats)} communities")

        levels = [level] if level is not None else sorted({s["level"] for s in community_stats})
        total_summaries = 0

        summarizer = CommunitySummarizer(
            self.graph_store,
            self.llm,
            prompt_template=self.summary_prompt,
        )
        embedder = CommunityEmbedder(self.graph_store, self.embedder)

        for lvl in levels:
            # Step 5: Summarize communities
            print(f"Step 5: Summarizing communities (level {lvl})...")
            summaries = await summarizer.summarize_all(level=lvl)
            print(f"  Summarized {len(summaries)} communities at level {lvl}")

            # Step 6: Embed community summaries
            print(f"Step 6: Embedding community summaries (level {lvl})...")
            await embedder.embed_communities(level=lvl)

            total_summaries += len(summaries)

        return {
            "communities": len(community_stats),
            "summaries": total_summaries,
        }
