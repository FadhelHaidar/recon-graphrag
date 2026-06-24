"""Global search: community-summaries-based map-reduce retrieval.

Supports two strategies:
  - semantic (default): vector search top-k communities → sequential map → reduce
  - paper: all-report, token-batched, scored, parallel map → reduce

This enables answering broad, corpus-wide questions by aggregating insights
across hierarchical community levels.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.community_levels import (
    CommunityLevelSelector,
    resolve_community_level,
)


DEFAULT_MAP_PROMPT = """Based on this report segment, answer the question.

Question: {query}

Report Segment:
{summary}

Provide a partial answer focusing on what this segment contributes.
Be specific and cite details from the segment.

Partial Answer:"""

DEFAULT_REDUCE_PROMPT = """Synthesize the following perspectives into a comprehensive answer.

Question: {query}

Perspectives from different report segments:
{partial_answers}

Combine these perspectives into a coherent final answer.
Remove redundancy, resolve contradictions, and organize the key insights.

Final Answer:"""


class GlobalSearchRetriever(BaseRetriever):
    """Global search: find relevant communities → map-reduce over summaries."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        map_prompt: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
        vector_index_name: str = "community-embeddings",
        graph_name: str = "entity-graph",
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.map_prompt = map_prompt or DEFAULT_MAP_PROMPT
        self.reduce_prompt = reduce_prompt or DEFAULT_REDUCE_PROMPT
        self.vector_index_name = vector_index_name
        self.graph_name = graph_name

    async def search(
        self,
        query: str,
        top_k: int = 5,
        level: CommunityLevelSelector = None,
        community_level: CommunityLevelSelector = None,
        strategy: str = "semantic",
        random_seed: int | None = 42,
    ) -> SearchResult:
        """Run global search over community summaries.

        Args:
            query: User question.
            top_k: Number of communities for semantic strategy.
            level/community_level: Community hierarchy level.
            strategy: "semantic" (vector top-k) or "paper" (all-report scored).
            random_seed: Seed for paper strategy shuffling.
        """
        if strategy == "paper":
            return await self._paper_search(query, community_level or level, random_seed)
        elif strategy != "semantic":
            raise ValueError(
                f"Unknown global strategy: '{strategy}'. Use 'semantic' or 'paper'."
            )

        # Semantic strategy (current behavior)
        selected_level = community_level if community_level is not None else level
        resolved_level = resolve_community_level(
            self.graph_store,
            self.graph_name,
            selected_level,
        )
        query_vector = await self.embedder.async_embed_query(query)
        communities = self.graph_store.search_communities(
            index_name=self.vector_index_name,
            query_vector=query_vector,
            graph_name=self.graph_name,
            top_k=top_k,
            level=resolved_level,
        )
        if not communities:
            return SearchResult(
                query=query, mode="global",
                answer="No relevant communities found.", context="",
            )

        context = self._format_communities(communities)
        partial_answers = await self._map_phase(query, communities)
        answer = await self._reduce_phase(query, partial_answers)
        return SearchResult(
            query=query, mode="global", answer=answer, context=context,
            metadata={
                "strategy": "semantic",
                "top_k": top_k,
                "communities_used": len(communities),
                "selected_level": resolved_level,
            },
        )

    async def _paper_search(
        self,
        query: str,
        selected_level: CommunityLevelSelector,
        random_seed: int | None,
    ) -> SearchResult:
        """Run paper-style global search.

        Uses paper-specific prompts (not semantic map/reduce prompts).
        """
        from recon_graphrag.retrieval.global_paper import PaperGlobalSearch

        resolved_level = resolve_community_level(
            self.graph_store,
            self.graph_name,
            selected_level,
        )
        paper = PaperGlobalSearch(
            graph_store=self.graph_store,
            llm=self.llm,
            graph_name=self.graph_name,
        )
        return await paper.search(
            query=query,
            level=resolved_level,
            random_seed=random_seed,
        )

    async def _map_phase(
        self, query: str, communities: list[dict]
    ) -> list[str]:
        """Generate a partial answer from each community summary."""
        partial_answers = []
        for comm in communities:
            prompt = self.map_prompt.format(query=query, summary=comm["summary"])
            response = await self.llm.ainvoke(prompt)
            partial_answers.append(response.content)
        return partial_answers

    async def _reduce_phase(
        self, query: str, partial_answers: list[str]
    ) -> str:
        """Combine partial answers into a final coherent answer."""
        numbered = "\n\n".join(
            f"[Perspective {i+1}]: {ans}"
            for i, ans in enumerate(partial_answers)
        )
        prompt = self.reduce_prompt.format(query=query, partial_answers=numbered)
        response = await self.llm.ainvoke(prompt)
        return response.content

    @staticmethod
    def _format_communities(communities: list[dict]) -> str:
        parts = []
        for comm in communities:
            parts.append(
                f"Report Segment {comm['id']} (level {comm['level']}):\n"
                f"{comm['summary']}"
            )
        return "\n\n---\n\n".join(parts)
