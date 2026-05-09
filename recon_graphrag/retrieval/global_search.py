"""Global search: community-summaries-based map-reduce retrieval.

Finds relevant communities via vector search on summary embeddings, then uses
a map-reduce pattern (Microsoft GraphRAG style):
  - Map: Generate a partial answer from each community summary
  - Reduce: Synthesize partial answers into a final coherent answer

This enables answering broad, corpus-wide questions by aggregating insights
across hierarchical community levels.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder

from recon_graphrag.graph_store import GraphStore
from recon_graphrag.types import SearchResult


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


class GlobalSearchRetriever:
    """Global search: find relevant communities → map-reduce over summaries."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        embedder: Embedder,
        map_prompt: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
        vector_index_name: str = "community-embeddings",
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.map_prompt = map_prompt or DEFAULT_MAP_PROMPT
        self.reduce_prompt = reduce_prompt or DEFAULT_REDUCE_PROMPT
        self.vector_index_name = vector_index_name

    async def search(
        self, query: str, top_k: int = 5, level: Optional[int] = None
    ) -> SearchResult:
        """Run global search over community summaries.

        1. Embed query → vector search on Community.summary embeddings
        2. Map: LLM generates partial answer from each community summary
        3. Reduce: LLM synthesizes all partial answers into final answer
        """
        communities = await self._search_communities(query, top_k, level)
        if not communities:
            return SearchResult(
                query=query, mode="global",
                answer="No relevant communities found.", context="",
            )

        context = self._format_communities(communities)
        partial_answers = await self._map_phase(query, communities)
        answer = await self._reduce_phase(query, partial_answers)
        return SearchResult(query=query, mode="global", answer=answer, context=context)

    async def _search_communities(
        self, query: str, top_k: int, level: Optional[int]
    ) -> list[dict]:
        """Vector search on community summary embeddings."""
        query_vector = await self.embedder.async_embed_query(query)

        if level is not None:
            cypher = """
            CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
            YIELD node AS community, score
            WHERE community.level = $level AND community.summary IS NOT NULL
            RETURN community.id AS id, community.summary AS summary,
                   community.level AS level, score
            ORDER BY score DESC
            """
            params = {
                "index_name": self.vector_index_name,
                "k": top_k,
                "query_vector": query_vector,
                "level": level,
            }
        else:
            cypher = """
            CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
            YIELD node AS community, score
            WHERE community.summary IS NOT NULL
            RETURN community.id AS id, community.summary AS summary,
                   community.level AS level, score
            ORDER BY score DESC
            """
            params = {
                "index_name": self.vector_index_name,
                "k": top_k,
                "query_vector": query_vector,
            }

        return self.graph_store.execute_query(cypher, params)

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
                f"Report Segment {comm['id']}:\n{comm['summary']}"
            )
        return "\n\n---\n\n".join(parts)
