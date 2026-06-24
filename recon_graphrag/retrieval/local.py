"""Local search: entity-centric subgraph traversal.

Finds relevant entities via vector similarity, traverses their relationships
to connected entities and source chunks, then uses LLM to generate an answer
from the gathered context.

This is the Microsoft GraphRAG "local search" pattern — specific, detailed
answers anchored to particular entities and their local graph neighborhood.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.citations import resolve_chunk_citations
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever, HybridRanker


DEFAULT_ANSWER_PROMPT = """Based on the findings below, answer the query.

Query: {query}

Findings and connections:
{context}

Provide a detailed, specific answer based on the findings above.
If the context doesn't contain enough information, say so.
Cite specific entities and findings when possible.

Answer:"""


class LocalSearchRetriever(BaseRetriever):
    """Local search: find entities matching query, traverse subgraph, generate answer."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        retrieval_query: Optional[str] = None,
        answer_prompt: Optional[str] = None,
        vector_index_name: str = "entity-embeddings",
        fulltext_index_name: str = "entity-names",
        graph_name: str = "entity-graph",
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query
        self.answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self.graph_name = graph_name
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridEntityRetriever:
        return HybridEntityRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            context_mode="local",
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        query_vector: list[float] | None = None,
        effective_search_ratio: int = 1,
        query_params: dict | None = None,
        ranker: HybridRanker | str = "naive",
        alpha: float | None = None,
    ) -> SearchResult:
        """Run local search: vector search on entities → subgraph traversal → LLM answer."""
        retriever_result = await self._retriever.search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
            ranker=ranker,
            alpha=alpha,
        )
        context = self._format_context(retriever_result)
        answer = await self._generate_answer(query, context)
        citations = self._resolve_citations(retriever_result)
        return SearchResult(
            query=query,
            mode="local",
            answer=answer,
            context=context,
            citations=citations,
        )

    def _format_context(self, retriever_result) -> str:
        """Format retriever results into a context string for the LLM."""
        parts = []
        for item in retriever_result.items:
            content = item.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, dict):
                title = content.get("title", "Unknown")
                rels = content.get("relationships", [])
                sources = content.get("source_text", [])
                section = f"Finding: {title}"
                if rels:
                    section += "\nConnections:\n  " + "\n  ".join(rels)
                if sources:
                    section += "\nEvidence:\n  " + "\n  ".join(sources[:3])
                parts.append(section)
        return "\n\n---\n\n".join(parts)

    def _resolve_citations(self, retriever_result):
        chunk_ids = _source_chunk_ids_from_result(retriever_result)
        try:
            return resolve_chunk_citations(
                self.graph_store,
                self.graph_name,
                chunk_ids,
            )
        except Exception:
            return []

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context using LLM."""
        prompt = self.answer_prompt.format(query=query, context=context)
        response = await self.llm.ainvoke(prompt)
        return response.content


def _source_chunk_ids_from_result(retriever_result) -> list[str]:
    seen: set[str] = set()
    chunk_ids: list[str] = []
    for item in retriever_result.items:
        content = item.content
        if not isinstance(content, dict):
            continue
        for chunk_id in content.get("source_chunk_ids", []):
            chunk_id = str(chunk_id).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids
