"""Local search: entity-centric subgraph traversal.

Finds relevant entities via vector similarity, traverses their relationships
to connected entities and source chunks, then uses LLM to generate an answer
from the gathered context.

This is the Microsoft GraphRAG "local search" pattern — specific, detailed
answers anchored to particular entities and their local graph neighborhood.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.retrievers import HybridCypherRetriever

from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.base import BaseRetriever


DEFAULT_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
WITH node, score, collect(DISTINCT {
    entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
    rel: type(r),
    neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, '')
}) AS connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, connections, collect(DISTINCT chunk.text) AS source_texts
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           c.entity + ' -[' + c.rel + ']-> ' + c.neighbor] AS relationships,
       source_texts AS source_text,
       score
"""

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
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query or DEFAULT_RETRIEVAL_QUERY
        self.answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridCypherRetriever:
        neo4j_database = getattr(self.graph_store, "_database", None)
        return HybridCypherRetriever(
            driver=self.graph_store.driver,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            embedder=self.embedder,
            neo4j_database=neo4j_database,
        )

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Run local search: vector search on entities → subgraph traversal → LLM answer."""
        retriever_result = self._retriever.search(query_text=query, top_k=top_k)
        context = self._format_context(retriever_result)
        answer = await self._generate_answer(query, context)
        return SearchResult(query=query, mode="local", answer=answer, context=context)

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

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context using LLM."""
        prompt = self.answer_prompt.format(query=query, context=context)
        response = await self.llm.ainvoke(prompt)
        return response.content
