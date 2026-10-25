"""Local search: entity-centric subgraph traversal.

Finds relevant entities via vector similarity, traverses their relationships
to connected entities and source chunks, then uses LLM to generate an answer
from the gathered context.

This is the Microsoft GraphRAG "local search" pattern — specific, detailed
answers anchored to particular entities and their local graph neighborhood.
"""

from __future__ import annotations

from typing import Any, Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.artifacts import Citation
from recon_graphrag.models.types import SearchResult
from recon_graphrag.observability import (
    run_scope,
    token_stage,
    track_usage,
    usage_snapshot,
)
from recon_graphrag.retrieval.citations import (
    format_citation_metadata,
    resolve_chunk_citations,
)
from recon_graphrag.retrieval.community_levels import CommunityLevelSelector
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever, HybridRanker
from recon_graphrag.retrieval.mixed_context import MixedContextBuilder

DEFAULT_ANSWER_PROMPT = """Based on the findings below, answer the query.

Query: {query}

Findings and connections:
{context}

Provide a detailed, specific answer based on the findings above.
If the context doesn't contain enough information, say so.
Cite specific entities and findings when possible.

Answer:"""


class LocalSearchRetriever:
    """Local search: find entities matching query, traverse subgraph, generate answer."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        retrieval_query: Optional[str] = None,  # noqa: UP045 - public signature compatibility
        answer_prompt: Optional[str] = None,  # noqa: UP045 - public signature compatibility
        vector_index_name: str = "entity-embeddings",
        fulltext_index_name: str = "entity-names",
        graph_name: str = "entity-graph",
        use_mixed_context: bool = True,
        top_k_relationships: int = 10,
        track_token_usage: bool = True,
    ):
        self.graph_store = graph_store
        self.llm = track_usage(llm) if track_token_usage else llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query
        self.answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self.graph_name = graph_name
        self.use_mixed_context = use_mixed_context
        self.top_k_relationships = top_k_relationships
        self._retriever = self._build_retriever()
        self._mixed_context_builder: MixedContextBuilder | None = None

    def _build_retriever(self) -> HybridEntityRetriever:
        return HybridEntityRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            graph_name=self.graph_name,
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
        synthesize_citation_metadata: bool = False,
        synthesis_metadata_keys: list[str] | None = None,
        synthesize_response: bool = True,
        community_level: CommunityLevelSelector = "coarsest",
        token_budget: int = 12000,
    ) -> SearchResult:
        """Run local search: vector search on entities → subgraph traversal → LLM answer.

        Args:
            query: User question.
            top_k: Number of top entities to retrieve.
            query_vector: Optional precomputed query vector.
            effective_search_ratio: Over-fetch multiplier before post-filtering.
            query_params: Optional dict forwarded to the underlying hybrid entity retriever.
            ranker: Hybrid ranker: "naive" or "linear".
            alpha: Required for the "linear" ranker.
            synthesize_citation_metadata: Include citation metadata in LLM context.
            synthesis_metadata_keys: Keys to include in citation metadata.
            synthesize_response: If False, skip LLM answer generation and return
                the retrieved context and citations without a final answer.
                Useful when an outer agent wants to synthesize the response itself.
            community_level: Community level for mixed context reports.
            token_budget: Total token budget for mixed context (only used
                when use_mixed_context=True).
        """
        with run_scope():
            retriever_result = await self._retriever.search(
                query_text=query,
                query_vector=query_vector,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
                ranker=ranker,
                alpha=alpha,
            )

            if self.use_mixed_context:
                return await self._search_mixed(
                    query=query,
                    retriever_result=retriever_result,
                    synthesize_response=synthesize_response,
                    community_level=community_level,
                    token_budget=token_budget,
                    synthesize_citation_metadata=synthesize_citation_metadata,
                    synthesis_metadata_keys=synthesis_metadata_keys,
                )

            citations, citation_warnings = self._resolve_citations(retriever_result)
            context = self._format_context(
                retriever_result,
                citations=citations if synthesize_citation_metadata else None,
                citation_metadata_keys=synthesis_metadata_keys,
            )
            if not synthesize_response:
                token_usage = usage_snapshot(self.llm)
                metadata: dict[str, Any] = {
                    "synthesize_response": False,
                    "response_synthesis_skipped": True,
                }
                if token_usage is not None:
                    metadata["token_usage"] = token_usage
                return SearchResult(
                    query=query,
                    mode="local",
                    answer="",
                    context=context,
                    citations=citations,
                    metadata={**metadata, **({"citation_warnings": citation_warnings} if citation_warnings else {})},
                )
            answer = await self._generate_answer(query, context)
            token_usage = usage_snapshot(self.llm)
            metadata = {}
            if token_usage is not None:
                metadata["token_usage"] = token_usage
            return SearchResult(
                query=query,
                mode="local",
                answer=answer,
                context=context,
                citations=citations,
                metadata={**metadata, **({"citation_warnings": citation_warnings} if citation_warnings else {})},
            )

    async def _search_mixed(
        self,
        query: str,
        retriever_result,
        synthesize_response: bool,
        community_level: CommunityLevelSelector,
        token_budget: int,
        synthesize_citation_metadata: bool,
        synthesis_metadata_keys: list[str] | None,
    ) -> SearchResult:
        """Run local search with mixed-context builder."""
        if self._mixed_context_builder is None:
            self._mixed_context_builder = MixedContextBuilder(
                graph_store=self.graph_store,
                embedder=self.embedder,
                graph_name=self.graph_name,
            )

        entity_matches = []
        entity_context_rows = []
        for item in retriever_result.items:
            content = item.content
            if not isinstance(content, dict):
                continue
            entity_context_rows.append(content)
            score = content.get("score", 0.0)
            entity_id = content.get("entity_id") or content.get("title", "")
            entity_matches.append({"id": entity_id, "score": score})

        mixed = self._mixed_context_builder.build_context(
            entity_matches=entity_matches,
            entity_context_rows=entity_context_rows,
            token_budget=token_budget,
            community_level=community_level,
            synthesize_citation_metadata=synthesize_citation_metadata,
            synthesis_metadata_keys=synthesis_metadata_keys,
        )

        if not synthesize_response:
            token_usage = usage_snapshot(self.llm)
            metadata = {
                "synthesize_response": False,
                "response_synthesis_skipped": True,
                "mixed_context": True,
                "used_tokens": mixed.used_tokens,
                "max_tokens": mixed.max_tokens,
            }
            if token_usage is not None:
                metadata["token_usage"] = token_usage
            return SearchResult(
                query=query,
                mode="local",
                answer="",
                context=mixed.context,
                citations=mixed.citations,
                metadata={**metadata, **({"citation_warnings": mixed.citation_warnings} if mixed.citation_warnings else {})},
            )

        answer = await self._generate_answer(query, mixed.context)
        token_usage = usage_snapshot(self.llm)
        metadata = {
            "mixed_context": True,
            "used_tokens": mixed.used_tokens,
            "max_tokens": mixed.max_tokens,
        }
        if token_usage is not None:
            metadata["token_usage"] = token_usage
        return SearchResult(
            query=query,
            mode="local",
            answer=answer,
            context=mixed.context,
            citations=mixed.citations,
            metadata={**metadata, **({"citation_warnings": mixed.citation_warnings} if mixed.citation_warnings else {})},
        )

    def _format_context(
        self,
        retriever_result,
        *,
        citations: list[Citation] | None = None,
        citation_metadata_keys: list[str] | None = None,
    ) -> str:
        """Format retriever results into a context string for the LLM."""
        return _format_entity_context(
            retriever_result,
            citations=citations,
            citation_metadata_keys=citation_metadata_keys,
            top_k_relationships=self.top_k_relationships,
        )

    def _resolve_citations(self, retriever_result):
        chunk_ids = _source_chunk_ids_from_result(retriever_result)
        try:
            citations = resolve_chunk_citations(
                self.graph_store,
                self.graph_name,
                chunk_ids,
            )
            return citations, []
        except Exception:
            return [], [{"code": "citation_resolution_failed", "stage": "chunk"}]

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context using LLM."""
        prompt = self.answer_prompt.format(query=query, context=context)
        with token_stage("retrieval.local"):
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


def _format_entity_context(
    retriever_result,
    *,
    citations: list[Citation] | None = None,
    citation_metadata_keys: list[str] | None = None,
    drift: bool = False,
    top_k_relationships: int | None = None,
) -> str:
    citation_lines = format_citation_metadata(citations or [], citation_metadata_keys)
    heading_prefix = "  " if drift else ""
    indent = "    " if drift else "  "
    source_limit = 2 if drift else 3
    item_separator = "\n\n" if drift else "\n\n---\n\n"

    def block(heading: str, values: list[str]) -> str:
        return f"\n{heading_prefix}{heading}:\n" + indent + f"\n{indent}".join(values)

    parts = []
    for item in retriever_result.items:
        content = item.content
        if not isinstance(content, dict):
            continue

        section = f"Finding: {content.get('title', 'Unknown')}"
        relationships = content.get("relationships", [])
        if top_k_relationships is not None and len(relationships) > top_k_relationships:
            # relationship_records is parallel to relationships; rank by weight
            # so truncation keeps the strongest connections, not collect() order.
            records = content.get("relationship_records", [])
            if len(records) == len(relationships):
                order = sorted(
                    range(len(records)),
                    key=lambda i: -float(records[i].get("weight", 1.0) or 1.0),
                )
                relationships = [relationships[i] for i in order]
            relationships = relationships[:top_k_relationships]
        if relationships:
            section += block("Connections", relationships)
        sources = content.get("source_text", [])
        if sources:
            section += block("Evidence", sources[:source_limit])
        parts.append(section)
    if parts and citation_lines:
        parts.append(
            block("Citation metadata", citation_lines).lstrip("\n")
        )
    return item_separator.join(parts)
