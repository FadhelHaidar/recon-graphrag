"""DRIFT search: hybrid entity-centric + community-context retrieval.

Combines the specificity of local search (entity subgraph traversal) with
the broader context of global search (community summaries). The pipeline:

1. Vector search on entities → find seed entities matching the query
2. Expand to communities via IN_COMMUNITY relationships
3. Fetch community summaries for high-level context
4. Fetch other entities in those communities for bridging connections
5. Build enriched context combining all signals
6. LLM generates answer from the combined context
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
from recon_graphrag.retrieval.citations import resolve_chunk_citations
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever, HybridRanker
from recon_graphrag.retrieval.local import _source_chunk_ids_from_result


DEFAULT_ANSWER_PROMPT = """You have access to detailed findings and broader context.

Query: {query}

=== Specific Findings ===
{entity_context}

=== Broader Context ===
{community_context}

=== Related Entities ===
{bridging_context}

Synthesize all the above information to answer the query. Use specific details
from the findings, high-level insights from the context, and relevant
connections from related entities.

Answer:"""


class DriftSearchRetriever(BaseRetriever):
    """DRIFT search: entity-centric with community expansion."""

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
        community_level: CommunityLevelSelector = 0,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query
        self.answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self.graph_name = graph_name
        self.community_level = community_level
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridEntityRetriever:
        return HybridEntityRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            context_mode="drift",
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        community_top_k: int = 3,
        community_level: CommunityLevelSelector = None,
        query_vector: list[float] | None = None,
        effective_search_ratio: int = 1,
        query_params: dict | None = None,
        ranker: HybridRanker | str = "naive",
        alpha: float | None = None,
    ) -> SearchResult:
        """Run DRIFT search.

        1. Vector search on entities → seed entities
        2. Extract community keys from seed entities
        3. Fetch community summaries
        4. Fetch other entities in those communities (bridging)
        5. Combine all context → LLM answer
        """
        retriever_result = await self._retriever.search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
            ranker=ranker,
            alpha=alpha,
        )

        entity_context = self._format_entity_context(retriever_result)
        target_selector = self.community_level if community_level is None else community_level
        target_level = resolve_community_level(
            self.graph_store,
            self.graph_name,
            target_selector,
        )
        community_keys = self._extract_community_keys(retriever_result, target_level)

        community_context = ""
        bridging_context = ""
        if community_keys:
            communities = self._fetch_community_summaries(community_keys, community_top_k)
            community_context = self._format_communities(communities)

            bridging_entities = self._fetch_community_entities(community_keys)
            bridging_context = self._format_bridging_entities(bridging_entities)

        answer = await self._generate_answer(
            query, entity_context, community_context, bridging_context
        )

        full_context = f"{entity_context}\n\n{community_context}\n\n{bridging_context}"
        citations = self._resolve_citations(retriever_result)
        return SearchResult(
            query=query,
            mode="drift",
            answer=answer,
            context=full_context,
            citations=citations,
        )

    def _format_entity_context(self, retriever_result) -> str:
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
                    section += "\n  Connections:\n    " + "\n    ".join(rels)
                if sources:
                    section += "\n  Evidence:\n    " + "\n    ".join(sources[:2])
                parts.append(section)
        return "\n\n".join(parts)

    def _extract_community_keys(
        self,
        retriever_result,
        community_level: Optional[int] = None,
    ) -> list[dict]:
        """Extract unique graph-scoped community keys from retriever results."""
        keys = set()
        for item in retriever_result.items:
            content = item.content
            if isinstance(content, dict):
                for comm in content.get("communities", []):
                    if not isinstance(comm, dict):
                        continue
                    cid = comm.get("id")
                    level = comm.get("level")
                    graph_name = comm.get("graph_name")
                    if cid is None or level is None or graph_name != self.graph_name:
                        continue
                    if community_level is not None and level != community_level:
                        continue
                    keys.add((str(cid), int(level)))

        return [{"id": cid, "level": level} for cid, level in keys]

    def _fetch_community_summaries(
        self, community_keys: list[dict], top_k: int
    ) -> list[dict]:
        return self.graph_store.get_community_summaries_by_keys(
            graph_name=self.graph_name,
            keys=community_keys,
            top_k=top_k,
        )

    def _fetch_community_entities(self, community_keys: list[dict]) -> list[dict]:
        """Fetch entities in specified communities."""
        return self.graph_store.get_community_entities_by_keys(
            graph_name=self.graph_name,
            keys=community_keys,
        )

    @staticmethod
    def _format_communities(communities: list[dict]) -> str:
        if not communities:
            return "No community context available."
        parts = []
        for comm in communities:
            parts.append(f"Segment {comm['id']} (level {comm['level']}):\n{comm['summary']}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_bridging_entities(entities: list[dict]) -> str:
        if not entities:
            return "No bridging entities found."
        parts = []
        for ent in entities:
            label = [lbl for lbl in ent.get("labels", []) if lbl != "__Entity__"]
            label = label[0] if label else "Entity"
            name = ent.get("name", "")
            rels = ent.get("rels", [])
            line = f"Related: [{label}] {name}"
            if rels:
                line += "\n    Connected to: " + "\n    Connected to: ".join(rels[:5])
            parts.append(line)
        return "\n".join(parts)

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

    async def _generate_answer(
        self,
        query: str,
        entity_context: str,
        community_context: str,
        bridging_context: str,
    ) -> str:
        prompt = self.answer_prompt.format(
            query=query,
            entity_context=entity_context,
            community_context=community_context,
            bridging_context=bridging_context,
        )
        response = await self.llm.ainvoke(prompt)
        return response.content
