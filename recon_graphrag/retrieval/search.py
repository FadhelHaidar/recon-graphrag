"""Unified search orchestrator for all retrieval modes."""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder

from recon_graphrag.graph_store import GraphStore
from recon_graphrag.types import SearchResult
from recon_graphrag.retrieval.local import LocalSearchRetriever
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever
from recon_graphrag.retrieval.drift import DriftSearchRetriever


class GraphRAG:
    """Unified search interface providing local, global, and drift modes.

    - **local**: Entity-centric subgraph traversal. Best for specific questions
      about particular entities.
    - **global**: Community-summaries map-reduce. Best for broad, holistic
      questions and overviews.
    - **drift**: Hybrid entity + community. Combines local specificity with
      global context for questions that benefit from both perspectives.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        embedder: Embedder,
    ):
        self.local = LocalSearchRetriever(graph_store, llm, embedder)
        self.global_ = GlobalSearchRetriever(graph_store, llm, embedder)
        self.drift = DriftSearchRetriever(graph_store, llm, embedder)

    async def search(
        self, query: str, mode: str = "local", **kwargs
    ) -> SearchResult:
        """Search with a specific mode.

        Args:
            query: The search query.
            mode: One of "local", "global", "drift".
            **kwargs: Additional arguments passed to the specific retriever:
                - local: top_k (int)
                - global: top_k (int), level (int)
                - drift: top_k (int), community_top_k (int)
        """
        modes = {
            "local": self.local,
            "global": self.global_,
            "drift": self.drift,
        }
        if mode not in modes:
            raise ValueError(
                f"Unknown search mode: '{mode}'. Use one of: local, global, drift"
            )
        return await modes[mode].search(query, **kwargs)
