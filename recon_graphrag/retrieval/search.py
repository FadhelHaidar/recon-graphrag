"""Unified search orchestrator for all retrieval modes."""

from __future__ import annotations

from recon_graphrag.communities.detection import DEFAULT_GRAPH_NAME
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.models.types import SearchResult
from recon_graphrag.retrieval.drift import DriftSearchRetriever
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever
from recon_graphrag.retrieval.local import LocalSearchRetriever


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
        llm: BaseLLM,
        embedder: BaseEmbedder,
        graph_name: str = DEFAULT_GRAPH_NAME,
    ):
        self.local = LocalSearchRetriever(graph_store, llm, embedder)
        self.global_ = GlobalSearchRetriever(
            graph_store, llm, embedder, graph_name=graph_name
        )
        self.drift = DriftSearchRetriever(
            graph_store, llm, embedder, graph_name=graph_name
        )

    async def search(
        self, query: str, mode: str = "local", **kwargs
    ) -> SearchResult:
        """Search with a specific mode.

        Args:
            query: The search query.
            mode: One of "local", "global", "drift".
            **kwargs: Additional arguments passed to the specific retriever:
                - local: top_k (int)
                - global: top_k (int), level/community_level
                  (int | "all" | "finest" | "coarsest")
                - drift: top_k (int), community_top_k (int), community_level
                  (int | "all" | "finest" | "coarsest")
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
