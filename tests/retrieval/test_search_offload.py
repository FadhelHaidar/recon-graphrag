"""Synchronous graph-store reads in async retrieval must not stall the loop."""

from typing import Any, cast

from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.graph_evidence import build_evidence
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.utils.tokens import TiktokenTokenCounter
from tests.pipelines.test_async_offload import BlockingCalls, with_heartbeat
from tests.retrieval.test_drift import (
    FakeEmbedder,
    MixedFakeGraphStore,
    _follow_up_llm,
)


class BlockingStore(MixedFakeGraphStore, BlockingCalls):
    """Every store call the retrieval path makes blocks its calling thread."""

    def __init__(self):
        MixedFakeGraphStore.__init__(self)
        BlockingCalls.__init__(self)

    def vector_search(self, *args: Any, **kwargs: Any):
        self.block("vector_search")
        return super().vector_search(*args, **kwargs)

    def keyword_search(self, *args: Any, **kwargs: Any):
        self.block("keyword_search")
        return super().keyword_search(*args, **kwargs)

    def fetch_entity_context(self, *args: Any, **kwargs: Any):
        self.block("fetch_entity_context")
        return super().fetch_entity_context(*args, **kwargs)

    def vector_search_community_reports(self, *args: Any, **kwargs: Any):
        self.block("report_search")
        return super().vector_search_community_reports(*args, **kwargs)

    def execute_query(self, *args: Any, **kwargs: Any):
        self.block("execute_query")
        return super().execute_query(*args, **kwargs)

    def get_claims_for_entities(self, *args: Any, **kwargs: Any):
        self.block("claims")
        return super().get_claims_for_entities(*args, **kwargs)

    def resolve_chunk_citations(self, *args: Any, **kwargs: Any):
        self.block("chunk_citations")
        return super().resolve_chunk_citations(*args, **kwargs)

    def fetch_community_member_graph(self, *args: Any, **kwargs: Any):
        self.block("community_expansion")
        return [
            {
                "members": [{"id": "a", "name": "Alice", "labels": ["Person"]}],
                "edges": [],
                "member_count": 1,
                "edge_count": 0,
            }
        ]


async def test_drift_store_reads_run_off_the_loop():
    # Loading the BPE vocabulary is a one-time ~200ms cost, not a store read.
    TiktokenTokenCounter().count("warm")
    store = BlockingStore()
    retriever = DriftSearchRetriever(
        cast(Any, store),
        cast(Any, _follow_up_llm()),
        cast(Any, FakeEmbedder()),
        config=DriftSearchConfig(use_hyde=False, max_depth=1),
    )

    result = await with_heartbeat(retriever.search("Who directed Inception?", top_k=2))

    store.assert_offloaded(
        "execute_query",
        "report_search",
        "vector_search",
        "keyword_search",
        "fetch_entity_context",
        "claims",
    )
    assert result.mode == "drift"
    assert result.answer == "Alice directed Inception."
    assert result.metadata["drift_trace"]["actions"][-1]["status"] == "completed"
    assert {"primer", "traversal", "reduce", "total"} <= set(
        result.metadata["drift_trace"]["phase_ms"]
    )


async def test_hybrid_store_reads_run_off_the_loop():
    store = BlockingStore()
    retriever = HybridEntityRetriever(
        cast(Any, store),
        cast(Any, FakeEmbedder()),
        retrieval_query=None,
        vector_index_name="entity-vector",
        fulltext_index_name="entity-fulltext",
        context_mode="drift",
    )

    result = await with_heartbeat(retriever.search("Who directed Inception?"))

    store.assert_offloaded("vector_search", "keyword_search", "fetch_entity_context")
    assert [item.content["score"] for item in result.items] == [1.0]


async def test_community_expansion_reads_run_off_the_loop():
    store = BlockingStore()

    evidence = await with_heartbeat(
        build_evidence(
            graph_name="entity-graph",
            mode="drift",
            graph_store=cast(Any, store),
            reports=[{"community_id": "c0", "level": 0}],
        )
    )

    store.assert_offloaded("community_expansion")
    assert [node["id"] for node in evidence["nodes"]] == ["a"]
