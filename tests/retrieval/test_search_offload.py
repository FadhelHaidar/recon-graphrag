"""Synchronous graph-store reads in async retrieval must not stall the loop."""

import json
from typing import Any, cast

from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.graph_evidence import build_evidence
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever
from recon_graphrag.utils.tokens import TiktokenTokenCounter
from tests.pipelines.test_async_offload import BlockingCalls, with_heartbeat
from tests.retrieval.test_drift import (
    FakeEmbedder,
    MixedFakeGraphStore,
    _follow_up_llm,
)
from tests.retrieval.test_dynamic_selection import (
    FakeGraphStore as DynamicGraphStore,
)
from tests.retrieval.test_dynamic_selection import FakeLLM as DynamicLLM
from tests.retrieval.test_dynamic_selection import report as dynamic_report


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
        config=DriftSearchConfig(use_hyde=False, max_rounds=1),
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


class LocalAnswerLLM:
    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="Alice directed Inception.")


async def test_local_citation_reads_run_off_the_loop():
    store = BlockingStore()
    retriever = LocalSearchRetriever(
        cast(Any, store),
        cast(Any, LocalAnswerLLM()),
        cast(Any, FakeEmbedder()),
        use_mixed_context=False,
    )

    result = await with_heartbeat(
        retriever.search("Who directed Inception?", top_k=2, synthesize_response=False)
    )

    store.assert_offloaded(
        "vector_search",
        "keyword_search",
        "fetch_entity_context",
        "chunk_citations",
    )
    assert [citation.chunk_id for citation in result.citations] == ["chunk:1"]


async def test_local_mixed_context_reads_run_off_the_loop():
    store = BlockingStore()
    retriever = LocalSearchRetriever(
        cast(Any, store),
        cast(Any, LocalAnswerLLM()),
        cast(Any, FakeEmbedder()),
        use_mixed_context=True,
    )

    result = await with_heartbeat(
        retriever.search("Who directed Inception?", top_k=2, synthesize_response=False)
    )

    store.assert_offloaded("execute_query", "claims", "chunk_citations")
    assert result.context


class BlockingGlobalStore(DynamicGraphStore, BlockingCalls):
    """Dynamic-selection store whose reads block their calling thread."""

    def __init__(self, reports=(), links=None):
        DynamicGraphStore.__init__(self, reports, links)
        BlockingCalls.__init__(self)

    def execute_query(self, query, params=None):
        self.block("global_query")
        if "FROM_CHUNK" in query:
            return [{"chunk_id": "chunk:1"}]
        return super().execute_query(query, params)

    def get_child_community_reports(
        self, graph_name, community_id, level, child_level
    ):
        self.block("child_reports")
        return super().get_child_community_reports(
            graph_name, community_id, level, child_level
        )

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.block("chunk_citations")
        return [
            {
                "chunk_id": "chunk:1",
                "document_id": "doc:1",
                "document_name": "Doc 1",
                "excerpt": "Alice directed Inception in 2010.",
            }
        ]


def _referenced_report(report_id, level=0):
    return dynamic_report(
        report_id,
        level,
        report_json=json.dumps(
            {
                "findings": [
                    {
                        "references": [
                            {
                                "target_id": "person:alice",
                                "target_type": "entity",
                            }
                        ]
                    }
                ]
            }
        ),
    )


async def test_global_static_report_and_citation_reads_run_off_the_loop():
    # Loading the BPE vocabulary is a one-time ~200ms cost, not a store read.
    TiktokenTokenCounter().count("warm")
    store = BlockingGlobalStore([_referenced_report("root")])
    retriever = GlobalSearchRetriever(cast(Any, store), cast(Any, DynamicLLM()))

    result = await with_heartbeat(retriever.search("q", community_level="finest"))

    store.assert_offloaded("global_query", "chunk_citations")
    assert result.metadata["selection_mode"] == "static"
    assert [citation.chunk_id for citation in result.citations] == ["chunk:1"]


async def test_global_dynamic_report_child_and_citation_reads_run_off_the_loop():
    # Loading the BPE vocabulary is a one-time ~200ms cost, not a store read.
    TiktokenTokenCounter().count("warm")
    root = _referenced_report("root")
    child = _referenced_report("child", 1)
    store = BlockingGlobalStore([root, child], {("root", 0): [child]})
    retriever = GlobalSearchRetriever(cast(Any, store), cast(Any, DynamicLLM()))

    result = await with_heartbeat(retriever.search("q"))

    store.assert_offloaded("global_query", "child_reports", "chunk_citations")
    assert result.metadata["selection_mode"] == "dynamic"
    assert result.metadata["selected_report_keys"] == [
        {"community_id": "child", "level": 1}
    ]
