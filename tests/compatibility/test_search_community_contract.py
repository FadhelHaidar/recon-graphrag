"""Public result shapes; deliberately exclude prompts and generated wording."""

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pytest

from recon_graphrag import (
    CommunityPipeline,
    DriftSearchConfig,
    DriftSearchRetriever,
    GlobalSearchRetriever,
    LLMResponse,
    LocalSearchRetriever,
    SearchResult,
)
from tests.communities.test_pipeline import FakeGraphStoreWithCommunities
from tests.compatibility.test_ingestion_contract import CharacterCounter
from tests.retrieval.test_drift import FakeEmbedder, FakeGraphStore, FakeLLM


FIXTURES = Path(__file__).with_name("fixtures")


def result_shape(value):
    """Keep every key/list entry and leaf type, not wording, timings or tokens."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return result_shape(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: result_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [result_shape(item) for item in value]
    return type(value).__name__


class SearchStore(FakeGraphStore):
    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def execute_query(self, query, parameters=None):
        if "FROM_CHUNK" in query:
            return [{"chunk_id": "chunk:1"}]
        if "Community" in query and "report_text" in query and "IN_COMMUNITY" not in query:
            return self._reports
        return super().execute_query(query, parameters)


async def search_result(mode, synthesize, tracked):
    # These old partial stores deliberately do not implement the full protocol.
    store = cast(Any, SearchStore())
    llm = cast(Any, FakeLLM())
    embedder = cast(Any, FakeEmbedder())
    if mode.startswith("local"):
        retriever = LocalSearchRetriever(
            store, llm, embedder, use_mixed_context=mode == "local-mixed",
            track_token_usage=tracked,
        )
    elif mode == "global":
        llm = cast(Any, FakeLLM([json.dumps({
            "answer": "A useful answer", "helpfulness": 80, "report_ids": ["report:c0:0"],
        }), "A final answer"]))
        retriever = GlobalSearchRetriever(graph_store=store, llm=llm, track_token_usage=tracked)
    else:
        retriever = DriftSearchRetriever(
            store, llm, embedder, config=DriftSearchConfig(use_hyde=False),
            token_counter=CharacterCounter(), track_token_usage=tracked,
        )
    if isinstance(retriever, GlobalSearchRetriever):
        if synthesize:
            return await retriever.search("Who leads Acme?", "coarsest")
        return await retriever.search(query="Who leads Acme?", community_level="coarsest", synthesize_response=False)
    if synthesize:
        return await retriever.search("Who leads Acme?", top_k=1)
    return await retriever.search(query="Who leads Acme?", top_k=1, synthesize_response=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "local-mixed", "global", "drift"])
@pytest.mark.parametrize("synthesize", [False, True], ids=["context-only", "synthesized"])
@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "usage-tracked"])
async def test_search_result_shapes(mode, synthesize, tracked):
    result = await search_result(mode, synthesize, tracked)
    assert isinstance(result, SearchResult)
    assert result.mode == ("local" if mode.startswith("local") else mode)
    assert result.query == "Who leads Acme?"
    assert bool(result.answer) is synthesize
    assert result.context
    assert result.citations and result.sources
    assert result.sources[0].document_id == result.citations[0].document_id
    assert result.sources[0].chunk_list == result.citations
    # Usage is absent before any LLM call, even with tracking enabled.
    assert ("token_usage" in result.metadata) is (tracked and (synthesize or mode in {"global", "drift"}))
    expected = json.loads((FIXTURES / "search_result_shapes.json").read_text(encoding="utf-8"))
    key = f"{mode}:synthesize={synthesize}:tracked={tracked}"
    assert {"result": result_shape(result), "sources": result_shape(result.sources)} == expected[key]


class CommunityStore(FakeGraphStoreWithCommunities):
    def __init__(self):
        super().__init__([{"id": "c0", "level": 0, "entity_count": 2},
                          {"id": "c1", "level": 1, "entity_count": 2}])
        self.embedded = []

    def get_unembedded_community_reports(self, graph_name, limit):
        return [{"id": report[1], "level": report[2], "report_text": report[3]}
                for report in self.reports if report[1] not in self.embedded]

    def upsert_community_report_vectors(self, ids, embeddings, graph_name, levels):
        assert graph_name == "compat-graph"
        assert len(ids) == len(embeddings) == len(levels) == 2
        self.embedded.extend(ids)


class ReportLLM:
    async def ainvoke(self, prompt, **kwargs):
        return LLMResponse(content=json.dumps({
            "title": "Community", "summary": "Alice leads Acme.", "rating": 7,
            "rating_explanation": "Connected entities.",
            "findings": [{"description": "Alice works at Acme.", "references": [
                {"target_id": "person:alice", "target_type": "entity"},
            ]}],
        }))


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "usage-tracked"])
async def test_community_build_result_shape(tracked):
    store = CommunityStore()
    counter = CharacterCounter()
    pipeline = CommunityPipeline(
        cast(Any, store), cast(Any, ReportLLM()), ["WORKS_AT"],
        graph_name="compat-graph", token_counter=counter,
        embedder=cast(Any, FakeEmbedder()), track_token_usage=tracked,
    )
    result = await pipeline.build(0)
    assert result["communities"] == result["reports"] == result["embedded_reports"] == 2
    assert result["levels"] == [1, 0]
    assert len(store.reports) == len(store.embedded) == 2
    assert counter.count_calls
    for stats, level in zip(result["level_stats"], [1, 0], strict=True):
        assert stats["level"] == level
        assert stats["attempted"] == stats["succeeded"] == 1
        assert stats["llm_calls"] == int(tracked)
        assert stats["skipped"] == stats["failed"] == 0
        assert stats["elapsed_seconds"] >= 0
    expected = json.loads((FIXTURES / "community_result_shapes.json").read_text(encoding="utf-8"))
    assert result_shape(result) == expected[str(tracked)]
