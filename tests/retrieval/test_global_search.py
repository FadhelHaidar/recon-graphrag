"""Tests for paper-style global search."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.retrieval.global_search import (
    GlobalSearchRetriever,
    MapBatch,
    PartialAnswer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reports(n: int = 5) -> list[dict]:
    return [
        {"id": f"report:{i}:0", "level": 0, "summary": f"Summary for community {i}."}
        for i in range(n)
    ]


def _make_map_response(helpfulness: int = 75, answer: str = "Test answer.") -> str:
    return json.dumps({
        "answer": answer,
        "helpfulness": helpfulness,
        "report_ids": ["report:0:0"],
    })


class FakeGraphStore:
    def __init__(self, reports: list[dict] | None = None):
        self._reports = _make_reports() if reports is None else reports
        self.resolve_calls: list[tuple[str, list[str]]] = []

    def execute_query(self, query, params=None):
        if "FROM_CHUNK" in query:
            assert params["graph_name"] == "entity-graph"
            return [{"chunk_id": "chunk:1"}]
        level = params.get("level") if params else None
        if level is not None:
            return [r for r in self._reports if r.get("level") == level]
        return self._reports

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.resolve_calls.append((graph_name, chunk_ids))
        return [
            {
                "chunk_id": "chunk:1",
                "document_id": "doc:1",
                "document_name": "Doc 1",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "Evidence text.",
            }
        ]


class FakeLLM:
    def __init__(self, map_responses: list[str] | None = None, reduce_response: str = "Final answer."):
        self._map_responses = map_responses or [_make_map_response()]
        self._reduce_response = reduce_response
        self._map_call_count = 0

    async def ainvoke(self, prompt):
        # Distinguish map vs reduce by prompt content
        if "Synthesize" in prompt or "Partial Answers" in prompt:
            return MagicMock(content=self._reduce_response)
        idx = min(self._map_call_count, len(self._map_responses) - 1)
        content = self._map_responses[idx]
        self._map_call_count += 1
        return MagicMock(content=content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShuffle:
    def test_deterministic_with_seed(self):
        reports = _make_reports(10)
        s1 = GlobalSearchRetriever._shuffle(reports, seed=42)
        s2 = GlobalSearchRetriever._shuffle(reports, seed=42)
        assert [r["id"] for r in s1] == [r["id"] for r in s2]

    def test_different_seeds_differ(self):
        reports = _make_reports(10)
        s1 = GlobalSearchRetriever._shuffle(reports, seed=1)
        s2 = GlobalSearchRetriever._shuffle(reports, seed=2)
        # Very unlikely to be the same with 10 items
        assert [r["id"] for r in s1] != [r["id"] for r in s2]

    def test_preserves_all_items(self):
        reports = _make_reports(5)
        shuffled = GlobalSearchRetriever._shuffle(reports, seed=42)
        assert len(shuffled) == 5
        assert {r["id"] for r in shuffled} == {r["id"] for r in reports}


class TestCreateBatches:
    def test_single_batch_fits_all(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm, map_budget_tokens=10000)
        reports = _make_reports(3)
        batches = search._create_batches("test query", reports)
        assert len(batches) == 1
        assert len(batches[0].report_ids) == 3

    def test_multiple_batches_when_budget_small(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm, map_budget_tokens=300)
        reports = _make_reports(5)
        batches = search._create_batches("test query", reports)
        assert len(batches) > 1


class TestMapPhase:
    @pytest.mark.asyncio
    async def test_map_parses_response(self):
        store = FakeGraphStore()
        llm = FakeLLM(map_responses=[_make_map_response(helpfulness=80, answer="Nolan directed.")])
        search = GlobalSearchRetriever(store, llm)
        batches = [MapBatch(batch_id="0", report_ids=["r1"], text="content", token_count=100)]

        partials = await search._map_phase("Who directed?", batches)
        assert len(partials) == 1
        assert partials[0].helpfulness == 80
        assert partials[0].answer == "Nolan directed."

    @pytest.mark.asyncio
    async def test_map_handles_error(self):
        store = FakeGraphStore()

        async def fail(prompt):
            raise RuntimeError("LLM failed")

        llm = MagicMock()
        llm.ainvoke = fail
        search = GlobalSearchRetriever(store, llm)
        batches = [MapBatch(batch_id="0", report_ids=["r1"], text="content", token_count=100)]

        partials = await search._map_phase("query", batches)
        assert len(partials) == 1
        assert partials[0].error is not None
        assert partials[0].helpfulness == 0


class TestReducePhase:
    @pytest.mark.asyncio
    async def test_reduce_calls_llm(self):
        store = FakeGraphStore()
        llm = FakeLLM(reduce_response="Combined final answer.")
        search = GlobalSearchRetriever(store, llm)

        partials = [
            PartialAnswer(batch_id="0", answer="Part 1.", helpfulness=80, report_ids=["r1"]),
            PartialAnswer(batch_id="1", answer="Part 2.", helpfulness=60, report_ids=["r2"]),
        ]
        result = await search._reduce_phase("query", partials)
        assert result == "Combined final answer."


class TestFullSearch:
    @pytest.mark.asyncio
    async def test_search_returns_result(self):
        reports = _make_reports(3)
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                _make_map_response(helpfulness=80, answer="Part 1."),
                _make_map_response(helpfulness=60, answer="Part 2."),
                _make_map_response(helpfulness=0, answer="No info."),
            ],
            reduce_response="Final synthesized answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", level=0, random_seed=42)

        assert result.mode == "global"
        assert "Final synthesized" in result.answer
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_search_resolves_only_explicit_map_references(self):
        reports = _make_reports(1)
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps(
                    {
                        "answer": "Alice is relevant.",
                        "helpfulness": 80,
                        "report_ids": ["report:0:0"],
                        "references": [
                            {"target_id": "person:alice", "target_type": "entity"}
                        ],
                    }
                )
            ],
            reduce_response="Final synthesized answer.",
        )
        search = GlobalSearchRetriever(store, llm)

        result = await search.search("test query", level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"
        assert store.resolve_calls == [("entity-graph", ["chunk:1"])]

    @pytest.mark.asyncio
    async def test_search_requires_level(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("query", level=None)
        assert "requires" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_search_empty_reports(self):
        store = FakeGraphStore(reports=[])
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("query", level=0)
        assert "No community reports" in result.answer

    @pytest.mark.asyncio
    async def test_search_all_score_zero(self):
        reports = _make_reports(2)
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                _make_map_response(helpfulness=0),
                _make_map_response(helpfulness=0),
            ]
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("query", level=0)
        assert "No relevant" in result.answer


class TestParseMapResponse:
    def test_parse_valid_json(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(_make_map_response(helpfulness=85))
        assert data["helpfulness"] == 85

    def test_parse_strips_fences(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        fenced = f"```json\n{_make_map_response()}\n```"
        data = search._parse_map_response(fenced)
        assert "answer" in data

    def test_parse_extracts_json_from_wrapped_response(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        wrapped = f"Here is the JSON:\n{_make_map_response(helpfulness=80)}\nThanks."
        data = search._parse_map_response(wrapped)
        assert data["helpfulness"] == 80

    def test_parse_accepts_numeric_string_score(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"answer": "x", "helpfulness": "70"}))
        assert data["helpfulness"] == 70

    def test_parse_clamps_invalid_score(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"answer": "x", "helpfulness": 150}))
        assert data["helpfulness"] == 0

    def test_parse_clamps_negative_score(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"answer": "x", "helpfulness": -5}))
        assert data["helpfulness"] == 0

    def test_parse_keeps_only_valid_references(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(
            json.dumps(
                {
                    "answer": "x",
                    "helpfulness": 50,
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"},
                        {"target_id": "bad", "target_type": "unsupported"},
                        {"target_type": "entity"},
                    ],
                }
            )
        )
        assert data["references"] == [
            {"target_id": "person:alice", "target_type": "entity"}
        ]
