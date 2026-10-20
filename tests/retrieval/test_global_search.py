"""Tests for paper-style global search."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from recon_graphrag.retrieval.search_global import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
    GlobalSearchRetriever,
    MapBatch,
    PartialAnswer,
)
from tests.protocol_fakes import AsyncOnlyLLM, UnsupportedGraphStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reports(n: int = 5) -> list[dict]:
    return [
        {"id": f"report:{i}:0", "level": 0, "report_text": f"Summary for community {i}."}
        for i in range(n)
    ]


def _make_map_response(helpfulness: int = 75, answer: str = "Test answer.") -> str:
    return json.dumps({
        "answer": answer,
        "helpfulness": helpfulness,
        "report_ids": ["report:0:0"],
    })


def _make_report_with_json_ref(
    rid: str, target_id: str, target_type: str = "entity"
) -> dict:
    return {
        "id": rid,
        "level": 0,
        "report_text": f"Summary for {rid}.",
        "report_json": json.dumps({
            "id": rid,
            "community_id": rid.replace("report:", "c").replace(":0", ""),
            "level": 0,
            "findings": [
                {
                    "id": "f1",
                    "description": "A finding",
                    "rank": 1.0,
                    "references": [
                        {"target_id": target_id, "target_type": target_type}
                    ],
                }
            ],
        }),
    }


def _make_report_with_text_ref(
    rid: str, target_id: str, target_type: str = "entity"
) -> dict:
    return {
        "id": rid,
        "level": 0,
        "report_text": f"Summary with [refs: {target_type}:{target_id}]",
    }


class FakeGraphStore(UnsupportedGraphStore):
    def __init__(self, reports: list[dict] | None = None):
        self._reports = _make_reports() if reports is None else reports
        self.resolve_calls: list[tuple[str, list[str]]] = []

    def execute_query(self, query, parameters=None):
        parameters = parameters or {}
        if "FROM_CHUNK" in query:
            assert parameters["graph_name"] == "entity-graph"
            return [{"chunk_id": "chunk:1"}]
        if "MATCH (c:Claim" in query:
            return [{"chunk_id": "chunk:claim:1"}]
        if "source_chunk_ids" in query or "relationship_key" in query:
            return [{"chunk_id": "chunk:rel:1"}]
        level = parameters.get("level") if parameters else None
        if level is not None:
            rows = []
            for r in self._reports:
                if r.get("level") != level:
                    continue
                rows.append(dict(r))
            return rows
        return self._reports

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.resolve_calls.append((graph_name, chunk_ids))
        seen = set()
        rows = []
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            rows.append({
                "chunk_id": cid,
                "document_id": f"doc:{cid.split(':')[1]}" if ":" in cid else "doc:1",
                "document_name": f"Doc {cid}",
                "page_start": 1,
                "page_end": 1,
                "excerpt": f"Evidence text for {cid}.",
            })
        return rows


class FakeLLM(AsyncOnlyLLM):
    def __init__(self, map_responses: list[str] | None = None, reduce_response: str = "Final answer."):
        self._map_responses = map_responses or [_make_map_response()]
        self._reduce_response = reduce_response
        self._map_call_count = 0

    async def ainvoke(self, prompt, **kwargs):
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
    def test_oversized_report_is_truncated_to_map_budget(self):
        retriever = GlobalSearchRetriever(
            FakeGraphStore(), FakeLLM(), map_budget_tokens=1000
        )
        reports = [{"id": "huge", "report_text": "x" * 20000}]

        batches = retriever._create_batches("question", reports)
        overhead = retriever.token_counter.count(
            retriever.map_prompt.format(query="question", batch_text="")
        )

        assert len(batches) == 1
        assert batches[0].token_count <= 1000 - overhead

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
        search = GlobalSearchRetriever(store, llm)
        search.map_budget_tokens = search.token_counter.count(
            search.map_prompt.format(query="test query", batch_text="")
        ) + 50
        reports = _make_reports(5)
        batches = search._create_batches("test query", reports)
        assert len(batches) > 1


class TestPromptContract:
    @pytest.mark.parametrize("phase", ["map", "reduce"])
    def test_default_render_preserves_grounding_instructions(self, phase):
        # Rendering proves instruction delivery, not model obedience or factuality.
        search = GlobalSearchRetriever(FakeGraphStore(), FakeLLM())
        if phase == "map":
            prompt = search.map_prompt.format(query="Question?", batch_text="Evidence.")
        else:
            prompt, _ = search._prepare_reduce_prompt(
                "Question?",
                [PartialAnswer("0", "Evidence.", 75, ["report:0:0"])],
            )
        normalized = " ".join(prompt.split())
        assert "Question?" in prompt
        assert "Evidence." in prompt
        assert "Preserve uncertainty and relationship meaning" in normalized
        assert (
            "Do not turn possibilities, associations, locations, or temporal statements "
            "into established causes or origins."
        ) in normalized
        assert "Do not add facts" in normalized
        assert "say when requested details are unspecified." in normalized


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
        result = await search.search("test query", community_level=0, random_seed=42)

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

        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"
        assert store.resolve_calls == [("entity-graph", ["chunk:1"])]

    @pytest.mark.asyncio
    async def test_search_requires_level(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("query", community_level=None)
        assert "requires" in result.answer.lower()
        # No LLM call happened before this early return, so nothing was
        # recorded -- the key must be omitted, not present with value None.
        assert "token_usage" not in result.metadata

    @pytest.mark.asyncio
    async def test_search_empty_reports(self):
        store = FakeGraphStore(reports=[])
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("query", community_level=0)
        assert "No community reports" in result.answer
        assert "token_usage" not in result.metadata

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
        result = await search.search("query", community_level=0)
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


class TestResolveUsedReportIds:
    def test_keeps_valid_returned_ids(self):
        partials = [
            PartialAnswer(
                batch_id="0",
                answer="a",
                helpfulness=80,
                report_ids=["report:0:0"],
                batch_report_ids=["report:0:0", "report:1:0"],
            ),
        ]
        used = GlobalSearchRetriever._resolve_used_report_ids(partials)
        assert used == ["report:0:0"]

    def test_ignores_invalid_returned_ids(self):
        partials = [
            PartialAnswer(
                batch_id="0",
                answer="a",
                helpfulness=80,
                report_ids=["report:2:0"],
                batch_report_ids=["report:0:0", "report:1:0"],
            ),
        ]
        used = GlobalSearchRetriever._resolve_used_report_ids(partials)
        assert used == ["report:0:0", "report:1:0"]

    def test_falls_back_to_batch_ids_when_empty(self):
        partials = [
            PartialAnswer(
                batch_id="0",
                answer="a",
                helpfulness=80,
                report_ids=[],
                batch_report_ids=["report:0:0"],
            ),
        ]
        used = GlobalSearchRetriever._resolve_used_report_ids(partials)
        assert used == ["report:0:0"]

    def test_falls_back_to_batch_ids_when_missing(self):
        partials = [
            PartialAnswer(
                batch_id="0",
                answer="a",
                helpfulness=80,
                report_ids=None,  # pyright: ignore[reportArgumentType] - exercise legacy missing IDs
                batch_report_ids=["report:0:0"],
            ),
        ]
        used = GlobalSearchRetriever._resolve_used_report_ids(partials)
        assert used == ["report:0:0"]

    def test_preserves_deterministic_order(self):
        partials = [
            PartialAnswer(
                batch_id="1",
                answer="b",
                helpfulness=60,
                report_ids=["report:1:0"],
                batch_report_ids=["report:1:0"],
            ),
            PartialAnswer(
                batch_id="0",
                answer="a",
                helpfulness=80,
                report_ids=["report:0:0", "report:1:0"],
                batch_report_ids=["report:0:0", "report:1:0"],
            ),
        ]
        used = GlobalSearchRetriever._resolve_used_report_ids(partials)
        assert used == ["report:1:0", "report:0:0"]


class TestExtractReportJsonRefs:
    def test_extracts_entity_refs(self):
        report_json = json.dumps({
            "findings": [
                {
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"}
                    ]
                }
            ]
        })
        refs = GlobalSearchRetriever._extract_report_json_refs(report_json)
        assert refs == [{"target_id": "person:alice", "target_type": "entity"}]

    def test_ignores_unsupported_types(self):
        report_json = json.dumps({
            "findings": [
                {
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"},
                        {"target_id": "bad", "target_type": "unsupported"},
                    ]
                }
            ]
        })
        refs = GlobalSearchRetriever._extract_report_json_refs(report_json)
        assert refs == [{"target_id": "person:alice", "target_type": "entity"}]

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError):
            GlobalSearchRetriever._extract_report_json_refs("not json")


class TestExtractReportTextRefs:
    def test_extracts_refs_with_colon_in_target_id(self):
        text = "Summary [refs: entity:product:a, claim:claim:123]"
        refs = GlobalSearchRetriever._extract_report_text_refs(text)
        assert refs == [
            {"target_id": "product:a", "target_type": "entity"},
            {"target_id": "claim:123", "target_type": "claim"},
        ]

    def test_ignores_arbitrary_prose(self):
        text = "This mentions entity:foo but has no [refs: ...] block."
        refs = GlobalSearchRetriever._extract_report_text_refs(text)
        assert refs == []

    def test_ignores_unsupported_types(self):
        text = "Summary [refs: entity:alice, unsupported:bad]"
        refs = GlobalSearchRetriever._extract_report_text_refs(text)
        assert refs == [{"target_id": "alice", "target_type": "entity"}]


class TestSourceResolution:
    @pytest.mark.asyncio
    async def test_search_resolves_citations_from_report_json_refs(self):
        reports = [_make_report_with_json_ref("report:0:0", "person:alice")]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"
        assert result.metadata["source_report_ids_used"] == 1
        assert result.metadata["source_references_extracted"] == 1
        assert result.metadata["source_citations_resolved"] == 1

    @pytest.mark.asyncio
    async def test_search_falls_back_to_rendered_refs(self):
        reports = [_make_report_with_text_ref("report:0:0", "person:bob")]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Bob is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"
        assert result.metadata["source_references_extracted"] == 1

    @pytest.mark.asyncio
    async def test_search_falls_back_to_batch_report_ids(self):
        reports = [_make_report_with_json_ref("report:0:0", "person:alice")]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.metadata["source_report_ids_used"] == 1

    @pytest.mark.asyncio
    async def test_search_combines_and_dedupes_map_and_report_refs(self):
        reports = [_make_report_with_json_ref("report:0:0", "person:alice")]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"}
                    ],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert result.metadata["source_references_extracted"] == 1

    @pytest.mark.asyncio
    async def test_search_is_nonfatal_on_malformed_report_json(self):
        reports = [
            {
                "id": "report:0:0",
                "level": 0,
                "report_text": "Summary",
                "report_json": "not valid json",
            }
        ]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"}
                    ],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert len(result.citations) == 1
        assert any(
            "reference extraction failed" in e
            for e in result.metadata["source_resolution_errors"]
        )

    @pytest.mark.asyncio
    async def test_search_is_nonfatal_on_citation_lookup_error(self):
        class RaisingGraphStore(FakeGraphStore):
            def resolve_chunk_citations(self, graph_name, chunk_ids):
                raise RuntimeError("lookup failed")

        reports = [_make_report_with_json_ref("report:0:0", "person:alice")]
        store = RaisingGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search("test query", community_level=0, random_seed=42)

        assert result.citations == []
        assert any(
            "citation resolution failed" in e
            for e in result.metadata["source_resolution_errors"]
        )


class TestParseMapResponseEdgeCases:
    def test_parse_missing_helpfulness_defaults_to_zero(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"answer": "some answer"}))
        assert data["helpfulness"] == 0

    def test_parse_non_numeric_helpfulness_defaults_to_zero(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"answer": "x", "helpfulness": "high"}))
        assert data["helpfulness"] == 0

    def test_parse_missing_answer_defaults_to_empty(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        search = GlobalSearchRetriever(store, llm)
        data = search._parse_map_response(json.dumps({"helpfulness": 50}))
        assert data.get("answer", "") == ""


class TestSynthesizeResponse:
    @pytest.mark.asyncio
    async def test_synthesize_false_skips_reduce_returns_context(self):
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
        result = await search.search(
            "test query", community_level=0, random_seed=42, synthesize_response=False
        )

        assert result.mode == "global"
        assert result.answer == ""
        assert "Part 1." in result.context
        assert "Final synthesized" not in result.context
        assert result.metadata["synthesize_response"] is False
        assert result.metadata["response_synthesis_skipped"] is True
        assert result.metadata["reduce_partials_used"] == 0
        # Map calls still ran
        assert result.metadata["map_succeeded"] >= 1
        # Reduce LLM was never called
        assert llm._map_call_count >= 1

    @pytest.mark.asyncio
    async def test_synthesize_false_preserves_citations(self):
        reports = [_make_report_with_json_ref("report:0:0", "person:alice")]
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                json.dumps({
                    "answer": "Alice is relevant.",
                    "helpfulness": 80,
                    "report_ids": ["report:0:0"],
                    "references": [
                        {"target_id": "person:alice", "target_type": "entity"}
                    ],
                })
            ],
            reduce_response="Final answer.",
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search(
            "test query", community_level=0, random_seed=42, synthesize_response=False
        )

        assert result.answer == ""
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"
        assert result.metadata["synthesize_response"] is False

    @pytest.mark.asyncio
    async def test_synthesize_false_all_score_zero(self):
        reports = _make_reports(2)
        store = FakeGraphStore(reports)
        llm = FakeLLM(
            map_responses=[
                _make_map_response(helpfulness=0),
                _make_map_response(helpfulness=0),
            ]
        )
        search = GlobalSearchRetriever(store, llm)
        result = await search.search(
            "query", community_level=0, synthesize_response=False
        )
        # All zero still returns early with "No relevant" message
        assert "No relevant" in result.answer


# Reduction admission is measured on the exact provider string, not item sums.
class ReductionCharacterCounter:
    def __init__(self):
        self.counted: list[str] = []

    def count(self, text: str) -> int:
        self.counted.append(text)
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        raise AssertionError("Reduction must not truncate evidence or the question")


class RecordingReductionLLM:
    def __init__(self, map_responses: list[str] | None = None):
        self.map_responses = list(map_responses or [])
        self.map_prompts: list[str] = []
        self.reduce_prompts: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs):
        if prompt.startswith("MAP|"):
            self.map_prompts.append(prompt)
            return MagicMock(content=self.map_responses.pop(0))
        self.reduce_prompts.append(prompt)
        return MagicMock(content="Reduced answer.")


def _reduction_search(
    budget: int,
    *,
    template: str = "Q:{query}\nP:{partial_text}\nEND",
    counter: ReductionCharacterCounter | None = None,
    store: FakeGraphStore | None = None,
    map_responses: list[str] | None = None,
    retriever_type: type[GlobalSearchRetriever] = GlobalSearchRetriever,
) -> tuple[GlobalSearchRetriever, RecordingReductionLLM]:
    llm = RecordingReductionLLM(map_responses)
    # These deliberate partial doubles implement only the exercised store/LLM seams.
    search = retriever_type(
        MagicMock(wraps=store if store is not None else FakeGraphStore()),
        MagicMock(wraps=llm),
        map_prompt="MAP|{query}|{batch_text}",
        reduce_prompt=template,
        token_counter=counter if counter is not None else ReductionCharacterCounter(),
        reduce_budget_tokens=budget,
        track_token_usage=False,
    )
    return search, llm


class TestReducePromptAdmission:
    @pytest.mark.parametrize("count", [1, 2], ids=["first", "later"])
    async def test_exact_fit_counts_unicode_scores_separators_and_references(self, count):
        partials = [
            PartialAnswer("0", "Café 🦉", 90, [], [
                {"target_type": "claim", "target_id": "claim:東京:" + "x" * 40},
            ]),
            PartialAnswer("1", "Résumé 第二", 70, [], [
                {"target_type": "entity", "target_id": "person:Élodie"},
            ]),
        ][:count]
        parts = ["[Score 90] Café 🦉\nReferences: claim:claim:東京:" + "x" * 40,
                 "[Score 70] Résumé 第二\nReferences: entity:person:Élodie"][:count]
        expected = "Q:Original question?\nP:" + "\n\n".join(parts) + "\nEND"
        search, llm = _reduction_search(len(expected))

        prompt, retained = search._prepare_reduce_prompt("Original question?", partials)
        assert prompt == expected
        assert retained == partials
        assert await search._reduce_phase("Original question?", partials) == "Reduced answer."
        assert llm.reduce_prompts == [expected]
        assert search.token_counter.count(expected) == search.reduce_budget_tokens
        assert partials[0].answer == "Café 🦉"  # No mutation/truncation.

    async def test_oversized_first_stops_prefix_even_when_later_would_fit(self):
        empty = "Q:question\nP:\nEND"
        search, llm = _reduction_search(len(empty) + 20)
        partials = [PartialAnswer("0", "large" * 100, 90, []),
                    PartialAnswer("1", "tiny", 70, [])]
        prompt, retained = search._prepare_reduce_prompt("question", partials)
        assert (prompt, retained) == (empty, [])
        await search._reduce_phase("question", partials)
        assert llm.reduce_prompts == [empty]
        assert len(empty) <= search.reduce_budget_tokens

    @pytest.mark.parametrize("boundary", ["separator", "references"])
    async def test_rejects_later_partial_when_complete_render_exceeds_budget(self, boundary):
        first = "[Score 90] one"
        second = "[Score 70] two"
        refs = []
        if boundary == "references":
            refs = [{"target_type": "claim", "target_id": "claim:" + "証" * 80}]
        partials = [PartialAnswer("0", "one", 90, []),
                    PartialAnswer("1", "two", 70, [], refs)]
        empty = "Q:q\nP:\nEND"
        # Without the separator/reference line both answers would appear to fit.
        search, llm = _reduction_search(len(empty) + len(first) + len(second))
        prompt, retained = search._prepare_reduce_prompt("q", partials)
        assert retained == partials[:1]
        await search._reduce_phase("q", partials)
        assert llm.reduce_prompts == [prompt] == [f"Q:q\nP:{first}\nEND"]
        assert search.token_counter.count(prompt) <= search.reduce_budget_tokens

    @pytest.mark.parametrize("query,template,general", [
        ("Original question?", "x" * 100 + "{query}{partial_text}", False),
        ("Original question? " + "q" * 100, "{query}{partial_text}", False),
        ("History: " + "past " * 30 + "\nOriginal question?", "{query}{partial_text}", False),
        ("Original question?", "History: " + "past " * 30 + "\n{query}{partial_text}", False),
        ("Original question?", "{query}{partial_text}", True),
    ], ids=["template", "query", "query-history", "literal-history", "general-knowledge"])
    async def test_fixed_overflow_rejects_before_provider_without_removing_question(
        self, query, template, general,
    ):
        counter = ReductionCharacterCounter()
        search, llm = _reduction_search(40, template=template, counter=counter)
        effective = template + (GENERAL_KNOWLEDGE_INSTRUCTION if general else "")
        fixed = effective.format(query=query, partial_text="")
        with pytest.raises(ValueError, match=f"count {len(fixed)}.*reduce_budget_tokens=40"):
            await search._reduce_phase(query, [], use_general_knowledge=general)
        assert llm.reduce_prompts == []
        assert counter.counted == [fixed]
        assert "Original question?" in fixed

    @pytest.mark.parametrize("general", [False, True])
    async def test_empty_partials_fixed_exact_fit_including_general_suffix(self, general):
        expected = "Q:History: earlier\nOriginal question?\nP:\nEND"
        if general:
            expected += GENERAL_KNOWLEDGE_INSTRUCTION
        search, llm = _reduction_search(len(expected))
        await search._reduce_phase(
            "History: earlier\nOriginal question?", [], use_general_knowledge=general
        )
        assert llm.reduce_prompts == [expected]

    async def test_stable_nonadditive_counter_counts_complete_candidates(self):
        class BoundaryCounter(ReductionCharacterCounter):
            def count(self, text: str) -> int:
                return super().count(text) + (1000 if "one\n\n[Score" in text else 0)

        counter = BoundaryCounter()
        search, llm = _reduction_search(100, counter=counter)
        partials = [PartialAnswer("0", "one", 90, []),
                    PartialAnswer("1", "two", 70, [])]
        await search._reduce_phase("q", partials)
        assert llm.reduce_prompts == ["Q:q\nP:[Score 90] one\nEND"]
        assert counter.counted == ["Q:q\nP:\nEND", "Q:q\nP:[Score 90] one\nEND",
                                   "Q:q\nP:[Score 90] one\n\n[Score 70] two\nEND",
                                   llm.reduce_prompts[0]]
        assert counter.count(llm.reduce_prompts[0]) <= 100

    async def test_final_recount_fails_closed_if_counter_changes(self):
        class ChangingCounter(ReductionCharacterCounter):
            def count(self, text: str) -> int:
                count = super().count(text)
                return count if len(self.counted) < 3 else 1001

        search, llm = _reduction_search(100, counter=ChangingCounter())
        with pytest.raises(ValueError, match="final prompt count 1001.*reduce_budget_tokens=100"):
            await search._reduce_phase("q", [PartialAnswer("0", "one", 90, [])])
        assert llm.reduce_prompts == []


class RetainedCitationStore(FakeGraphStore):
    def __init__(self, reports):
        super().__init__(reports)
        self.report_lookups: list[list[str]] = []

    def execute_query(self, query, parameters=None):
        parameters = parameters or {}
        if "report_ids" in parameters:
            self.report_lookups.append(parameters["report_ids"])
            return [r for r in self._reports if r["id"] in parameters["report_ids"]]
        if "entity_ids" in parameters:
            return [{"chunk_id": f"chunk:{rid}"} for rid in parameters["entity_ids"]]
        if "claim_ids" in parameters:
            return [{"chunk_id": f"chunk:{rid}"} for rid in parameters["claim_ids"]]
        return super().execute_query(query, parameters)


class TestRetainedReductionEvidence:
    @staticmethod
    def make_search(
        budget, *, source="explicit", score=90, retriever_type=GlobalSearchRetriever,
    ):
        report_factory = _make_report_with_text_ref if source == "text" else _make_report_with_json_ref
        store = RetainedCitationStore([
            report_factory("report:0:0", "kept", "claim"),
            report_factory("report:1:0", "dropped", "claim"),
        ])
        responses = []
        for i, target in enumerate(["kept", "dropped"]):
            responses.append(json.dumps({
                "answer": target, "helpfulness": score - i if score else 0,
                # Invalid IDs exercise existing batch-ID fallback/validation.
                "report_ids": ["report:invented"],
                "references": ([{"target_type": "entity", "target_id": target}]
                               if source == "explicit" else []),
            }))
        search, llm = _reduction_search(
            budget, store=store, map_responses=responses, retriever_type=retriever_type,
        )
        search._create_batches = MagicMock(return_value=[
            MapBatch(str(i), [f"report:{i}:0"], f"report {i}", 10) for i in range(2)
        ])
        return search, llm, store

    @pytest.mark.parametrize("source", ["explicit", "json", "text"])
    @pytest.mark.parametrize("admitted", [0, 1])
    async def test_only_admitted_explicit_and_report_fallback_citations(self, source, admitted):
        expected = "Q:q\nP:\nEND"
        if admitted:
            expected = "Q:q\nP:[Score 90] kept"
            if source == "explicit":
                expected += "\nReferences: entity:kept"
            expected += "\nEND"
        search, llm, store = self.make_search(len(expected), source=source)
        result = await search.search("q", 0)
        assert llm.reduce_prompts == [expected]
        assert result.metadata["reduce_partials_used"] == admitted
        assert [c.chunk_id for c in result.citations] == (["chunk:kept"] if admitted else [])
        assert result.metadata["source_report_ids_used"] == admitted
        assert result.metadata["source_references_extracted"] == (
            admitted * (2 if source == "explicit" else 1)
        )
        assert result.metadata["source_citations_resolved"] == admitted
        assert result.metadata["source_resolution_errors"] == []
        assert store.report_lookups == ([["report:0:0"]] if admitted else [])
        assert "kept" in result.context and "dropped" in result.context
        assert result.context == (
            "[Batch 0] (score: 90)\nkept\n\n---\n\n[Batch 1] (score: 89)\ndropped"
        )  # Retrieval context is intentionally broader than synthesis evidence.

    async def test_search_dispatches_to_reduce_override_with_admitted_partials(self):
        calls = []

        class CustomReducer(GlobalSearchRetriever):
            async def _reduce_phase(self, query, partials, *, use_general_knowledge=False):
                calls.append((query, list(partials), use_general_knowledge))
                return "CUSTOM REDUCER SENTINEL"

        expected = "Q:q\nP:[Score 90] kept\nReferences: entity:kept\nEND"
        search, llm, _ = self.make_search(len(expected), retriever_type=CustomReducer)
        result = await search.search("q", 0)
        assert result.answer == "CUSTOM REDUCER SENTINEL"
        assert len(calls) == 1
        assert calls[0][0] == "q" and calls[0][2] is False
        assert [p.answer for p in calls[0][1]] == ["kept"]
        assert result.metadata["reduce_partials_used"] == 1
        assert [c.chunk_id for c in result.citations] == ["chunk:kept"]
        assert llm.reduce_prompts == []

    @pytest.mark.parametrize("overflow", [False, True])
    async def test_delegating_override_reuses_admitted_prompt_and_recounts(self, overflow):
        class DelegatingReducer(GlobalSearchRetriever):
            async def _reduce_phase(self, query, partials, *, use_general_knowledge=False):
                return await super()._reduce_phase(
                    query, partials, use_general_knowledge=use_general_knowledge,
                )

        expected = "Q:q\nP:[Score 90] kept\nReferences: entity:kept\nEND"

        class ChangingCounter(ReductionCharacterCounter):
            def count(self, text):
                count = super().count(text)
                # Admission accepts this string once; the only repeat is final recount.
                if text == expected and self.counted.count(text) > 1:
                    return len(expected) + 1 if overflow else count
                if text == "Q:q\nP:\nEND" and self.counted.count(text) > 1:
                    raise AssertionError("Search must not repeat admission")
                return count

        search, llm, _ = self.make_search(len(expected), retriever_type=DelegatingReducer)
        search.token_counter = ChangingCounter()
        if overflow:
            with pytest.raises(ValueError, match="final prompt count"):
                await search.search("q", 0)
            assert llm.reduce_prompts == []
        else:
            result = await search.search("q", 0)
            assert llm.reduce_prompts == [expected]
            assert result.metadata["reduce_partials_used"] == 1
            assert [c.chunk_id for c in result.citations] == ["chunk:kept"]
        assert search.token_counter.counted.count(expected) == 2

    @pytest.mark.parametrize("mutation", ["answer", "references", "list", "query", "general"])
    async def test_delegating_override_mutation_fails_closed(self, mutation):
        class MutatingReducer(GlobalSearchRetriever):
            async def _reduce_phase(self, query, partials, *, use_general_knowledge=False):
                if mutation == "answer":
                    partials[0].answer = "changed"
                elif mutation == "references":
                    partials[0].references[0]["target_id"] = "changed"
                elif mutation == "list":
                    partials.clear()
                elif mutation == "query":
                    query = "changed"
                else:
                    use_general_knowledge = not use_general_knowledge
                return await super()._reduce_phase(
                    query, partials, use_general_knowledge=use_general_knowledge,
                )

        search, llm, _ = self.make_search(1000, retriever_type=MutatingReducer)
        with pytest.raises(ValueError, match="hook changed admitted inputs"):
            await search.search("q", 0)
        assert llm.reduce_prompts == []

    async def test_concurrent_delegating_searches_keep_request_local_prompts(self):
        both_entered = asyncio.Event()
        queries = []

        class ConcurrentReducer(GlobalSearchRetriever):
            async def _reduce_phase(self, query, partials, *, use_general_knowledge=False):
                queries.append(query)
                if len(queries) == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=2)
                return await super()._reduce_phase(
                    query, partials, use_general_knowledge=use_general_knowledge,
                )

        search, llm, _ = self.make_search(1000, retriever_type=ConcurrentReducer)
        llm.map_responses *= 2
        results = await asyncio.gather(search.search("first", 0), search.search("second", 0))
        assert {p.splitlines()[0] for p in llm.reduce_prompts} == {"Q:first", "Q:second"}
        assert all(result.metadata["reduce_partials_used"] == 2 for result in results)

    async def test_search_fixed_overflow_may_follow_map_but_never_calls_reduce(self):
        search, llm, _ = self.make_search(1)
        with pytest.raises(ValueError, match="fixed prompt count.*reduce_budget_tokens=1"):
            await search.search("Original question?", 0)
        assert len(llm.map_prompts) == 2
        assert llm.reduce_prompts == []

    async def test_general_knowledge_search_counts_suffix_before_provider(self):
        search, llm, _ = self.make_search(100, score=0)
        search.allow_general_knowledge = True
        with pytest.raises(ValueError, match="fixed prompt count.*reduce_budget_tokens=100"):
            await search.search("q", 0)
        assert len(llm.map_prompts) == 2
        assert llm.reduce_prompts == []

    @pytest.mark.parametrize("general", [False, True])
    async def test_nonsynthesis_tiny_budget_preserves_all_context_and_citations(self, general):
        search, llm, store = self.make_search(1, score=0 if general else 90)
        search.allow_general_knowledge = general
        result = await search.search("q", 0, synthesize_response=False)
        assert result.answer == ""
        assert "kept" in result.context and "dropped" in result.context
        assert [c.chunk_id for c in result.citations] == ["chunk:kept", "chunk:dropped"]
        assert result.metadata["reduce_partials_used"] == 0
        assert result.metadata["source_report_ids_used"] == 2
        assert result.metadata["source_references_extracted"] == 4
        assert result.metadata["source_citations_resolved"] == 2
        assert result.metadata["response_synthesis_skipped"] is True
        assert store.report_lookups == [["report:0:0", "report:1:0"]]
        assert len(llm.map_prompts) == 2
        assert llm.reduce_prompts == []
        assert isinstance(search.token_counter, ReductionCharacterCounter)
        assert not any(text.startswith("Q:") for text in search.token_counter.counted)
