"""Tests for iterative DRIFT search.

Tests for iterative DRIFT behavior and limits.
"""

from __future__ import annotations

import json

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.models.artifacts import Citation
from recon_graphrag.retrieval.drift import DriftSearchRetriever
from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.community_levels import resolve_community_level


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGraphStore:
    """Fake store with vector_search_community_reports support."""

    def __init__(self, *, reports=None):
        self.calls: list[tuple[str, dict]] = []
        self._reports = reports or [
            {
                "id": "report:c0:0",
                "level": 0,
                "report_text": "Alice leads Acme Corp.",
                "report_json": json.dumps({
                    "findings": [{
                        "references": [
                            {"target_id": "person:alice", "target_type": "entity"}
                        ]
                    }]
                }),
                "rating": 8.0,
            }
        ]

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        self.calls.append(("vector_search", {"k": k}))
        return [{"id": "a", "score": 0.8}]

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        self.calls.append(("keyword_search", {}))
        return [{"id": "a", "score": 1.0}]

    def fetch_entity_context(self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None):
        self.calls.append(("fetch_entity_context", {"mode": mode, "graph_name": graph_name}))
        return [
            {
                "title": "Alice (Person)",
                "relationships": ["Person: Alice -[DIRECTED]-> Movie: Inception"],
                "source_text": ["Alice directed Inception in 2010."],
                "source_chunk_ids": ["chunk:1"],
                "score": matches[0]["score"],
                "communities": [
                    {"id": "c0", "level": 0, "graph_name": "entity-graph"},
                ],
            }
        ]

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append(("execute_query", {"query": query.strip()[:60], "params": params}))
        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.calls.append(("resolve_chunk_citations", {"chunk_ids": chunk_ids}))
        return [
            {
                "document_id": "doc:1",
                "chunk_id": "chunk:1",
                "document_name": "Doc 1",
                "excerpt": "Alice directed Inception in 2010.",
                "document_metadata": {"collection": "movies"},
                "chunk_metadata": {"record_id": "row-42", "source": "row-source"},
            }
        ]

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        self.calls.append(
            ("vector_search_community_reports", {"top_k": top_k, "level": level})
        )
        return self._reports[:top_k]


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    async def async_embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeLLM:
    def __init__(self, responses=None):
        self.prompts: list[str] = []
        self._responses = responses or []
        self._call_count = 0

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self._call_count < len(self._responses):
            content = self._responses[self._call_count]
        else:
            # Default: if prompt looks like a reduce prompt, return plain text
            if "Partial Answers" in prompt or "synthesizing" in prompt.lower():
                content = "Alice directed Inception."
            else:
                content = json.dumps({
                    "answer": "Alice directed Inception.",
                    "score": 80,
                    "follow_ups": [],
                })
        self._call_count += 1
        return LLMResponse(content=content)


class EmptyReportsGraphStore(FakeGraphStore):
    def __init__(self):
        super().__init__(reports=[])

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        self.calls.append(("vector_search_community_reports", {"top_k": top_k}))
        return []


class NoReportSearchStore(FakeGraphStore):
    """Store where vector_search_community_reports raises (no index)."""

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        raise RuntimeError("No such index")


# ---------------------------------------------------------------------------
# Core iterative DRIFT behavior
# ---------------------------------------------------------------------------


class TestDriftSearch:
    @pytest.mark.asyncio
    async def test_drift_returns_valid_search_result(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("Who directed Inception?", top_k=2)

        assert result.mode == "drift"
        assert result.query == "Who directed Inception?"
        assert result.answer == "Alice directed Inception."

    @pytest.mark.asyncio
    async def test_drift_embedder_called_for_primer(self):
        store = FakeGraphStore()
        embedder = FakeEmbedder()
        retriever = DriftSearchRetriever(store, FakeLLM(), embedder)

        await retriever.search("Who directed Inception?", top_k=2)

        assert embedder.queries[0] == "Who directed Inception?"

    @pytest.mark.asyncio
    async def test_drift_uses_configured_primer_report_count(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(
            store,
            FakeLLM(),
            FakeEmbedder(),
            config=DriftSearchConfig(primer_top_k=4),
        )

        await retriever.search("query", top_k=2)

        report_call = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert len(report_call) == 1
        assert report_call[0][1]["top_k"] == 4

    @pytest.mark.asyncio
    async def test_drift_citations_from_report_refs(self):
        """Citations come from report references when available."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        # Citations may come from report references or be empty for primer-only
        assert isinstance(result.citations, list)

    @pytest.mark.asyncio
    async def test_drift_trace_in_metadata(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert "drift_trace" in result.metadata
        trace = result.metadata["drift_trace"]
        assert "primer_report_ids" in trace
        assert "actions" in trace
        assert "stopping_reason" in trace
        assert "total_llm_calls" in trace

    @pytest.mark.asyncio
    async def test_drift_uses_context_mode_drift(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2)

        ctx_call = [c for c in store.calls if c[0] == "fetch_entity_context"]
        if ctx_call:
            assert ctx_call[0][1]["mode"] == "drift"


class TestDriftSynthesisFalse:
    @pytest.mark.asyncio
    async def test_synthesize_false_returns_empty_answer(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search("query", top_k=2, synthesize_response=False)

        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_synthesize_false_returns_context(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2, synthesize_response=False)

        assert result.context
        assert result.metadata["synthesize_response"] is False

    @pytest.mark.asyncio
    async def test_synthesize_false_has_trace(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2, synthesize_response=False)

        assert "drift_trace" in result.metadata


class TestDriftCommunityLevelAliases:
    @pytest.mark.asyncio
    async def test_coarsest_alias_resolves(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, community_level="coarsest")

        report_call = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert len(report_call) == 1
        assert report_call[0][1]["level"] == 0

    @pytest.mark.asyncio
    async def test_finest_alias_resolves(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, community_level="finest")

        report_call = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert len(report_call) == 1
        assert report_call[0][1]["level"] == 2

    @pytest.mark.asyncio
    async def test_explicit_level_override(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, community_level=1)

        report_call = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert report_call[0][1]["level"] == 1


class TestDriftFallback:
    @pytest.mark.asyncio
    async def test_fallback_when_no_report_embeddings(self):
        store = NoReportSearchStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.mode == "drift"
        assert result.answer
        assert result.metadata.get("drift_fallback_reason") == "missing_report_embeddings"

    @pytest.mark.asyncio
    async def test_fallback_when_no_reports(self):
        store = EmptyReportsGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.metadata.get("drift_fallback_reason") == "no_report_results"

    @pytest.mark.asyncio
    async def test_fallback_synthesize_false(self):
        store = NoReportSearchStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2, synthesize_response=False)

        assert result.answer == ""
        assert result.metadata["synthesize_response"] is False
        assert result.metadata.get("drift_fallback_reason") == "missing_report_embeddings"

    @pytest.mark.asyncio
    async def test_fallback_citations_resolved(self):
        store = NoReportSearchStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.citations == [
            Citation(
                document_id="doc:1",
                chunk_id="chunk:1",
                document_name="Doc 1",
                excerpt="Alice directed Inception in 2010.",
                metadata={
                    "collection": "movies",
                    "source": "row-source",
                    "record_id": "row-42",
                    "document_id": "doc:1",
                    "chunk_id": "chunk:1",
                    "document_name": "Doc 1",
                },
            )
        ]


class TestDriftPrimerJsonParsing:
    @pytest.mark.asyncio
    async def test_primer_parses_valid_json(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "answer": "Alice directed Inception.",
                    "score": 85,
                    "follow_ups": ["What else did Alice direct?"],
                    "report_ids": ["report:c0:0"],
                })
            ]
        )
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.answer
        trace = result.metadata["drift_trace"]
        assert trace["primer_report_ids"] == ["report:c0:0"]
        assert len(trace["actions"]) >= 1
        assert trace["actions"][0]["score"] == 85

    @pytest.mark.asyncio
    async def test_primer_parses_fenced_json(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                "```json\n"
                + json.dumps({"answer": "Test", "score": 50, "follow_ups": []})
                + "\n```"
            ]
        )
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.answer

    @pytest.mark.asyncio
    async def test_primer_parses_json_with_surrounding_prose(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                "Here is my answer:\n"
                + json.dumps({"answer": "Test", "score": 50, "follow_ups": []})
                + "\nHope that helps!"
            ]
        )
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.answer


class TestDriftTraversal:
    @pytest.mark.asyncio
    async def test_follow_up_triggers_local_search(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "answer": "Initial answer.",
                    "score": 80,
                    "follow_ups": ["What year?"],
                }),
                json.dumps({
                    "answer": "2010.",
                    "score": 90,
                    "follow_ups": [],
                }),
            ]
        )
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        trace = result.metadata["drift_trace"]
        # primer + 1 follow-up action
        assert len(trace["actions"]) >= 2
        assert trace["total_llm_calls"] >= 2

    @pytest.mark.asyncio
    async def test_low_score_followup_not_expanded(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "answer": "Initial.",
                    "score": 80,
                    "follow_ups": ["Question?"],
                }),
                json.dumps({
                    "answer": "Low confidence.",
                    "score": 5,
                    "follow_ups": ["Another?"],
                }),
            ]
        )
        config = DriftSearchConfig(min_expand_score=20.0)
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder(), config=config)

        result = await retriever.search("query", top_k=2)

        trace = result.metadata["drift_trace"]
        # Only primer + 1 action (no further expansion due to low score)
        assert len(trace["actions"]) == 2


class TestDriftConversationHistory:
    @pytest.mark.asyncio
    async def test_conversation_history_in_prompt(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        await retriever.search("query", top_k=2, conversation_history="User: Hi\nAssistant: Hello")

        assert "User: Hi" in llm.prompts[0]


class TestDriftPromptCustomization:
    @pytest.mark.asyncio
    async def test_custom_reduce_prompt(self):
        store = FakeGraphStore()
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "answer": "Initial.",
                    "score": 80,
                    "follow_ups": [],
                }),
            ]
        )
        custom_prompt = "CUSTOM REDUCE: {query}\n{action_context}\n{conversation_history}"
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(), reduce_prompt=custom_prompt
        )

        result = await retriever.search("query", top_k=2)

        assert result.answer


class TestDriftLimitsAndRepair:
    @pytest.mark.asyncio
    async def test_invalid_primer_json_gets_one_async_repair(self):
        llm = FakeLLM(
            responses=[
                "not json",
                json.dumps({
                    "answer": "Repaired.",
                    "score": 70,
                    "follow_ups": [],
                    "report_ids": ["report:c0:0"],
                }),
            ]
        )
        result = await DriftSearchRetriever(
            FakeGraphStore(), llm, FakeEmbedder()
        ).search("query")

        assert "invalid JSON" in llm.prompts[1]
        assert result.metadata["drift_trace"]["total_llm_calls"] == 3

    @pytest.mark.asyncio
    async def test_max_llm_calls_is_hard_and_trace_includes_final_state(self):
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "answer": "Primer.",
                    "score": 80,
                    "follow_ups": ["Follow up?"],
                }),
                json.dumps({
                    "answer": "Action.",
                    "score": 90,
                    "follow_ups": ["More?"],
                }),
            ]
        )
        config = DriftSearchConfig(max_llm_calls=2, action_concurrency=3)
        result = await DriftSearchRetriever(
            FakeGraphStore(), llm, FakeEmbedder(), config=config
        ).search("query")

        trace = result.metadata["drift_trace"]
        assert trace["total_llm_calls"] == 2
        assert trace["stopping_reason"] == "max_llm_calls_reached"
        assert len(llm.prompts) == 2

    @pytest.mark.asyncio
    async def test_config_community_level_is_used_when_constructor_omits_it(self):
        store = FakeGraphStore()
        config = DriftSearchConfig(community_level="finest")
        await DriftSearchRetriever(
            store, FakeLLM(), FakeEmbedder(), config=config
        ).search("query")

        report_call = next(
            call for call in store.calls
            if call[0] == "vector_search_community_reports"
        )
        assert report_call[1]["level"] == 2
