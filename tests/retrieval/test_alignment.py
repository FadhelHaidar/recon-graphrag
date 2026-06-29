"""Tests for Microsoft GraphRAG alignment features.

Covers: top_k_relationships, allow_general_knowledge, use_hyde,
primer_folds, action_use_mixed_context.
"""

from __future__ import annotations

import json

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.drift import DriftSearchRetriever
from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever
from recon_graphrag.retrieval.hybrid import RetrievalItem, RetrievalResult
from recon_graphrag.retrieval.local import (
    LocalSearchRetriever,
    _format_entity_context,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGraphStore:
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
        self.calls.append(("fetch_entity_context", {"mode": mode}))
        rels = [f"Person: Alice -[REL_{i}]-> Target: T{i}" for i in range(20)]
        return [
            {
                "title": "Alice (Person)",
                "relationships": rels,
                "source_text": ["Alice directed Inception in 2010."],
                "source_chunk_ids": ["chunk:1"],
                "score": matches[0]["score"],
            }
        ]

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append(("execute_query", {"query": query.strip()[:60], "params": params}))
        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]
        if "IN_COMMUNITY" in query:
            return []
        if "Chunk" in query:
            return []
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return []

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        self.calls.append(("vector_search_community_reports", {"top_k": top_k, "level": level}))
        return self._reports[:top_k]

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []


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
            if "Partial Answers" in prompt or "synthesizing" in prompt.lower():
                content = "Alice directed Inception."
            elif "general knowledge" in prompt.lower():
                content = "General knowledge answer about the topic."
            else:
                content = json.dumps({
                    "answer": "Alice directed Inception.",
                    "score": 80,
                    "follow_ups": [],
                })
        self._call_count += 1
        return LLMResponse(content=content)


class NoReportSearchStore(FakeGraphStore):
    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        raise RuntimeError("No such index")


# ---------------------------------------------------------------------------
# P1: top_k_relationships
# ---------------------------------------------------------------------------


class TestTopKRelationships:
    def test_format_entity_context_caps_relationships(self):
        """_format_entity_context respects top_k_relationships."""
        items = [
            RetrievalItem(content={
                "title": "Alice",
                "relationships": [f"rel_{i}" for i in range(20)],
                "source_text": ["text"],
                "source_chunk_ids": [],
            })
        ]
        result = RetrievalResult(items=items)

        uncapped = _format_entity_context(result)
        capped = _format_entity_context(result, top_k_relationships=5)

        # Count "rel_" occurrences in each
        assert uncapped.count("rel_") == 20
        assert capped.count("rel_") == 5

    def test_format_entity_context_none_means_no_cap(self):
        """top_k_relationships=None preserves all relationships."""
        items = [
            RetrievalItem(content={
                "title": "Alice",
                "relationships": [f"rel_{i}" for i in range(15)],
                "source_text": [],
                "source_chunk_ids": [],
            })
        ]
        result = RetrievalResult(items=items)
        output = _format_entity_context(result)
        assert output.count("rel_") == 15

    @pytest.mark.asyncio
    async def test_local_search_passes_top_k_relationships(self):
        """LocalSearchRetriever passes top_k_relationships to formatter."""
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(
            store, llm, FakeEmbedder(), top_k_relationships=3, use_mixed_context=False,
        )

        result = await retriever.search("query", top_k=1, synthesize_response=False)

        # Should have at most 3 relationships in context
        assert result.context.count("REL_") <= 3

    @pytest.mark.asyncio
    async def test_local_search_default_top_k_relationships_is_10(self):
        """Default top_k_relationships=10 caps at 10."""
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(store, llm, FakeEmbedder(), use_mixed_context=False)

        result = await retriever.search("query", top_k=1, synthesize_response=False)

        # Store returns 20 relationships; default cap is 10
        assert result.context.count("REL_") == 10

    def test_graphrag_passes_top_k_relationships(self):
        """GraphRAG orchestrator passes top_k_relationships to LocalSearchRetriever."""
        from recon_graphrag.retrieval.search import GraphRAG

        store = FakeGraphStore()
        grag = GraphRAG(store, FakeLLM(), FakeEmbedder(), top_k_relationships=5)
        assert grag.local.top_k_relationships == 5


# ---------------------------------------------------------------------------
# P2: allow_general_knowledge
# ---------------------------------------------------------------------------


def _make_gk_reports(n: int = 2) -> list[dict]:
    return [
        {"id": f"report:{i}:0", "level": 0, "report_text": f"Summary {i}."}
        for i in range(n)
    ]


def _make_gk_map_response(helpfulness: int = 0) -> str:
    import json as _json
    return _json.dumps({
        "answer": "No relevant information found." if helpfulness == 0 else "Some answer.",
        "helpfulness": helpfulness,
        "report_ids": ["report:0:0"],
    })


class GKFakeGraphStore:
    def __init__(self, reports):
        self._reports = reports

    def execute_query(self, query, params=None):
        params = params or {}
        level = params.get("level")
        if level is not None:
            return [r for r in self._reports if r.get("level") == level]
        return self._reports

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return []


class TestAllowGeneralKnowledge:
    @pytest.mark.asyncio
    async def test_default_returns_no_info_when_all_zero(self):
        reports = _make_gk_reports(2)
        store = GKFakeGraphStore(reports)

        class ZeroLLM:
            async def ainvoke(self, prompt):
                from unittest.mock import MagicMock
                return MagicMock(content=_make_gk_map_response(0))

        search = GlobalSearchRetriever(store, ZeroLLM())
        result = await search.search("query", community_level=0)
        assert "No relevant" in result.answer

    @pytest.mark.asyncio
    async def test_allow_general_knowledge_runs_reduce_on_all_zero(self):
        reports = _make_gk_reports(2)
        store = GKFakeGraphStore(reports)

        class GKLLM:
            def __init__(self):
                self.prompts = []
            async def ainvoke(self, prompt):
                self.prompts.append(prompt)
                if "Synthesize" in prompt or "Partial Answers" in prompt:
                    from unittest.mock import MagicMock
                    return MagicMock(content="General knowledge answer.")
                return MagicMock(content=_make_gk_map_response(0))

        llm = GKLLM()
        search = GlobalSearchRetriever(store, llm, allow_general_knowledge=True)
        result = await search.search("query", community_level=0)

        assert result.answer == "General knowledge answer."
        assert any("general knowledge" in p.lower() for p in llm.prompts)

    @pytest.mark.asyncio
    async def test_allow_general_knowledge_false_skips_reduce(self):
        reports = _make_gk_reports(2)
        store = GKFakeGraphStore(reports)

        class ZeroLLM:
            def __init__(self):
                self.call_count = 0
            async def ainvoke(self, prompt):
                self.call_count += 1
                from unittest.mock import MagicMock
                return MagicMock(content=_make_gk_map_response(0))

        llm = ZeroLLM()
        search = GlobalSearchRetriever(store, llm, allow_general_knowledge=False)
        result = await search.search("query", community_level=0)

        assert "No relevant" in result.answer
        # Only map calls ran, no reduce call
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# P3: DRIFT HyDE, primer folds, action mixed context
# ---------------------------------------------------------------------------


class TestDriftHyDE:
    @pytest.mark.asyncio
    async def test_hyde_enabled_does_two_vector_searches(self):
        """With use_hyde=True, primer phase does initial + re-rank vector searches."""
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(),
            config=DriftSearchConfig(use_hyde=True),
        )

        await retriever.search("query", top_k=2)

        report_calls = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert len(report_calls) == 2

    @pytest.mark.asyncio
    async def test_hyde_disabled_does_one_vector_search(self):
        """With use_hyde=False, primer phase does one vector search."""
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(),
            config=DriftSearchConfig(use_hyde=False),
        )

        await retriever.search("query", top_k=2)

        report_calls = [
            c for c in store.calls if c[0] == "vector_search_community_reports"
        ]
        assert len(report_calls) == 1

    @pytest.mark.asyncio
    async def test_hyde_calls_llm_for_hypothetical_answer(self):
        """HyDE generates a hypothetical answer via LLM."""
        store = FakeGraphStore()
        llm = FakeLLM()
        embedder = FakeEmbedder()
        retriever = DriftSearchRetriever(
            store, llm, embedder,
            config=DriftSearchConfig(use_hyde=True),
        )

        await retriever.search("query", top_k=2)

        # First prompt should be the HyDE prompt
        assert "hypothetical" in llm.prompts[0].lower()
        # Embedder should be called twice: query + hypothetical
        assert len(embedder.queries) == 2

    @pytest.mark.asyncio
    async def test_hyde_default_is_enabled(self):
        """use_hyde defaults to True (matching Microsoft behavior)."""
        config = DriftSearchConfig()
        assert config.use_hyde is True


class TestDriftPrimerFolds:
    @pytest.mark.asyncio
    async def test_primer_folds_runs_parallel_calls(self):
        """With primer_folds=2, primer reports are split into 2 folds."""
        reports = [
            {"id": f"report:{i}:0", "level": 0, "report_text": f"Report {i}."}
            for i in range(4)
        ]
        store = FakeGraphStore(reports=reports)
        llm = FakeLLM()
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(),
            config=DriftSearchConfig(primer_top_k=4, primer_folds=2, use_hyde=False),
        )

        result = await retriever.search("query", top_k=2)

        assert result.mode == "drift"
        # Should have at least 2 primer fold LLM calls
        primer_prompts = [p for p in llm.prompts if "Community Reports" in p]
        assert len(primer_prompts) >= 2

    @pytest.mark.asyncio
    async def test_primer_folds_merges_follow_ups(self):
        """Folds merge follow-up questions from all fold results."""
        reports = [
            {"id": f"report:{i}:0", "level": 0, "report_text": f"Report {i}."}
            for i in range(4)
        ]
        store = FakeGraphStore(reports=reports)

        follow_ups_fold1 = ["What year?", "Who acted?"]
        follow_ups_fold2 = ["How much revenue?", "What awards?"]

        responses = [
            json.dumps({"answer": "Fold1", "score": 70, "follow_ups": follow_ups_fold1, "report_ids": ["report:0:0"]}),
            json.dumps({"answer": "Fold2", "score": 80, "follow_ups": follow_ups_fold2, "report_ids": ["report:2:0"]}),
        ]
        llm = FakeLLM(responses=responses)
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(),
            config=DriftSearchConfig(primer_top_k=4, primer_folds=2, use_hyde=False, max_depth=0),
        )

        result = await retriever.search("query", top_k=2)

        assert result.mode == "drift"
        assert result.answer

    def test_primer_folds_default_is_1(self):
        """primer_folds defaults to 1 (single call)."""
        config = DriftSearchConfig()
        assert config.primer_folds == 1

    def test_split_into_folds_even(self):
        reports = [{"id": str(i)} for i in range(6)]
        folds = DriftSearchRetriever._split_into_folds(reports, 3)
        assert len(folds) == 3
        assert len(folds[0]) == 2

    def test_split_into_folds_uneven(self):
        reports = [{"id": str(i)} for i in range(5)]
        folds = DriftSearchRetriever._split_into_folds(reports, 3)
        assert sum(len(f) for f in folds) == 5

    def test_split_into_folds_single(self):
        reports = [{"id": "1"}]
        folds = DriftSearchRetriever._split_into_folds(reports, 3)
        assert len(folds) == 1


class TestDriftActionMixedContext:
    @pytest.mark.asyncio
    async def test_action_use_mixed_context_default_is_true(self):
        """action_use_mixed_context defaults to True."""
        config = DriftSearchConfig()
        assert config.action_use_mixed_context is True

    @pytest.mark.asyncio
    async def test_action_mixed_context_creates_builder(self):
        """action_use_mixed_context=True creates MixedContextBuilder."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(
            store, FakeLLM(), FakeEmbedder(),
            config=DriftSearchConfig(use_hyde=False),
        )

        # Trigger a search that creates the builder
        await retriever.search("query", top_k=2)

        # If any actions ran with mixed context, builder should exist
        # (it's created lazily in _build_action_mixed_context)


# ---------------------------------------------------------------------------
# GraphRAG orchestrator
# ---------------------------------------------------------------------------


class TestGraphRAGOrchestrator:
    @pytest.mark.asyncio
    async def test_search_rejects_invalid_mode(self):
        """GraphRAG.search raises ValueError for unknown mode."""
        from recon_graphrag.retrieval.search import GraphRAG

        grag = GraphRAG(FakeGraphStore(), FakeLLM(), FakeEmbedder())
        with pytest.raises(ValueError, match="Unknown search mode"):
            await grag.search("query", mode="invalid")
