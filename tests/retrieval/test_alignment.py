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
from recon_graphrag.retrieval.global_search import (
    GlobalSearchRetriever,
    PartialAnswer,
)
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
