"""Tests for mixed-context builder used in local search."""

from __future__ import annotations

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.local import LocalSearchRetriever
from recon_graphrag.retrieval.mixed_context import MixedContextBuilder


class FakeGraphStore:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

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
                "entity_id": "a",
                "entity_name": "Alice",
                "entity_labels": ["__Entity__", "Person"],
                "entity_description": "CEO of Acme",
                "relationships": ["Person: Alice -[WORKS_AT]-> Organization: Acme"],
                "relationship_records": [{
                    "source_id": "a",
                    "source_name": "Alice",
                    "target_id": "org:acme",
                    "target_name": "Acme",
                    "rel": "WORKS_AT",
                    "description": "Alice leads Acme",
                    "weight": 4.0,
                }],
                "source_text": ["Alice directed Inception."],
                "source_chunk_ids": ["chunk:1"],
                "score": 0.8,
            }
        ]

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append(("execute_query", {"query": query.strip()[:60], "params": params}))

        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]
        if "relationship_keys" in params:
            return [{"chunk_id": "chunk:1"}]
        if "claim_ids" in params:
            return [{"chunk_id": "chunk:1"}]
        if "Chunk" in query and "FROM_CHUNK" in query:
            return [
                {
                    "chunk_id": "chunk:1",
                    "text": "Alice is the CEO of Acme Corp.",
                    "linked_entities": ["a"],
                }
            ]
        if "Community" in query and "IN_COMMUNITY" in query:
            return [
                {
                    "community_id": "c0",
                    "level": 0,
                    "report_text": "Alice leads Acme Corp in the tech sector.",
                    "rating": 8.0,
                }
            ]
        return []

    def get_claims_for_entities(self, graph_name, entity_ids):
        self.calls.append(("get_claims_for_entities", {"entity_ids": entity_ids}))
        return [
            {
                "claim_id": "claim:1",
                "entity_id": "a",
                "claim_type": "role",
                "description": "Alice is the CEO of Acme",
            }
        ]

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.calls.append(("resolve_chunk_citations", {"chunk_ids": chunk_ids}))
        return [
            {
                "chunk_id": "chunk:1",
                "document_id": "doc:1",
                "document_name": "Doc 1",
                "excerpt": "Alice is the CEO of Acme Corp.",
            }
        ]


class FakeEmbedder:
    async def async_embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(content="Alice is the CEO of Acme Corp.")


class TestMixedContextBuilder:
    def test_build_context_returns_all_candidate_types(self):
        store = FakeGraphStore()
        builder = MixedContextBuilder(store, graph_name="entity-graph")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "entity_id": "a",
                    "entity_name": "Alice",
                    "entity_labels": ["__Entity__", "Person"],
                    "entity_description": "CEO of Acme",
                    "relationships": ["Person: Alice -[WORKS_AT]-> Organization: Acme"],
                    "relationship_records": [{
                        "source_id": "a",
                        "source_name": "Alice",
                        "target_id": "org:acme",
                        "target_name": "Acme",
                        "rel": "WORKS_AT",
                        "description": "Alice leads Acme",
                        "weight": 4.0,
                    }],
                    "source_text": ["Alice directed Inception."],
                    "source_chunk_ids": ["chunk:1"],
                    "score": 0.8,
                }
            ],
            token_budget=12000,
        )

        assert result.context
        assert "Graph Facts" in result.context
        assert "Source Text" in result.context
        assert "Community Reports" in result.context
        assert "Alice" in result.context
        assert "Alice leads Acme" in result.context
        assert "chunk:1" in result.included_chunk_ids
        assert "c0" in result.included_community_ids
        assert "claim:1" in result.included_claim_ids

    def test_build_context_resolves_citations(self):
        store = FakeGraphStore()
        builder = MixedContextBuilder(store, graph_name="entity-graph")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "relationships": [],
                    "source_text": [],
                    "source_chunk_ids": ["chunk:1"],
                    "score": 0.8,
                }
            ],
            token_budget=12000,
        )

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk:1"

    def test_build_context_respects_token_budget(self):
        store = FakeGraphStore()
        builder = MixedContextBuilder(store, graph_name="entity-graph")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "relationships": ["Person: Alice -[WORKS_AT]-> Organization: Acme"],
                    "source_text": ["Alice directed Inception."],
                    "source_chunk_ids": ["chunk:1"],
                    "score": 0.8,
                }
            ],
            token_budget=50,
        )

        assert result.used_tokens <= 50 + 10  # small rounding tolerance

    def test_build_context_with_no_communities(self):
        class NoCommStore(FakeGraphStore):
            def execute_query(self, query, parameters=None):
                params = parameters or {}
                if "RETURN max(c.level) AS level" in query:
                    return [{"level": None}]
                if "Chunk" in query:
                    return [{"chunk_id": "chunk:1", "text": "Some text", "linked_entities": ["a"]}]
                return []

        store = NoCommStore()
        builder = MixedContextBuilder(store, graph_name="entity-graph")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "relationships": [],
                    "source_text": [],
                    "source_chunk_ids": ["chunk:1"],
                    "score": 0.8,
                }
            ],
            token_budget=12000,
        )

        assert result.context
        assert result.included_community_ids == []

    def test_build_context_with_no_claims(self):
        class NoClaimStore(FakeGraphStore):
            def get_claims_for_entities(self, graph_name, entity_ids):
                return []

        store = NoClaimStore()
        builder = MixedContextBuilder(store, graph_name="entity-graph")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "relationships": [],
                    "source_text": [],
                    "source_chunk_ids": [],
                    "score": 0.8,
                }
            ],
            token_budget=12000,
        )

        assert result.included_claim_ids == []


class TestLocalSearchMixedContext:
    @pytest.mark.asyncio
    async def test_mixed_context_enabled(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(
            store, llm, FakeEmbedder(), use_mixed_context=True
        )

        result = await retriever.search("Who is Alice?", top_k=1, token_budget=12000)

        assert result.mode == "local"
        assert result.answer == "Alice is the CEO of Acme Corp."
        assert result.metadata.get("mixed_context") is True
        assert "used_tokens" in result.metadata

    @pytest.mark.asyncio
    async def test_mixed_context_synthesize_false(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(
            store, llm, FakeEmbedder(), use_mixed_context=True
        )

        result = await retriever.search(
            "Who is Alice?", top_k=1, synthesize_response=False, token_budget=12000
        )

        assert result.answer == ""
        assert result.metadata["synthesize_response"] is False
        assert result.metadata.get("mixed_context") is True
        assert llm.prompts == []

    @pytest.mark.asyncio
    async def test_mixed_context_disabled_uses_original(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(
            store, llm, FakeEmbedder(), use_mixed_context=False
        )

        result = await retriever.search("Who is Alice?", top_k=1)

        assert result.mode == "local"
        assert result.metadata.get("mixed_context") is None
        assert "Finding: Alice (Person)" in result.context
