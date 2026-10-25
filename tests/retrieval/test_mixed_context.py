"""Tests for mixed-context builder used in local search."""

from __future__ import annotations

from typing import Any, cast

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.search_local import LocalSearchRetriever
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
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")

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

    def test_queries_match_all_entity_id_forms(self):
        """entity_matches carry human-readable ids while e.id is a UUID; the
        community and text-unit queries must match on every id form."""

        class IdFormStore(FakeGraphStore):
            def execute_query(self, query, parameters=None):
                if "IN_COMMUNITY" in query:
                    assert "e.human_readable_id = eid" in query
                    assert "e.canonical_key = eid" in query
                if "linked_entities" in query:
                    assert "e.human_readable_id" in query
                return super().execute_query(query, parameters)

        store = IdFormStore()
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")
        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=[
                {
                    "title": "Alice (Person)",
                    "entity_id": "a",
                    "relationships": [],
                    "source_text": [],
                    "source_chunk_ids": ["chunk:1"],
                    "score": 0.8,
                }
            ],
            token_budget=12000,
        )
        assert "c0" in result.included_community_ids

    def test_build_context_resolves_citations(self):
        store = FakeGraphStore()
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")

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
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")

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
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")

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
        builder = MixedContextBuilder(cast(Any, store), graph_name="entity-graph")

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
        retriever = LocalSearchRetriever(cast(Any, store), cast(Any, llm), cast(Any, FakeEmbedder()), use_mixed_context=True)

        result = await retriever.search("Who is Alice?", top_k=1, token_budget=12000)

        assert result.mode == "local"
        assert result.answer == "Alice is the CEO of Acme Corp."
        assert result.metadata.get("mixed_context") is True
        assert "used_tokens" in result.metadata

    @pytest.mark.asyncio
    async def test_mixed_context_synthesize_false(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = LocalSearchRetriever(cast(Any, store), cast(Any, llm), cast(Any, FakeEmbedder()), use_mixed_context=True)

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
        retriever = LocalSearchRetriever(cast(Any, store), cast(Any, llm), cast(Any, FakeEmbedder()), use_mixed_context=False)

        result = await retriever.search("Who is Alice?", top_k=1)

        assert result.mode == "local"
        assert result.metadata.get("mixed_context") is None
        assert "Finding: Alice (Person)" in result.context


class BrokenChunkCitationStore(FakeGraphStore):
    def resolve_chunk_citations(self, graph_name, chunk_ids):
        raise RuntimeError("driver said: password=hunter2")


class MetadataCitationStore(FakeGraphStore):
    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return [
            {
                "chunk_id": "chunk:1",
                "document_id": "doc:1",
                "document_name": "Doc 1",
                "excerpt": "Alice is the CEO of Acme Corp.",
                "chunk_metadata": {"record_id": "row-42", "source": "row-source"},
            }
        ]


ENTITY_ROWS = [
    {
        "title": "Alice (Person)",
        "entity_id": "a",
        "entity_name": "Alice",
        "entity_labels": ["__Entity__", "Person"],
        "relationships": [],
        "source_text": ["Alice directed Inception."],
        "source_chunk_ids": ["chunk:1"],
        "score": 0.8,
    }
]


class TestCitationDegradation:
    def test_build_context_records_chunk_citation_failure(self):
        builder = MixedContextBuilder(cast(Any, BrokenChunkCitationStore()), graph_name="g")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=ENTITY_ROWS,
            token_budget=12000,
        )

        assert result.citations == []
        assert result.citation_warnings == [
            {"code": "citation_resolution_failed", "stage": "chunk"},
            {"code": "citation_resolution_failed", "stage": "reference"},
        ]
        assert "hunter2" not in str(result.citation_warnings)
        assert result.context

    def test_build_context_records_reference_citation_failure(self, monkeypatch):
        import recon_graphrag.retrieval.mixed_context as mixed_context

        def explode(*args, **kwargs):
            raise RuntimeError("driver said: password=hunter2")

        monkeypatch.setattr(
            mixed_context, "resolve_reference_citations", explode
        )
        builder = MixedContextBuilder(cast(Any, FakeGraphStore()), graph_name="g")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=ENTITY_ROWS,
            token_budget=12000,
        )

        # Chunk citations survive; only the reference lookup degraded.
        assert [c.chunk_id for c in result.citations] == ["chunk:1"]
        assert len(result.citation_warnings) == 1
        assert result.citation_warnings[0] == {
            "code": "citation_resolution_failed", "stage": "reference"
        }
        assert "hunter2" not in str(result.citation_warnings)

    def test_build_context_has_no_warnings_when_resolution_succeeds(self):
        builder = MixedContextBuilder(cast(Any, FakeGraphStore()), graph_name="g")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=ENTITY_ROWS,
            token_budget=12000,
        )

        assert result.citation_warnings == []


class TestMixedCitationMetadataOption:
    def test_metadata_absent_by_default(self):
        builder = MixedContextBuilder(cast(Any, MetadataCitationStore()), graph_name="g")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=ENTITY_ROWS,
            token_budget=12000,
        )

        assert "Citation metadata" not in result.context

    def test_metadata_appended_with_key_filter(self):
        builder = MixedContextBuilder(cast(Any, MetadataCitationStore()), graph_name="g")

        result = builder.build_context(
            entity_matches=[{"id": "a", "score": 0.8}],
            entity_context_rows=ENTITY_ROWS,
            token_budget=12000,
            synthesize_citation_metadata=True,
            synthesis_metadata_keys=["record_id"],
        )

        assert "=== Citation metadata ===" in result.context
        assert '"record_id": "row-42"' in result.context
        assert "row-source" not in result.context


class TestLocalSearchMixedDegradation:
    @pytest.mark.asyncio
    async def test_mixed_search_surfaces_citation_warning(self):
        retriever = LocalSearchRetriever(cast(Any, BrokenChunkCitationStore()), cast(Any, FakeLLM()), cast(Any, FakeEmbedder()), use_mixed_context=True,)

        result = await retriever.search("Who is Alice?", top_k=1, token_budget=12000)

        assert result.citations == []
        assert result.metadata["citation_warnings"][0] == {
            "code": "citation_resolution_failed", "stage": "chunk"
        }
        assert "hunter2" not in str(result.metadata)
        assert result.answer

    @pytest.mark.asyncio
    async def test_mixed_search_forwards_citation_metadata_options(self):
        llm = FakeLLM()
        retriever = LocalSearchRetriever(cast(Any, MetadataCitationStore()), cast(Any, llm), cast(Any, FakeEmbedder()), use_mixed_context=True)

        result = await retriever.search(
            "Who is Alice?",
            top_k=1,
            token_budget=12000,
            synthesize_citation_metadata=True,
            synthesis_metadata_keys=["record_id"],
        )

        assert "=== Citation metadata ===" in result.context
        assert '"record_id": "row-42"' in result.context
        # The context handed back must be exactly what the LLM saw.
        assert '"record_id": "row-42"' in llm.prompts[0]
        assert result.context in llm.prompts[0]
