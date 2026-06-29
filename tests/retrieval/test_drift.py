"""Characterization tests for DRIFT search.

These tests document the current (pre-Phase 6) DRIFT behavior and serve as
regression guards.  Target-behavior tests that describe the Phase 6 iterative
DRIFT design are marked ``xfail`` until the rewrite lands.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.models.artifacts import Citation
from recon_graphrag.retrieval.drift import DriftSearchRetriever
from recon_graphrag.retrieval.community_levels import resolve_community_level


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGraphStore:
    """Minimal fake store matching the pattern in test_hybrid.py."""

    def __init__(self, *, communities=None, bridging_entities=None):
        self.calls: list[tuple[str, dict]] = []
        self._communities = communities or [
            {"id": "c0", "level": 0, "summary": "Fine community summary"},
            {"id": "c2", "level": 2, "summary": "Coarse community summary"},
        ]
        self._bridging_entities = bridging_entities or [
            {
                "name": "RelatedEntity",
                "labels": ["__Entity__", "Person"],
                "rels": ["Connected to: SomeOrg"],
            }
        ]

    # -- Search methods --

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        self.calls.append(("vector_search", {"index_name": index_name, "k": k}))
        return [
            {"id": "a", "score": 0.8},
            {"id": "b", "score": 0.6},
        ]

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        self.calls.append(("keyword_search", {"index_name": index_name, "k": k}))
        return [
            {"id": "a", "score": 1.0},
        ]

    def fetch_entity_context(
        self, matches, retrieval_query=None, query_params=None, mode="local"
    ):
        self.calls.append(
            ("fetch_entity_context", {"matches": matches, "mode": mode})
        )
        return [
            {
                "title": "Alice (Person)",
                "relationships": ["Person: Alice -[DIRECTED]-> Movie: Inception"],
                "source_text": ["Alice directed Inception in 2010."],
                "source_chunk_ids": ["chunk:1"],
                "score": matches[0]["score"],
                "communities": [
                    {
                        "id": "c0",
                        "level": 0,
                        "graph_name": "entity-graph",
                        "summary": "Fine community summary",
                    },
                    {
                        "id": "c2",
                        "level": 2,
                        "graph_name": "entity-graph",
                        "summary": "Coarse community summary",
                    },
                ],
            }
        ]

    # -- Community methods --

    def get_community_summaries_by_keys(self, graph_name, keys, top_k):
        self.calls.append(
            (
                "get_community_summaries_by_keys",
                {"graph_name": graph_name, "keys": keys, "top_k": top_k},
            )
        )
        result = []
        for comm in self._communities:
            for key in keys:
                if comm["id"] == key["id"] and comm["level"] == key["level"]:
                    result.append(comm)
        return result[:top_k]

    def get_community_entities_by_keys(self, graph_name, keys):
        self.calls.append(
            (
                "get_community_entities_by_keys",
                {"graph_name": graph_name, "keys": keys},
            )
        )
        return self._bridging_entities

    # -- Query / citation methods --

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append(
            ("execute_query", {"query": query.strip(), "params": params})
        )
        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]
        if "RETURN min(c.level) AS level" in query:
            return [{"level": 0}]
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.calls.append(
            ("resolve_chunk_citations", {"graph_name": graph_name, "chunk_ids": chunk_ids})
        )
        return [
            {
                "document_id": "doc:1",
                "chunk_id": "chunk:1",
                "document_name": "Doc 1",
                "page_start": None,
                "page_end": None,
                "document_metadata": {"collection": "movies", "source": "dataset-a"},
                "chunk_metadata": {
                    "record_id": "row-42",
                    "source": "row-source",
                    "embedding": [0.0],
                    "graph_name": graph_name,
                },
                "excerpt": "Alice directed Inception in 2010.",
            }
        ]


class EmptyCommunitiesGraphStore(FakeGraphStore):
    """Fake store that returns no communities."""

    def __init__(self):
        super().__init__(communities=[], bridging_entities=[])

    def get_community_summaries_by_keys(self, graph_name, keys, top_k):
        self.calls.append(
            (
                "get_community_summaries_by_keys",
                {"graph_name": graph_name, "keys": keys, "top_k": top_k},
            )
        )
        return []

    def get_community_entities_by_keys(self, graph_name, keys):
        self.calls.append(
            (
                "get_community_entities_by_keys",
                {"graph_name": graph_name, "keys": keys},
            )
        )
        return []


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    async def async_embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeLLM:
    def __init__(self, answer: str = "Alice directed Inception."):
        self.prompts: list[str] = []
        self._answer = answer

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(content=self._answer)


# ---------------------------------------------------------------------------
# Compatibility tests — document current behavior
# ---------------------------------------------------------------------------


class TestDriftSearchCompatibility:
    """Baseline tests for current DRIFT search behavior."""

    @pytest.mark.asyncio
    async def test_drift_returns_valid_search_result(self):
        """mode='drift' returns a SearchResult with answer, context, citations."""
        store = FakeGraphStore()
        embedder = FakeEmbedder()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, embedder)

        result = await retriever.search("Who directed Inception?", top_k=2)

        assert result.mode == "drift"
        assert result.query == "Who directed Inception?"
        assert result.answer == "Alice directed Inception."
        assert result.context
        assert "Alice" in result.context
        assert isinstance(result.citations, list)

    @pytest.mark.asyncio
    async def test_drift_entity_context_in_result(self):
        """Entity findings appear in the context."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert "Finding: Alice (Person)" in result.context
        assert "Alice directed Inception in 2010." in result.context

    @pytest.mark.asyncio
    async def test_drift_community_context_in_result(self):
        """Community summaries appear in the context."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2, community_level=0)

        assert "Fine community summary" in result.context

    @pytest.mark.asyncio
    async def test_drift_bridging_entities_in_result(self):
        """Bridging entities appear in the context."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert "RelatedEntity" in result.context

    @pytest.mark.asyncio
    async def test_drift_citations_resolved(self):
        """Citations are resolved from entity source chunk IDs."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2, community_top_k=2)

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

    @pytest.mark.asyncio
    async def test_drift_embedder_called(self):
        """The embedder is called with the query text."""
        store = FakeGraphStore()
        embedder = FakeEmbedder()
        retriever = DriftSearchRetriever(store, FakeLLM(), embedder)

        await retriever.search("Who directed Inception?", top_k=2)

        assert embedder.queries == ["Who directed Inception?"]

    @pytest.mark.asyncio
    async def test_drift_uses_context_mode_drift(self):
        """HybridEntityRetriever is created with context_mode='drift'."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2)

        ctx_call = [c for c in store.calls if c[0] == "fetch_entity_context"][0]
        assert ctx_call[1]["mode"] == "drift"

    @pytest.mark.asyncio
    async def test_drift_community_top_k_limits_results(self):
        """community_top_k is passed to get_community_summaries_by_keys."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, community_top_k=1)

        summary_call = [
            c for c in store.calls if c[0] == "get_community_summaries_by_keys"
        ][0]
        assert summary_call[1]["top_k"] == 1


class TestDriftSynthesisFalse:
    """Tests for synthesize_response=False."""

    @pytest.mark.asyncio
    async def test_synthesize_false_returns_empty_answer(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search(
            "query", top_k=2, synthesize_response=False
        )

        assert result.answer == ""
        assert llm.prompts == []

    @pytest.mark.asyncio
    async def test_synthesize_false_returns_context(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search(
            "query", top_k=2, synthesize_response=False
        )

        assert result.context
        assert "Alice" in result.context

    @pytest.mark.asyncio
    async def test_synthesize_false_returns_citations(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search(
            "query", top_k=2, synthesize_response=False
        )

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

    @pytest.mark.asyncio
    async def test_synthesize_false_metadata(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search(
            "query", top_k=2, synthesize_response=False
        )

        assert result.metadata["synthesize_response"] is False
        assert result.metadata["response_synthesis_skipped"] is True


class TestDriftCitationMetadata:
    """Tests for synthesize_citation_metadata."""

    @pytest.mark.asyncio
    async def test_citation_metadata_in_context(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        result = await retriever.search(
            "query",
            top_k=2,
            synthesize_citation_metadata=True,
            synthesis_metadata_keys=["record_id", "collection"],
        )

        assert "Citation metadata:" in result.context
        assert '"record_id": "row-42"' in result.context
        assert '"collection": "movies"' in result.context
        assert "row-source" not in result.context

    @pytest.mark.asyncio
    async def test_citation_metadata_in_llm_prompt(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(store, llm, FakeEmbedder())

        await retriever.search(
            "query",
            top_k=2,
            synthesize_citation_metadata=True,
            synthesis_metadata_keys=["record_id"],
        )

        assert "Citation metadata:" in llm.prompts[0]
        assert '"record_id": "row-42"' in llm.prompts[0]


class TestDriftCommunityLevelAliases:
    """Tests for community_level alias resolution in DRIFT."""

    @pytest.mark.asyncio
    async def test_default_community_level(self):
        """Default community_level=0 is used when not specified."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2)

        summary_call = [
            c for c in store.calls if c[0] == "get_community_summaries_by_keys"
        ][0]
        # Keys should only contain level-0 communities
        for key in summary_call[1]["keys"]:
            assert key["level"] == 0

    @pytest.mark.asyncio
    async def test_coarsest_alias_resolves(self):
        """community_level='coarsest' resolves to max level."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(
            store, FakeLLM(), FakeEmbedder(), community_level="coarsest"
        )

        await retriever.search("query", top_k=2)

        summary_call = [
            c for c in store.calls if c[0] == "get_community_summaries_by_keys"
        ][0]
        assert summary_call[1]["keys"] == [{"id": "c2", "level": 2}]

    @pytest.mark.asyncio
    async def test_finest_alias_resolves(self):
        """community_level='finest' resolves to 0 (current semantics)."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(
            store, FakeLLM(), FakeEmbedder(), community_level="finest"
        )

        await retriever.search("query", top_k=2)

        summary_call = [
            c for c in store.calls if c[0] == "get_community_summaries_by_keys"
        ][0]
        for key in summary_call[1]["keys"]:
            assert key["level"] == 0

    @pytest.mark.asyncio
    async def test_explicit_level_override(self):
        """community_level in search() overrides constructor value."""
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(
            store, FakeLLM(), FakeEmbedder(), community_level="finest"
        )

        await retriever.search("query", top_k=2, community_level=2)

        summary_call = [
            c for c in store.calls if c[0] == "get_community_summaries_by_keys"
        ][0]
        for key in summary_call[1]["keys"]:
            assert key["level"] == 2


class TestDriftMissingCommunities:
    """Tests for behavior when no communities exist."""

    @pytest.mark.asyncio
    async def test_missing_communities_still_returns_result(self):
        """Entity search succeeds even when no communities are found."""
        store = EmptyCommunitiesGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert result.mode == "drift"
        assert result.answer
        assert "Alice" in result.context

    @pytest.mark.asyncio
    async def test_missing_communities_community_context_empty(self):
        """Community and bridging context is empty when no communities exist."""
        store = EmptyCommunitiesGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        result = await retriever.search("query", top_k=2)

        assert "No community context available" in result.context
        assert "No bridging entities found" in result.context

    @pytest.mark.asyncio
    async def test_missing_communities_citations_still_resolved(self):
        """Citations are resolved from entity chunks even without communities."""
        store = EmptyCommunitiesGraphStore()
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


class TestDriftForwardedParams:
    """Tests for parameter forwarding to the hybrid retriever."""

    @pytest.mark.asyncio
    async def test_effective_search_ratio_forwarded(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, effective_search_ratio=3)

        vector_call = [c for c in store.calls if c[0] == "vector_search"][0]
        keyword_call = [c for c in store.calls if c[0] == "keyword_search"][0]
        assert vector_call[1]["k"] == 6  # 2 * 3
        assert keyword_call[1]["k"] == 6

    @pytest.mark.asyncio
    async def test_query_params_forwarded(self):
        store = FakeGraphStore()
        retriever = DriftSearchRetriever(store, FakeLLM(), FakeEmbedder())

        await retriever.search("query", top_k=2, query_params={"custom": "value"})

        ctx_call = [c for c in store.calls if c[0] == "fetch_entity_context"][0]
        assert ctx_call[1]["matches"][0]["id"] in ("a", "b")


class TestDriftAnswerPrompt:
    """Tests for custom answer prompts."""

    @pytest.mark.asyncio
    async def test_custom_answer_prompt(self):
        store = FakeGraphStore()
        llm = FakeLLM()
        retriever = DriftSearchRetriever(
            store, llm, FakeEmbedder(),
            answer_prompt="Q: {query}\nE: {entity_context}\nC: {community_context}\nB: {bridging_context}\nA:",
        )

        await retriever.search("query", top_k=2)

        assert "Q: query" in llm.prompts[0]
        assert "E:" in llm.prompts[0]
        assert "C:" in llm.prompts[0]
        assert "B:" in llm.prompts[0]


# ---------------------------------------------------------------------------
# Target behavioral tests — xfail until Phase 6 rewrite
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 6: iterative DRIFT primer with community report semantic retrieval"
)
class TestDriftPrimerSemantics:
    """Target behavior: primer retrieves community reports semantically."""

    @pytest.mark.asyncio
    async def test_primer_retrieves_reports_semantically(self):
        """Primer should vector-search community reports, not just entity communities."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT primer")

    @pytest.mark.asyncio
    async def test_primer_returns_json_with_score_and_followups(self):
        """Primer LLM returns strict JSON: answer, score, follow_ups, report_ids."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT primer")


@pytest.mark.xfail(reason="Phase 6: iterative DRIFT follow-up traversal")
class TestDriftFollowUpTraversal:
    """Target behavior: follow-up questions drive local searches."""

    @pytest.mark.asyncio
    async def test_followups_use_generated_questions(self):
        """Follow-up local searches use primer-generated questions, not raw query."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT traversal")

    @pytest.mark.asyncio
    async def test_traversal_respects_depth_limit(self):
        """Traversal stops at max_depth (default 3)."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT traversal")

    @pytest.mark.asyncio
    async def test_traversal_respects_llm_call_limit(self):
        """Traversal stops after max_llm_calls (default 20)."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT traversal")

    @pytest.mark.asyncio
    async def test_followups_filtered_by_score(self):
        """Only follow-ups with score >= min_expand_score are expanded."""
        pytest.fail("Not implemented: Phase 6 iterative DRIFT traversal")


@pytest.mark.xfail(reason="Phase 6: iterative DRIFT trace metadata")
class TestDriftTraceMetadata:
    """Target behavior: metadata['drift_trace'] contains traversal details."""

    @pytest.mark.asyncio
    async def test_trace_contains_primer_reports(self):
        """drift_trace includes primer_report_ids."""
        pytest.fail("Not implemented: Phase 6 DRIFT trace metadata")

    @pytest.mark.asyncio
    async def test_trace_contains_generated_questions(self):
        """drift_trace includes generated_questions from primer."""
        pytest.fail("Not implemented: Phase 6 DRIFT trace metadata")

    @pytest.mark.asyncio
    async def test_trace_contains_actions(self):
        """drift_trace includes actions with id, parent, depth, query, score, status."""
        pytest.fail("Not implemented: Phase 6 DRIFT trace metadata")

    @pytest.mark.asyncio
    async def test_trace_contains_stopping_reason(self):
        """drift_trace includes stopping_reason."""
        pytest.fail("Not implemented: Phase 6 DRIFT trace metadata")


@pytest.mark.xfail(reason="Phase 6: DRIFT combined citations")
class TestDriftCombinedCitations:
    """Target behavior: citations include both local chunks and report references."""

    @pytest.mark.asyncio
    async def test_citations_include_local_chunks(self):
        """Local chunk citations from entity retrieval are included."""
        pytest.fail("Not implemented: Phase 6 DRIFT combined citations")

    @pytest.mark.asyncio
    async def test_citations_include_report_references(self):
        """Report-derived references (from report JSON or [refs: ...]) are included."""
        pytest.fail("Not implemented: Phase 6 DRIFT combined citations")

    @pytest.mark.asyncio
    async def test_citations_are_deduplicated(self):
        """Duplicate citations across local and report sources are deduped."""
        pytest.fail("Not implemented: Phase 6 DRIFT combined citations")


@pytest.mark.xfail(reason="Phase 6: DRIFT fallback behavior")
class TestDriftFallbackBehavior:
    """Target behavior: graceful fallback when report embeddings missing."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_report_embeddings(self):
        """Falls back to local-style retrieval, records reason in metadata."""
        pytest.fail("Not implemented: Phase 6 DRIFT fallback behavior")

    @pytest.mark.asyncio
    async def test_fallback_reason_in_metadata(self):
        """metadata['drift_fallback_reason'] = 'missing_report_embeddings'."""
        pytest.fail("Not implemented: Phase 6 DRIFT fallback behavior")
