"""Structured graph evidence on ``SearchResult.graph`` (opt-in)."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from recon_graphrag.retrieval import graph_evidence
from recon_graphrag.retrieval.graph_evidence import (
    GraphEvidenceBuilder,
    relationship_key,
)
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever
from tests.retrieval import test_drift as drift_fakes
from tests.retrieval import test_global_search as global_fakes
from tests.retrieval import test_mixed_context as local_fakes


def member_row(members: list[dict], edges: list[dict]) -> dict:
    return {
        "members": members,
        "edges": edges,
        "member_count": len(members),
        "edge_count": len(edges),
    }


DEFAULT_MEMBERS = [
    {"id": "person:alice", "name": "Alice", "labels": ["Person"], "description": "CEO"},
    {"id": "org:acme", "name": "Acme", "labels": ["Organization"], "description": "Tech"},
]
DEFAULT_EDGES = [
    {
        "source": "person:alice",
        "target": "org:acme",
        "type": "LEADS",
        "description": "Alice leads Acme",
        "weight": 3.0,
    }
]


class ExpansionStore:
    """Adds the community-member read and records every expansion call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expansion_calls: list[tuple[str, str, int, int]] = []
        self.member_rows: dict[tuple[str, int], dict] = {}
        self.expansion_error: Exception | None = None

    def fetch_community_member_graph(self, graph_name, community_id, level, member_limit=25):
        self.expansion_calls.append((graph_name, community_id, level, member_limit))
        if self.expansion_error is not None:
            raise self.expansion_error
        row = self.member_rows.get((community_id, level))
        if row is None:
            return [member_row(DEFAULT_MEMBERS, DEFAULT_EDGES)]
        return [row]


class LocalEvidenceStore(ExpansionStore, local_fakes.FakeGraphStore):
    pass


class RowEvidenceStore(ExpansionStore, drift_fakes.FakeGraphStore):
    """DRIFT store whose entity rows look like the local mixed-context fakes."""

    def fetch_entity_context(
        self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None
    ):
        return local_rows()

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []


class DriftEvidenceStore(ExpansionStore, drift_fakes.FakeGraphStore):
    pass


class GlobalEvidenceStore(ExpansionStore, global_fakes.FakeGraphStore):
    pass


def ids(payload: dict) -> set[str]:
    return {node["id"] for node in payload["nodes"]}


def edges(payload: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["type"], edge["target"]) for edge in payload["edges"]}


# -- builder unit tests ---------------------------------------------------
def test_edge_identity_never_splits_compound_keys():
    key = relationship_key("person:alice", "LEADS", "org:acme:1")
    assert key == "person:alice:LEADS:org:acme:1"


def test_edges_keep_direction_and_dedupe_and_close_endpoints():
    builder = GraphEvidenceBuilder(graph_name="g", mode="local")
    builder.add_edge("a", "b", "KNOWS", source_name="A", target_name="B")
    builder.add_edge("a", "b", "KNOWS")
    builder.add_edge("b", "a", "KNOWS")
    builder.add_node("orphan", name="Lonely")
    payload = builder.build()

    assert edges(payload) == {("a", "KNOWS", "b"), ("b", "KNOWS", "a")}
    assert ids(payload) == {"a", "b", "orphan"}
    origins = {node["id"]: node["origin"] for node in payload["nodes"]}
    assert origins["a"] == "edge_endpoint"
    assert origins["orphan"] == "retrieved"


def test_context_row_cut_matches_rendered_top_k():
    builder = GraphEvidenceBuilder(graph_name="g", mode="local")
    builder.add_context_row(
        {
            "entity_id": "a",
            "entity_name": "Alice",
            "entity_labels": ["__Entity__", "Person"],
            "relationship_records": [
                {"source_id": "a", "target_id": f"t{i}", "rel": "REL", "weight": i}
                for i in range(1, 5)
            ],
        },
        top_k_relationships=2,
    )
    payload = builder.build()

    assert len(payload["edges"]) == 2
    assert {edge["weight"] for edge in payload["edges"]} == {4.0, 3.0}
    reason = next(r for r in payload["truncation"]["reasons"] if r["code"] == "top_k_relationships")
    assert reason["kept"] == 2 and reason["dropped"] == 2
    # The plain __Entity__ label is not surfaced.
    assert payload["nodes"][0]["labels"] == ["Person"] or payload["nodes"][0]["id"] != "a"


@pytest.mark.asyncio
async def test_expansion_never_returns_community_vertices():
    store = LocalEvidenceStore()
    store.member_rows = {}
    builder = GraphEvidenceBuilder(graph_name="g", mode="local", graph_store=store)
    await builder.expand_communities([{"community_id": "c2", "level": 2}], origin="context")
    payload = builder.build()

    assert payload["nodes"], "expansion produced members"
    assert all("Community" not in node["labels"] for node in payload["nodes"])
    assert payload["provenance"]["communities"][0]["level"] == 2
    assert store.expansion_calls == [("g", "c2", 2, 25)]


@pytest.mark.asyncio
async def test_expansion_caps_are_reported_and_deterministic():
    store = LocalEvidenceStore()
    members = [{"id": f"m{i}", "name": f"M{i}"} for i in range(3)]
    heavy_edges = [
        {"source": "m0", "target": f"m{i}", "type": "R", "weight": float(i)}
        for i in range(1, 6)
    ]
    store.member_rows = {("c0", 0): member_row(members, heavy_edges)}
    builder = GraphEvidenceBuilder(
        graph_name="g",
        mode="global",
        graph_store=store,
        member_limit=2,
        edge_limit=2,
        max_reports=1,
    )
    await builder.expand_communities(
        [
            {"community_id": "c0", "level": 0},
            {"community_id": "c1", "level": 0},
            {"community_id": "c2", "level": 0},
        ]
    )
    first = builder.build()
    second = builder.build()

    codes = {reason["code"] for reason in first["truncation"]["reasons"]}
    assert {"report_limit", "edge_limit", "member_limit"} <= codes
    report_limit = next(
        r for r in first["truncation"]["reasons"] if r["code"] == "report_limit"
    )
    assert report_limit["kept"] == 1 and report_limit["dropped"] == 2
    assert first["provenance"]["communities"][0]["member_count_known"] is False
    # Only the two strongest edges survive; output order is stable by key.
    assert {edge["weight"] for edge in first["edges"]} == {5.0, 4.0}
    assert first["nodes"] == second["nodes"] and first["edges"] == second["edges"]


@pytest.mark.asyncio
async def test_expansion_without_level_is_a_warning_not_a_read():
    store = LocalEvidenceStore()
    builder = GraphEvidenceBuilder(graph_name="g", mode="local", graph_store=store)
    await builder.expand_communities([{"community_id": "c0"}])
    payload = builder.build()

    assert store.expansion_calls == []
    assert payload["nodes"] == []
    assert payload["provenance"]["warnings"][0]["code"] == "community_level_unknown"


@pytest.mark.asyncio
async def test_expansion_failure_is_a_warning_not_an_error():
    store = LocalEvidenceStore()
    store.expansion_error = RuntimeError("db down")
    builder = GraphEvidenceBuilder(graph_name="g", mode="local", graph_store=store)
    await builder.expand_communities([{"community_id": "c0", "level": 0}])
    payload = builder.build()

    assert payload["nodes"] == []
    warning = payload["provenance"]["warnings"][0]
    assert warning["code"] == "community_expansion_failed"
    assert "db down" in warning["error"]


# -- local ---------------------------------------------------------------
def local_rows() -> list[dict]:
    return [
        {
            "title": "Alice (Person)",
            "entity_id": "a",
            "entity_name": "Alice",
            "entity_labels": ["__Entity__", "Person"],
            "entity_description": "CEO of Acme",
            "relationships": [],
            "relationship_records": [
                {
                    "source_id": "a",
                    "source_name": "Alice",
                    "target_id": "org:acme",
                    "target_name": "Acme",
                    "rel": "WORKS_AT",
                    "description": "Alice leads Acme",
                    "weight": 4.0,
                },
                {
                    "source_id": "a",
                    "source_name": "Alice",
                    "target_id": "org:other",
                    "target_name": "Other",
                    "rel": "ADVISES",
                    "description": "Alice advises Other",
                    "weight": 1.0,
                },
            ],
            "source_text": ["Alice directed Inception."],
            "source_chunk_ids": ["chunk:1"],
            "score": 0.8,
        }
    ]


@pytest.mark.asyncio
async def test_local_non_mixed_keeps_the_rendered_relationship_cut():
    store = LocalEvidenceStore()
    store.fetch_entity_context = lambda *a, **kw: local_rows()
    retriever = LocalSearchRetriever(
        cast(Any, store), cast(Any, local_fakes.FakeLLM()), cast(Any, local_fakes.FakeEmbedder()),
        use_mixed_context=False, top_k_relationships=1, graph_name="entity-graph",
    )
    result = await retriever.search("Who leads Acme?", top_k=1, include_graph_evidence=True)

    assert ids(result.graph) == {"a", "org:acme"}
    assert edges(result.graph) == {("a", "WORKS_AT", "org:acme")}
    assert result.graph["provenance"]["graph_name"] == "entity-graph"
    assert result.graph["provenance"]["mode"] == "local"


@pytest.mark.asyncio
async def test_local_flag_off_returns_empty_graph_and_no_expansion_reads():
    store = LocalEvidenceStore()
    store.fetch_entity_context = lambda *a, **kw: local_rows()
    retriever = LocalSearchRetriever(
        cast(Any, store), cast(Any, local_fakes.FakeLLM()), cast(Any, local_fakes.FakeEmbedder()),
    )
    result = await retriever.search("Who leads Acme?", top_k=1)

    assert result.graph == {}
    assert store.expansion_calls == []


@pytest.mark.asyncio
async def test_local_mixed_reports_only_what_survived_packing():
    store = LocalEvidenceStore()
    store.member_rows = {("c0", 0): member_row(DEFAULT_MEMBERS, DEFAULT_EDGES)}
    rows = local_rows() + [
        {
            "title": "Bob (Person)",
            "entity_id": "b",
            "entity_name": "Bob",
            "entity_labels": ["Person"],
            "entity_description": "Engineer",
            "relationships": [],
            "relationship_records": [],
            "source_text": ["Bob works at Acme."],
            "source_chunk_ids": ["chunk:2"],
            "score": 0.1,
        }
    ]
    store.fetch_entity_context = lambda *a, **kw: rows
    retriever = LocalSearchRetriever(
        cast(Any, store), cast(Any, local_fakes.FakeLLM()), cast(Any, local_fakes.FakeEmbedder()),
        use_mixed_context=True, graph_name="entity-graph",
    )
    result = await retriever.search(
        "Who leads Acme?", top_k=2, include_graph_evidence=True, token_budget=120,
    )

    assert result.context
    assert "a" in ids(result.graph)
    # The community report that contributed is expanded, never shown as a vertex.
    assert {"person:alice", "org:acme"} <= ids(result.graph)
    assert all("Community" not in node["labels"] for node in result.graph["nodes"])
    assert result.graph["provenance"]["communities"][0]["id"] == "c0"
    assert store.expansion_calls and store.expansion_calls[0][0] == "entity-graph"


@pytest.mark.asyncio
async def test_local_synthesize_false_still_carries_evidence():
    store = LocalEvidenceStore()
    store.fetch_entity_context = lambda *a, **kw: local_rows()
    retriever = LocalSearchRetriever(
        cast(Any, store), cast(Any, local_fakes.FakeLLM()), cast(Any, local_fakes.FakeEmbedder()),
        use_mixed_context=False, graph_name="entity-graph",
    )
    result = await retriever.search(
        "Who leads Acme?", top_k=1, synthesize_response=False, include_graph_evidence=True,
    )

    assert result.answer == ""
    assert "a" in ids(result.graph)


# -- drift ---------------------------------------------------------------
def drift_action(action_id, answer, score, batch_reports=None):
    action = __import__(
        "recon_graphrag.retrieval.drift_types", fromlist=["DriftAction"]
    ).DriftAction(id=action_id, query=f"q-{action_id}", answer=answer, score=score)
    action.status = "completed"
    return action


@pytest.mark.asyncio
async def test_drift_main_graph_uses_only_packed_actions():
    retriever = DriftSearchRetriever(
        cast(Any, DriftEvidenceStore()), cast(Any, drift_fakes.FakeLLM()), cast(Any, drift_fakes.FakeEmbedder()),
    )
    from recon_graphrag.retrieval.drift_types import DriftQueryState

    state = DriftQueryState(query="Who?")
    retained = drift_action("action:1", "Retained answer.", 90.0)
    retained.graph = {
        "nodes": [{"id": "a", "name": "Alice", "labels": ["Person"], "origin": "retrieved"}],
        "edges": [],
        "provenance": {"communities": []},
    }
    dropped = drift_action("action:2", "Dropped answer.", 1.0)
    dropped.graph = {
        "nodes": [{"id": "z", "name": "Zed", "labels": [], "origin": "retrieved"}],
        "edges": [],
        "provenance": {"communities": []},
    }
    primer = drift_action("primer", "Primer answer.", 50.0)
    primer.report_ids = []
    state.actions = [primer, retained, dropped]

    graph = await retriever._reduction_graph_evidence(
        state, [retained, dropped], {"action:1"}
    )

    assert ids(graph) == {"a"}
    assert graph["provenance"]["retained_actions"] == ["action:1"]
    # Explored = completed but not packed; the un-packed primer is not evidence.
    assert graph["provenance"]["explored_actions"] == ["action:2"]


@pytest.mark.asyncio
async def test_drift_primer_expands_only_when_packed():
    store = DriftEvidenceStore()
    retriever = DriftSearchRetriever(
        cast(Any, store), cast(Any, drift_fakes.FakeLLM()), cast(Any, drift_fakes.FakeEmbedder()),
    )
    from recon_graphrag.retrieval.drift_types import DriftQueryState

    state = DriftQueryState(query="Who?")
    primer = drift_action("primer", "Primer answer.", 10.0)
    primer.report_ids = ["report:c0:0"]
    state.actions = [primer]
    state.primer_reports = [
        {"id": "report:c0:0", "level": 0, "report_text": "Alice leads Acme."},
        {"id": "report:c1:0", "level": 0, "report_text": "Unrelated."},
    ]

    not_packed = await retriever._reduction_graph_evidence(state, [primer], set())
    assert store.expansion_calls == []
    assert not_packed["nodes"] == []

    packed = await retriever._reduction_graph_evidence(state, [primer], {"primer"})
    # Only the report the primer selected is expanded, at its own level.
    assert store.expansion_calls == [("entity-graph", "report:c0:0", 0, 25)]
    assert {"person:alice", "org:acme"} <= ids(packed)
    assert packed["provenance"]["primer_reports"] == [
        {"id": "report:c0:0", "level": 0}
    ]


@pytest.mark.asyncio
async def test_drift_fallback_returns_local_style_evidence():
    class FallbackStore(DriftEvidenceStore):
        def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
            return []

        def fetch_entity_context(self, matches, retrieval_query=None, query_params=None,
                                 mode="local", graph_name=None):
            return local_rows()

    store = FallbackStore()
    retriever = DriftSearchRetriever(
        cast(Any, store), cast(Any, drift_fakes.FakeLLM()), cast(Any, drift_fakes.FakeEmbedder()),
    )
    result = await retriever.search("Who leads Acme?", top_k=1, include_graph_evidence=True)

    assert result.metadata["drift_fallback_reason"]
    assert ids(result.graph) == {"a", "org:acme", "org:other"}
    assert result.graph["provenance"]["fallback_reason"] == "no_report_results"


@pytest.mark.asyncio
async def test_drift_full_search_attaches_per_action_evidence():
    store = RowEvidenceStore()
    llm = drift_fakes.FakeLLM(
        responses=[
            json.dumps({
                "answer": "Alice leads Acme.",
                "score": 85,
                "follow_ups": ["What else does Alice do?"],
                "report_ids": ["report:c0:0"],
            }),
            json.dumps({"answer": "She advises too.", "score": 70, "follow_ups": []}),
            json.dumps({"answer": "Final synthesis."}),
        ]
    )
    retriever = DriftSearchRetriever(
        cast(Any, store), cast(Any, llm), cast(Any, drift_fakes.FakeEmbedder()),
        config=drift_fakes.DriftSearchConfig(use_hyde=False),
    )

    result = await retriever.search("Who leads Acme?", top_k=2, include_graph_evidence=True)

    assert result.metadata["drift_trace"]["actions"]
    # The action that reached reduction contributes its own entity/relationship
    # evidence; the packed primer contributes its expanded report members.
    assert result.graph["provenance"]["retained_actions"] == ["action:1", "primer"]
    assert ids(result.graph) >= {"a", "org:acme", "person:alice"}
    assert store.expansion_calls, "packed primer reports are expanded once"


@pytest.mark.asyncio
async def test_drift_full_search_flag_off_skips_expansion():
    store = RowEvidenceStore()
    llm = drift_fakes.FakeLLM(
        responses=[
            json.dumps({
                "answer": "Alice leads Acme.",
                "score": 85,
                "follow_ups": [],
                "report_ids": ["report:c0:0"],
            }),
            json.dumps({"answer": "Final synthesis."}),
        ]
    )
    retriever = DriftSearchRetriever(
        cast(Any, store), cast(Any, llm), cast(Any, drift_fakes.FakeEmbedder()),
        config=drift_fakes.DriftSearchConfig(use_hyde=False),
    )

    result = await retriever.search("Who leads Acme?", top_k=2)

    assert result.graph == {}
    assert store.expansion_calls == []


# -- global --------------------------------------------------------------
def global_reports() -> list[dict]:
    return [
        {
            "id": "report:c0:0",
            "level": 0,
            "report_text": "Alice leads Acme.",
            "report_json": json.dumps(
                {"findings": [{"references": [{"target_id": "person:alice", "target_type": "entity"}]}]}
            ),
            "rating": 8.0,
        },
        {
            "id": "report:c1:0",
            "level": 0,
            "report_text": "Bob writes code.",
            "report_json": json.dumps(
                {"findings": [{"references": [{"target_id": "person:bob", "target_type": "entity"}]}]}
            ),
            "rating": 8.0,
        },
    ]


@pytest.mark.asyncio
async def test_global_expands_only_reports_that_reached_the_reduce_prompt():
    map_responses = [
        json.dumps({"answer": "Alice leads Acme.", "helpfulness": 90, "report_ids": ["report:c0:0"]}),
        json.dumps({"answer": "Bob writes code.", "helpfulness": 10, "report_ids": ["report:c1:0"]}),
    ]
    store = GlobalEvidenceStore(global_reports())
    retriever = GlobalSearchRetriever(
        cast(Any, store),
        cast(Any, global_fakes.FakeLLM(map_responses=map_responses)),
        graph_name="entity-graph",
        map_budget_tokens=10000,
        # Room for the first partial only; the second never reaches reduce.
        reduce_budget_tokens=10,
    )
    result = await retriever.search(
        "Who leads Acme?", community_level=0, include_graph_evidence=True,
    )

    assert result.graph["provenance"]["retained_batches"] == ["0"]
    assert [call[1] for call in store.expansion_calls] == ["report:c0:0"]
    assert result.graph["provenance"]["synthesis_included"] is True


@pytest.mark.asyncio
async def test_global_synthesize_false_reports_context_partials():
    map_responses = [
        json.dumps({"answer": "Alice leads Acme.", "helpfulness": 90, "report_ids": ["report:c0:0"]}),
    ]
    store = GlobalEvidenceStore(global_reports()[:1])
    retriever = GlobalSearchRetriever(
        cast(Any, store), cast(Any, global_fakes.FakeLLM(map_responses=map_responses)),
        graph_name="entity-graph",
    )
    result = await retriever.search(
        "Who leads Acme?", community_level=0,
        synthesize_response=False, include_graph_evidence=True,
    )

    assert result.answer == ""
    assert result.graph["provenance"]["synthesis_included"] is False
    assert result.graph["nodes"], "context partials still expand their reports"


@pytest.mark.asyncio
async def test_global_flag_off_makes_no_expansion_reads():
    store = GlobalEvidenceStore(global_reports()[:1])
    retriever = GlobalSearchRetriever(
        cast(Any, store),
        cast(Any, global_fakes.FakeLLM(map_responses=[json.dumps(
            {"answer": "Alice leads Acme.", "helpfulness": 90, "report_ids": ["report:c0:0"]}
        )])),
        graph_name="entity-graph",
    )
    result = await retriever.search("Who leads Acme?", community_level=0)

    assert result.graph == {}
    assert store.expansion_calls == []


def test_module_constants_are_the_approved_caps():
    assert graph_evidence.DEFAULT_MEMBER_LIMIT == 25
    assert graph_evidence.DEFAULT_EDGE_LIMIT == 50
    assert graph_evidence.DEFAULT_MAX_REPORTS == 5
    assert graph_evidence.DEFAULT_MAX_NODES == 200
    assert graph_evidence.DEFAULT_MAX_EDGES == 500
