"""Deterministic workflow tests against real graph databases.

Uses fake LLM and embedder with controlled output to exercise the full
pipeline (extraction -> aggregation -> seeded community -> search) against real
Neo4j and Memgraph without provider cost.

Marked with ``@pytest.mark.workflow`` and ``@pytest.mark.database``.
Requires ``RUN_WORKFLOW_INTEGRATION_TESTS=1`` and database env vars.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.integration.factories import get_memgraph_store, get_neo4j_store
from recon_graphrag import (
    CommunityReportEmbedder,
    DriftSearchConfig,
    DriftSearchRetriever,
    GlobalSearchRetriever,
    GraphBuilderPipeline,
    LocalSearchRetriever,
)
from recon_graphrag.models.artifacts import (
    CommunityFinding,
    CommunityReport,
    FindingReference,
)
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.llm.base import LLMResponse, LLMUsage
from tests.integration.support import cleanup_graph, single_count, require_integration_env

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

WORKFLOW_SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Person",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Organization",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="System",
            properties=[PropertyType(name="name", type="STRING")],
        ),
    ],
    relationship_types=[
        RelationshipType(label="WORKS_AT"),
        RelationshipType(label="OPERATES"),
    ],
    patterns=[
        ("Person", "WORKS_AT", "Organization"),
        ("Organization", "OPERATES", "System"),
    ],
)

# ---------------------------------------------------------------------------
# Deterministic fake LLM / embedder
# ---------------------------------------------------------------------------

_EXTRACTION_RESPONSE = json.dumps({
    "nodes": [
        {"id": "alice", "label": "Person", "properties": {"name": "Alice Rivera"}},
        {"id": "acme", "label": "Organization", "properties": {"name": "Acme Security"}},
        {"id": "sentinel", "label": "System", "properties": {"name": "Sentinel"}},
    ],
    "relationships": [
        {"source_id": "alice", "target_id": "acme", "type": "WORKS_AT"},
        {"source_id": "acme", "target_id": "sentinel", "type": "OPERATES"},
    ],
})


def _make_fake_llm():
    async def extract(prompt, **kwargs):
        assert not kwargs
        assert "Allowed node types:" in prompt
        assert "Allowed relationship patterns:" in prompt
        assert "Text:\n" in prompt
        assert _SOURCE_TEXT in prompt
        return LLMResponse(
            content=_EXTRACTION_RESPONSE,
            usage=LLMUsage(request_tokens=100, response_tokens=150, total_tokens=250),
        )

    llm = MagicMock(model_name="deterministic-fake")
    llm.ainvoke = AsyncMock(side_effect=extract)
    llm.aclose = AsyncMock()
    return llm


# Match the existing test profile; Memgraph indexes are global per label/property.
_FAKE_EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
assert _FAKE_EMBED_DIM > 0, "EMBEDDING_DIM must be positive"
_SOURCE_TEXT = (
    "Alice Rivera works at Acme Security. "
    "Acme Security operates the Sentinel monitoring system."
)
_QUESTION = "How is Alice Rivera connected to Sentinel?"
_FOLLOW_UP = "Which system does Acme Security operate?"
_EXPECTED_ANSWER = "Alice Rivera is connected to Sentinel through Acme Security."


def _make_fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1] * _FAKE_EMBED_DIM)
    embedder.aclose = AsyncMock()
    return embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_workflow_fixture(store_factory, run_flag, required_env, graph_name):
    @pytest.fixture
    def fixture():
        require_integration_env(
            run_flag, required_env, "Workflow integration test", fail_on_missing=True,
        )
        store = store_factory()
        cleanup_graph(store, graph_name)
        try:
            yield store, graph_name
        finally:
            try:
                cleanup_graph(store, graph_name)
                assert single_count(
                    store,
                    "MATCH (n {graph_name: $graph_name}) RETURN count(n) AS count",
                    graph_name,
                ) == 0
            finally:
                store.driver.close()

    return fixture


neo4j_workflow = _make_workflow_fixture(
    get_neo4j_store,
    "RUN_WORKFLOW_INTEGRATION_TESTS",
    ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "workflow-deterministic-neo4j",
)

memgraph_workflow = _make_workflow_fixture(
    get_memgraph_store,
    "RUN_WORKFLOW_INTEGRATION_TESTS",
    ["MEMGRAPH_URL"],
    "workflow-deterministic-memgraph",
)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def _run_deterministic_workflow(store, graph_name: str):
    llm = _make_fake_llm()
    embedder = _make_fake_embedder()

    try:
        # Use the idempotent store API: IndexManager.create_indexes drops indexes.
        store.create_vector_index(
            "entity-embeddings", "__Entity__", "embedding", _FAKE_EMBED_DIM,
        )
        store.create_fulltext_index("entity-names", "__Entity__", ["name"])
        store.create_vector_index(
            "community-report-embeddings", "Community", "report_embedding", _FAKE_EMBED_DIM,
        )

        builder = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            graph_name=graph_name,
            perform_entity_resolution=False,
        )
        build_result = await builder.build_from_text(
            _SOURCE_TEXT,
            metadata={"source": f"{graph_name}-source", "collection": "workflow-test"},
            schema=WORKFLOW_SCHEMA,
            chunk_size=500,
            chunk_overlap=50,
            max_gleanings=0,
        )

        validation = build_result.get("validation", {})
        for key in ("chunk_count", "entity_count", "evidence_link_count", "entity_relationship_count"):
            assert validation.get(key, 0) > 0, f"Expected {key} > 0, got {validation.get(key)}"

        assert single_count(
            store,
            "MATCH (d:Document {graph_name: $graph_name}) RETURN count(d) AS count",
            graph_name,
        ) == 1
        assert single_count(
            store,
            "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN count(e) AS count",
            graph_name,
        ) == 3
        assert single_count(
            store,
            "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
            graph_name,
        ) >= 1

        entity_types = set()
        result = store.execute_query(
            "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN DISTINCT labels(e) AS labels",
            {"graph_name": graph_name},
        )
        for row in result:
            for label in row["labels"]:
                if label not in ("__Entity__",):
                    entity_types.add(label)
        assert entity_types == {"Person", "Organization", "System"}, (
            f"Unexpected entity types: {entity_types}"
        )

        rel_types_result = store.execute_query(
            """
            MATCH (a:__Entity__ {graph_name: $graph_name})-[r]->(b:__Entity__ {graph_name: $graph_name})
            WHERE r.graph_name = $graph_name
            RETURN DISTINCT type(r) AS rel_type
            """,
            {"graph_name": graph_name},
        )
        rel_types = {row["rel_type"] for row in rel_types_result}
        assert rel_types == {"WORKS_AT", "OPERATES"}, f"Unexpected relationship types: {rel_types}"

        assert llm.ainvoke.await_count == 1
        entity_rows = store.execute_query(
            "MATCH (e:__Entity__ {graph_name: $graph_name}) "
            "RETURN e.id AS id, e.name AS name",
            {"graph_name": graph_name},
        )
        alice_id = next(row["id"] for row in entity_rows if row["name"] == "Alice Rivera")

        # Seed one explicit community instead of depending on backend-specific
        # Leiden partitions/levels. Reports and vectors still use real store APIs.
        community_id = "report:workflow:0"
        report_id = community_id
        store.execute_query(
            "CREATE (c:Community {graph_name: $graph_name, id: $id, level: 0}) "
            "WITH c MATCH (e:__Entity__ {graph_name: $graph_name}) "
            "CREATE (e)-[:IN_COMMUNITY {graph_name: $graph_name}]->(c)",
            {"graph_name": graph_name, "id": community_id},
        )
        store.store_community_report(
            CommunityReport(
                id=report_id,
                community_id=community_id,
                level=0,
                title="Acme Security and Sentinel",
                summary=_SOURCE_TEXT,
                rating=9.0,
                findings=[CommunityFinding(
                    id="finding:workflow",
                    description=_SOURCE_TEXT,
                    references=[FindingReference(target_id=alice_id, target_type="entity")],
                )],
            ),
            graph_name=graph_name,
        )
        assert await CommunityReportEmbedder(store, embedder, graph_name).embed_reports() == 1
        reports = store.get_community_reports_by_keys(
            graph_name=graph_name, keys=[{"id": community_id, "level": 0}], top_k=1,
        )
        assert len(reports) == 1
        assert _SOURCE_TEXT in reports[0]["report_text"]
        assert single_count(
            store,
            "MATCH (c:Community {graph_name: $graph_name}) "
            "WHERE c.report_embedding IS NOT NULL RETURN count(c) AS count",
            graph_name,
        ) == 1

        stages = []

        async def search_response(prompt, **kwargs):
            assert _QUESTION in prompt
            response_format = kwargs.get("response_format")
            if response_format is not None:
                assert set(kwargs) == {"response_format"}
                assert response_format["type"] == "json_schema"
                contract = response_format["json_schema"]
                assert contract["strict"] is True
                schema = contract["schema"]
                assert schema["additionalProperties"] is False
                if contract["name"] == "drift_primer":
                    stage = "primer"
                    assert f"Report {report_id} (level 0):" in prompt
                    assert _SOURCE_TEXT in prompt
                    assert alice_id in prompt
                    content = {
                        "answer": "Alice Rivera works at Acme Security.", "score": 80,
                        "follow_ups": [_FOLLOW_UP], "report_ids": [report_id], "references": [],
                    }
                else:
                    assert contract["name"] == "drift_action"
                    stage = "action"
                    assert f"## Follow-up Question\n{_FOLLOW_UP}" in prompt
                    assert "## Parent Answer\nAlice Rivera works at Acme Security." in prompt
                    local_context = prompt.split("## Local Context\n", 1)[1]
                    assert "=== Graph Facts ===" in local_context
                    assert "=== Source Text ===" in local_context
                    assert "=== Community Reports ===" in local_context
                    assert _SOURCE_TEXT in local_context
                    assert "OPERATES" in local_context
                    content = {
                        "answer": "Acme Security operates Sentinel.", "score": 95, "follow_ups": [],
                    }
                assert set(schema["properties"]) == set(content)
                assert set(schema["required"]) == set(content)
                content = json.dumps(content)
            else:
                assert not kwargs
                if "Findings and connections:" in prompt:
                    stage = "local"
                    assert "=== Graph Facts ===" in prompt
                    assert "OPERATES" in prompt
                    assert _SOURCE_TEXT in prompt
                    content = _EXPECTED_ANSWER
                elif "## Community Reports (sorted by helpfulness" in prompt:
                    stage = "map"
                    assert report_id in prompt
                    assert _SOURCE_TEXT in prompt
                    assert alice_id in prompt
                    content = json.dumps({
                        "answer": f"Acme Security operates Sentinel. [Report:{report_id}]",
                        "helpfulness": 95, "report_ids": [report_id], "references": [],
                    })
                elif "## Partial Answers (sorted by helpfulness)" in prompt:
                    stage = "global_reduce"
                    assert "Acme Security operates Sentinel." in prompt
                    assert report_id in prompt
                    content = f"Global: {_EXPECTED_ANSWER}"
                else:
                    stage = "drift_reduce"
                    assert "## Partial Answers (sorted by confidence)" in prompt
                    assert "Alice Rivera works at Acme Security." in prompt
                    assert "Acme Security operates Sentinel." in prompt
                    content = f"DRIFT: {_EXPECTED_ANSWER}"
            stages.append(stage)
            return LLMResponse(content=content)

        llm.ainvoke.side_effect = search_response
        local = await LocalSearchRetriever(store, llm, embedder, graph_name=graph_name).search(
            _QUESTION, top_k=3, community_level=0,
        )
        assert local.mode == "local"
        # These verify response plumbing, not a real model's factual accuracy.
        assert local.answer == _EXPECTED_ANSWER
        assert _SOURCE_TEXT in local.context
        assert local.metadata["mixed_context"] is True

        global_result = await GlobalSearchRetriever(store, llm, graph_name=graph_name).search(
            _QUESTION, community_level=0, random_seed=42,
        )
        assert global_result.mode == "global"
        assert global_result.answer == f"Global: {_EXPECTED_ANSWER}"
        for key in (
            "reports_available", "reports_used", "map_batches", "map_succeeded",
            "reduce_partials_used", "source_report_ids_used",
        ):
            assert global_result.metadata[key] == 1
        assert global_result.metadata["map_failed"] == 0
        assert global_result.metadata["map_filtered_zero"] == 0
        assert global_result.metadata["source_resolution_errors"] == []

        drift = await DriftSearchRetriever(
            store, llm, embedder, graph_name=graph_name,
            config=DriftSearchConfig(
                use_hyde=False, primer_top_k=1, primer_folds=1, community_level=0,
                max_depth=1, max_followups=1, max_llm_calls=3, action_concurrency=1,
                action_use_mixed_context=True,
            ),
        ).search(_QUESTION, top_k=3)
        assert drift.mode == "drift"
        assert drift.answer == f"DRIFT: {_EXPECTED_ANSWER}"
        assert "drift_fallback_reason" not in drift.metadata
        trace = drift.metadata["drift_trace"]
        assert trace["primer_report_ids"] == [report_id]
        assert trace["generated_questions"] == [_FOLLOW_UP]
        assert trace["total_llm_calls"] == 3
        assert trace["failures"] == []
        assert "fallback_reason" not in trace
        assert set(trace["phase_tokens"]) - {"elapsed_ms"} == {"primer", "action", "reduce"}
        assert all(trace["phase_tokens"][phase] > 0 for phase in ("primer", "action", "reduce"))
        assert trace["stopping_reason"] == "all_actions_completed"
        assert [(a["id"], a["depth"], a["status"]) for a in trace["actions"]] == [
            ("primer", 0, "completed"), ("action:1", 1, "completed"),
        ]
        assert stages == ["local", "map", "global_reduce", "primer", "action", "drift_reduce"]

        source_rows = store.execute_query(
            "MATCH (c:Chunk {graph_name: $graph_name})-[:PART_OF]->"
            "(d:Document {graph_name: $graph_name}) "
            "RETURN c.id AS chunk_id, d.id AS document_id",
            {"graph_name": graph_name},
        )
        expected_sources = {(row["document_id"], row["chunk_id"]) for row in source_rows}
        assert len(expected_sources) == 1
        for result in (local, global_result, drift):
            assert {(c.document_id, c.chunk_id) for c in result.citations} == expected_sources
            assert len(result.sources) == 1
            assert result.sources[0].document_name == f"{graph_name}-source"
            assert result.sources[0].chunk_list == result.citations
            assert all(c.excerpt is not None and _SOURCE_TEXT in c.excerpt for c in result.citations)

        print(f"\n  Deterministic workflow PASSED ({graph_name})")
        print(f"  Build: {validation}")
        print(f"  Entity types: {entity_types}")
        print(f"  Relationship types: {rel_types}")

    finally:
        await llm.aclose()
        await embedder.aclose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.workflow
@pytest.mark.database
async def test_deterministic_workflow_neo4j(neo4j_workflow):
    store, graph_name = neo4j_workflow
    await _run_deterministic_workflow(store, graph_name)


@pytest.mark.integration
@pytest.mark.workflow
@pytest.mark.database
async def test_deterministic_workflow_memgraph(memgraph_workflow):
    store, graph_name = memgraph_workflow
    await _run_deterministic_workflow(store, graph_name)
