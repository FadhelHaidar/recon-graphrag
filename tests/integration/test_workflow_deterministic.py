"""Deterministic workflow tests against real graph databases.

Uses fake LLM and embedder with controlled output to exercise the full
pipeline (extraction -> aggregation -> community -> search) against real
Neo4j and Memgraph without provider cost.

Marked with ``@pytest.mark.workflow`` and ``@pytest.mark.database``.
Requires ``RUN_WORKFLOW_INTEGRATION_TESTS=1`` and database env vars.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.integration.factories import get_memgraph_store, get_neo4j_store
from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, IndexManager
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
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
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=_EXTRACTION_RESPONSE))
    return llm


def _make_fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1] * 64)
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
        IndexManager(store, embedding_dim=64).create_indexes()

        builder = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            schema=WORKFLOW_SCHEMA,
            graph_name=graph_name,
            perform_entity_resolution=False,
        )
        build_result = (await builder.build_from_text(
            "Alice Rivera works at Acme Security. Acme Security operates the Sentinel monitoring system.",
            metadata={"source": f"{graph_name}-source", "collection": "workflow-test"},
            chunk_size=500,
            chunk_overlap=50,
        ))[0]

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

        print(f"\n  Deterministic workflow PASSED ({graph_name})")
        print(f"  Build: {validation}")
        print(f"  Entity types: {entity_types}")
        print(f"  Relationship types: {rel_types}")

    finally:
        for resource in (llm, embedder):
            close = getattr(resource, "aclose", None)
            if callable(close):
                await close()


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
