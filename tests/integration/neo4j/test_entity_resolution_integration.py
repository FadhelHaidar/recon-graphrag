"""Opt-in Neo4j integration tests for entity resolution."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from examples.movie_industry.config import get_embedder, get_llm, get_neo4j_store


RUN_FLAG = "RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS"
RUN_AI_FLAG = "RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS"
GRAPH_NAME = "entity-resolution-integration"
REQUIRED_NEO4J_ENV = [
    "NEO4J_URL",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]
REQUIRED_AI_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
]


def _neo4j_env_or_skip() -> None:
    load_dotenv()

    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to run Neo4j entity resolution tests.")

    missing = [name for name in REQUIRED_NEO4J_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required Neo4j env vars: {', '.join(missing)}")


def _ai_env_or_skip() -> None:
    if os.getenv(RUN_AI_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_AI_FLAG}=1 to run real LLM/embedder resolution test.")

    missing = [name for name in REQUIRED_AI_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required AI env vars: {', '.join(missing)}")


def _preflight_or_fail(store) -> None:
    checks = [
        ("Neo4j connectivity", "RETURN 1 AS ok"),
        ("APOC", "RETURN apoc.version() AS version"),
    ]
    for label, query in checks:
        try:
            store.execute_query(query)
        except Exception as exc:
            pytest.fail(f"{label} preflight failed: {exc}")


def _cleanup_graph(store) -> None:
    store.execute_query(
        """
        MATCH (n {graph_name: $graph_name})
        DETACH DELETE n
        """,
        {"graph_name": GRAPH_NAME},
    )


def _seed_entities(store, rows: list[dict]) -> None:
    store.execute_query(
        """
        UNWIND $rows AS row
        CALL apoc.create.node(
            ['__Entity__', row.domain_label],
            {
                id: row.id,
                graph_name: $graph_name,
                name: row.name,
                description: coalesce(row.description, '')
            }
        ) YIELD node
        RETURN count(node) AS count
        """,
        {"graph_name": GRAPH_NAME, "rows": rows},
    )


def _entity_count(store) -> int:
    rows = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})
        RETURN count(e) AS count
        """,
        {"graph_name": GRAPH_NAME},
    )
    return int(rows[0]["count"]) if rows else 0


@pytest.fixture
def neo4j_store():
    _neo4j_env_or_skip()
    store = get_neo4j_store()
    _preflight_or_fail(store)
    _cleanup_graph(store)
    try:
        yield store
    finally:
        _cleanup_graph(store)
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_normalized_entity_resolution_merges_real_nodes(neo4j_store):
    _seed_entities(
        neo4j_store,
        [
            {"id": "org:openai-a", "domain_label": "Organization", "name": "OpenAI"},
            {"id": "org:openai-b", "domain_label": "Organization", "name": "openai"},
        ],
    )

    result = await neo4j_store.resolve_entities(
        graph_name=GRAPH_NAME,
        strategy="normalized",
    )

    assert result["skipped"] is False
    assert result["merged_groups"] == 1
    assert _entity_count(neo4j_store) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_hybrid_alias_dry_run_returns_candidate_without_merging(
    neo4j_store,
):
    _seed_entities(
        neo4j_store,
        [
            {"id": "org:ibm", "domain_label": "Organization", "name": "IBM"},
            {
                "id": "org:international-business-machines",
                "domain_label": "Organization",
                "name": "International Business Machines",
            },
        ],
    )

    result = await neo4j_store.resolve_entities(
        graph_name=GRAPH_NAME,
        strategy="hybrid",
        dry_run=True,
        aliases={"Organization": {"IBM": ["International Business Machines"]}},
    )

    assert result["skipped"] is False
    assert result["signals"]["aliases"] == "used"
    assert result["merged_groups"] == 1
    assert _entity_count(neo4j_store) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_hybrid_uses_real_embedder_and_llm_for_review(neo4j_store):
    _ai_env_or_skip()
    _seed_entities(
        neo4j_store,
        [
            {
                "id": "person:john-smith",
                "domain_label": "Person",
                "name": "John Smith",
                "description": "A person named John Smith.",
            },
            {
                "id": "person:jon-smith",
                "domain_label": "Person",
                "name": "Jon Smith",
                "description": "A person named Jon Smith.",
            },
        ],
    )
    llm = get_llm()
    embedder = get_embedder()

    try:
        result = await neo4j_store.resolve_entities(
            graph_name=GRAPH_NAME,
            strategy="hybrid",
            dry_run=True,
            merge_threshold=95.0,
            review_threshold=85.0,
            embedder=embedder,
            llm=llm,
            llm_guidance=(
                "This is an integration test. Return JSON only. Treat the pair "
                "as a review candidate unless the evidence is conclusive."
            ),
        )
    finally:
        close = getattr(llm, "aclose", None)
        if callable(close):
            await close()

    assert result["skipped"] is False
    assert result["signals"]["embeddings"] == "used"
    assert result["signals"]["llm"] == "used"
    assert result["review_groups"]
    review = result["review_groups"][0]
    assert review["scores"]["embedding"] is not None
    assert "llm_review" in review
    assert "error" not in review["llm_review"]
    assert _entity_count(neo4j_store) == 2
