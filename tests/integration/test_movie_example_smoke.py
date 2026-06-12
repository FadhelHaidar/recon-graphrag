"""Opt-in Azure OpenAI + Neo4j smoke test for the movie example."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from examples.movie_industry.config import get_embedder, get_llm, get_neo4j_store
from examples.movie_industry.data import MOVIE_EXAMPLE_PAGES
from examples.movie_industry.prompts import (
    COMMUNITY_SUMMARY_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
    LOCAL_ANSWER_PROMPT,
)
from examples.movie_industry.schema import (
    COMMUNITY_RELATIONSHIP_TYPES,
    MOVIE_SCHEMA,
)
from recon_graphrag import (
    CommunityPipeline,
    GraphBuilderPipeline,
    GraphRAG,
    IndexManager,
)


RUN_FLAG = "RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS"
GRAPH_NAME = "movie-smoke"
REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
    "NEO4J_URL",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]


def _env_or_skip() -> None:
    load_dotenv()

    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to run the Neo4j movie example smoke test.")

    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required smoke test env vars: {', '.join(missing)}")


def _single_count(store, query: str, params: dict | None = None) -> int:
    result = store.execute_query(query, params or {})
    if not result:
        return 0
    return int(next(iter(result[0].values())))


def _preflight_or_fail(store) -> None:
    checks = [
        ("Neo4j connectivity", "RETURN 1 AS ok"),
        ("APOC", "RETURN apoc.version() AS version"),
        ("GDS", "RETURN gds.version() AS version"),
    ]
    for label, query in checks:
        try:
            store.execute_query(query)
        except Exception as exc:
            pytest.fail(f"{label} preflight failed: {exc}")


def _cleanup_smoke_graph(store) -> None:
    store.execute_query(
        """
        MATCH (n {graph_name: $graph_name})
        DETACH DELETE n
        """,
        {"graph_name": GRAPH_NAME},
    )


def _assert_graph_shape(store) -> None:
    params = {"graph_name": GRAPH_NAME}
    assert _single_count(
        store,
        "MATCH (d:Document {graph_name: $graph_name}) RETURN count(d) AS count",
        params,
    ) > 0
    assert _single_count(
        store,
        "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
        params,
    ) > 0
    assert _single_count(
        store,
        "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN count(e) AS count",
        params,
    ) > 0
    assert _single_count(
        store,
        """
        MATCH (:Chunk {graph_name: $graph_name})-[:FROM_CHUNK]->
              (:__Entity__ {graph_name: $graph_name})
        RETURN count(*) AS count
        """,
        params,
    ) > 0
    assert _single_count(
        store,
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[r]-
              (:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        RETURN count(r) AS count
        """,
        params,
    ) > 0


def _configure_movie_prompts(graph_rag: GraphRAG) -> None:
    graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
    graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
    graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT
    graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_movie_example_azure_neo4j_smoke():
    _env_or_skip()

    store = get_neo4j_store()
    llm = get_llm()
    embedder = get_embedder()

    try:
        _preflight_or_fail(store)
        _cleanup_smoke_graph(store)

        dimension_probe = await embedder.async_embed_query(
            "Movie example smoke embedding dimension probe"
        )
        IndexManager(store, embedding_dim=len(dimension_probe)).create_indexes()

        builder = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            schema=MOVIE_SCHEMA,
            graph_name=GRAPH_NAME,
        )
        build_result = await builder.build_from_pages(
            MOVIE_EXAMPLE_PAGES,
            metadata={"source": "movie-example-smoke"},
            window_size=2,
            window_overlap=1,
        )

        validation = build_result.get("validation", {})
        assert validation.get("chunk_count", 0) > 0
        assert validation.get("entity_count", 0) > 0
        assert validation.get("evidence_link_count", 0) > 0
        assert validation.get("entity_relationship_count", 0) > 0
        _assert_graph_shape(store)

        community = CommunityPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
            graph_name=GRAPH_NAME,
            summary_prompt=COMMUNITY_SUMMARY_PROMPT,
        )
        community_result = await community.build()
        assert community_result.get("communities", 0) > 0
        assert community_result.get("summaries", 0) > 0
        assert _single_count(
            store,
            """
            MATCH (c:Community {graph_name: $graph_name})
            WHERE c.embedding IS NOT NULL
            RETURN count(c) AS count
            """,
            {"graph_name": GRAPH_NAME},
        ) > 0

        graph_rag = GraphRAG(store, llm, embedder, graph_name=GRAPH_NAME)
        _configure_movie_prompts(graph_rag)

        local = await graph_rag.search(
            "Which movies in the database were directed by Christopher Nolan "
            "and feature Cillian Murphy?",
            mode="local",
        )
        global_result = await graph_rag.search(
            "What are the most common themes across high-budget sci-fi films "
            "in this collection?",
            mode="global",
            community_level="coarsest",
        )
        drift = await graph_rag.search(
            "How does the work of Hans Zimmer connect Inception to Dune?",
            mode="drift",
            community_level="finest",
        )

        assert local.answer.strip()
        assert local.context.strip()
        assert global_result.answer.strip()
        assert global_result.context.strip()
        assert drift.answer.strip()
        assert drift.context.strip()
    finally:
        close = getattr(llm, "aclose", None)
        if callable(close):
            await close()
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()
