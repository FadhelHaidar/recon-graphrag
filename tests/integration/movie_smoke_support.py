"""Backend-neutral assertions for the movie example smoke tests."""

from __future__ import annotations

from examples.data import MOVIE_EXAMPLE_PAGES
from examples.prompts import (
    COMMUNITY_SUMMARY_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
    LOCAL_ANSWER_PROMPT,
)
from examples.schema import COMMUNITY_RELATIONSHIP_TYPES, MOVIE_SCHEMA
from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, GraphRAG


def cleanup_graph(store, graph_name: str) -> None:
    """Delete only data owned by one smoke-test graph."""
    store.execute_query(
        "MATCH (n {graph_name: $graph_name}) DETACH DELETE n",
        {"graph_name": graph_name},
    )


def single_count(store, query: str, graph_name: str) -> int:
    result = store.execute_query(query, {"graph_name": graph_name})
    if not result:
        return 0
    return int(next(iter(result[0].values())))


def assert_graph_shape(store, graph_name: str) -> None:
    queries = [
        "MATCH (d:Document {graph_name: $graph_name}) RETURN count(d) AS count",
        "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
        "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN count(e) AS count",
        """
        MATCH (:Chunk {graph_name: $graph_name})-[:FROM_CHUNK]->
              (:__Entity__ {graph_name: $graph_name})
        RETURN count(*) AS count
        """,
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[r]-
              (:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        RETURN count(r) AS count
        """,
    ]
    assert all(single_count(store, query, graph_name) > 0 for query in queries)


def configure_movie_prompts(graph_rag: GraphRAG) -> None:
    graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
    graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
    graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT
    graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT


async def run_movie_smoke(
    *,
    store,
    index_manager_cls,
    llm,
    embedder,
    graph_name: str,
) -> None:
    """Build, cluster, and search the movie graph against one backend."""
    dimension_probe = await embedder.async_embed_query(
        "Movie example smoke embedding dimension probe"
    )
    index_manager_cls(store, embedding_dim=len(dimension_probe)).create_indexes()

    builder = GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=embedder,
        schema=MOVIE_SCHEMA,
        graph_name=graph_name,
    )
    build_result = await builder.build_from_pages(
        MOVIE_EXAMPLE_PAGES,
        metadata={"source": f"{graph_name}-source"},
        window_size=2,
        window_overlap=1,
    )

    validation = build_result.get("validation", {})
    for key in (
        "chunk_count",
        "entity_count",
        "evidence_link_count",
        "entity_relationship_count",
    ):
        assert validation.get(key, 0) > 0
    assert_graph_shape(store, graph_name)

    community = CommunityPipeline(
        graph_store=store,
        llm=llm,
        relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
        graph_name=graph_name,
        summary_prompt=COMMUNITY_SUMMARY_PROMPT,
    )
    community_result = await community.build()
    assert community_result.get("communities", 0) > 0
    assert community_result.get("summaries", 0) > 0
    assert single_count(
        store,
        """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE coalesce(c.report_text, c.summary, '') <> ''
        RETURN count(c) AS count
        """,
        graph_name,
    ) > 0

    graph_rag = GraphRAG(store, llm, embedder, graph_name=graph_name)
    configure_movie_prompts(graph_rag)
    results = [
        await graph_rag.search(
            "Which movies were directed by Christopher Nolan and feature Cillian Murphy?",
            mode="local",
        ),
        await graph_rag.search(
            "What are the most common themes across the science-fiction films?",
            mode="global",
            community_level="coarsest",
        ),
        await graph_rag.search(
            "How does Hans Zimmer connect Inception to Dune?",
            mode="drift",
            community_level="finest",
        ),
    ]
    assert all(result.answer.strip() and result.context.strip() for result in results)


async def close_resources(store, llm) -> None:
    close = getattr(llm, "aclose", None)
    try:
        if callable(close):
            await close()
    finally:
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()
