"""Backend-neutral end-to-end smoke test with 5-page corpus.

Validates the complete pipeline: extraction -> aggregation -> community ->
search, and checks that PR 5/6 features (observation_count, strength,
description_observations) are populated correctly.

Runs on both Neo4j and Memgraph when their respective run flags are set.
"""

from __future__ import annotations

import os

import pytest

from examples.config import get_embedder, get_llm, get_memgraph_store, get_neo4j_store
from examples.prompts import (
    COMMUNITY_SUMMARY_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
    LOCAL_ANSWER_PROMPT,
)
from examples.schema import COMMUNITY_RELATIONSHIP_TYPES, MOVIE_SCHEMA
from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, GraphRAG, IndexManager
from tests.integration.movie_smoke_support import cleanup_graph, close_resources, single_count
from tests.integration.support import require_integration_env, require_selected_provider_env

# 5-page corpus covering key entities and cross-document observations
SMOKE_PAGES = [
    # Page 1: Nolan, Interstellar, Hathaway, Zimmer
    """
    Christopher Nolan directed Interstellar (2014), a science-fiction film starring
    Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Casey Affleck.
    Hans Zimmer composed the score for Interstellar. Zimmer's scores often use
    Shepard tones, a musical illusion associated with rising tension.
    """,
    # Page 2: Inception, DiCaprio, Zimmer again (cross-doc observation)
    """
    Leonardo DiCaprio starred in Inception (2010), directed by Christopher Nolan.
    DiCaprio played Dom Cobb, a thief who extracts secrets through dream-sharing
    technology. Hans Zimmer also composed the score for Inception, using deep brass
    sounds and auditory tension.
    """,
    # Page 3: Dark Knight Rises, Hardy, Murphy, Hathaway again
    """
    The Dark Knight Rises (2012) was directed by Christopher Nolan and featured
    Christian Bale as Batman, Tom Hardy as Bane, and Anne Hathaway as Selina Kyle.
    Cillian Murphy appeared as Dr. Jonathan Crane, also known as Scarecrow.
    """,
    # Page 4: Dune, Villeneuve, Zimmer again, Chalamet
    """
    Denis Villeneuve directed Dune (2021), a science-fiction epic starring
    Timothee Chalamet as Paul Atreides. Hans Zimmer composed the score for Dune,
    connecting it to Interstellar and Inception through his musical style.
    """,
    # Page 5: Oppenheimer, Murphy again, awards
    """
    Oppenheimer (2023) was directed by Christopher Nolan and starred Cillian
    Murphy as J. Robert Oppenheimer. The film won the Oscar for Best Picture
    at the 2024 Academy Awards. Hoyte van Hoytema served as cinematographer.
    """,
]


def _make_store_fixture(store_factory, run_flag, required_env, graph_name):
    """Create a parameterized store fixture."""

    @pytest.fixture
    def fixture():
        require_integration_env(run_flag, required_env, "Full smoke test", fail_on_missing=True)
        store = store_factory()
        cleanup_graph(store, graph_name)
        yield store, graph_name

    return fixture


# Neo4j fixture
neo4j_smoke = _make_store_fixture(
    get_neo4j_store,
    "RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS",
    ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "smoke-full-neo4j",
)

# Memgraph fixture
memgraph_smoke = _make_store_fixture(
    get_memgraph_store,
    "RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS",
    ["MEMGRAPH_URL"],
    "smoke-full-memgraph",
)


async def _run_smoke_pipeline(store, graph_name: str):
    """Run the full pipeline and validate all features."""
    require_selected_provider_env("Full smoke test")
    llm = get_llm()
    embedder = get_embedder()

    try:
        # --- Index setup ---
        dimension_probe = await embedder.async_embed_query("probe")
        IndexManager(store, embedding_dim=len(dimension_probe)).create_indexes()

        # --- Build graph ---
        builder = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            schema=MOVIE_SCHEMA,
            graph_name=graph_name,
            extract_claims=True,
        )
        build_result = await builder.build_from_pages(
            SMOKE_PAGES,
            metadata={"source": f"{graph_name}-source"},
            window_size=2,
            window_overlap=1,
        )

        # Validate build
        validation = build_result.get("validation", {})
        for key in ("chunk_count", "entity_count", "evidence_link_count", "entity_relationship_count"):
            assert validation.get(key, 0) > 0, f"Expected {key} > 0, got {validation.get(key)}"

        # --- Validate PR 5/6 features: observation semantics ---
        writer_doc = build_result.get("graph_document")
        if writer_doc:
            entities_with_observations = [
                e for e in writer_doc.entities if e.description_observations
            ]
            assert len(entities_with_observations) > 0, (
                "Expected some entities to have description_observations"
            )

            for entity in writer_doc.entities:
                if entity.description_observations:
                    has_non_empty = any(
                        o.description.strip() for o in entity.description_observations
                    )
                    if has_non_empty:
                        assert entity.properties.get("description", "").strip(), (
                            f"Entity {entity.id} has observations but empty consolidated description"
                        )

            for rel in writer_doc.relationships:
                assert rel.observation_count >= 1, (
                    f"Relationship {rel.id} has observation_count < 1"
                )

            for entity in entities_with_observations:
                for obs in entity.description_observations:
                    assert obs.source.document_id, "Source missing document_id"
                    assert obs.source.chunk_id, "Source missing chunk_id"

        # --- Validate claims (Phase 5A) ---
        write_stats = build_result.get("extraction", {}).get("write_stats", {})
        claims_written = write_stats.get("claims", 0)
        print(f"Claims written: {claims_written}")

        claim_count_query = "MATCH (c:Claim {graph_name: $graph_name}) RETURN count(c) AS count"
        claim_count = single_count(store, claim_count_query, graph_name)
        print(f"Claim nodes in graph: {claim_count}")
        # Claims depend on LLM output; at minimum the pipeline should not crash
        # when extract_claims=True

        # --- Graph shape ---
        assert all(
            single_count(store, q, graph_name) > 0
            for q in [
                "MATCH (d:Document {graph_name: $graph_name}) RETURN count(d) AS count",
                "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
                "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN count(e) AS count",
            ]
        )

        # --- Community detection ---
        community = CommunityPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
            graph_name=graph_name,
            summary_prompt=COMMUNITY_SUMMARY_PROMPT,
        )
        community_result = await community.build()
        assert community_result.get("communities", 0) > 0
        assert community_result.get("summaries", 0) > 0

        # --- Search ---
        graph_rag = GraphRAG(store, llm, embedder, graph_name=graph_name)
        graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
        graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
        graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT
        graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT

        results = [
            await graph_rag.search(
                "Which movies were directed by Christopher Nolan?",
                mode="local",
            ),
            await graph_rag.search(
                "What are the most common themes across science-fiction films?",
                mode="global",
                community_level="coarsest",
            ),
            await graph_rag.search(
                "How does Hans Zimmer connect Inception to Dune?",
                mode="drift",
                community_level="finest",
            ),
        ]

        for result in results:
            assert result.answer.strip(), f"Empty answer for {result.mode} search"
            assert result.context.strip(), f"Empty context for {result.mode} search"

        print(f"\n{'='*60}")
        print(f"SMOKE TEST PASSED ({graph_name})")
        print(f"{'='*60}")
        print(f"Build: {validation}")
        print(f"Communities: {community_result}")
        for r in results:
            print(f"\n[{r.mode.upper()}] {r.answer[:200]}...")

    finally:
        await close_resources(store, llm)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smoke_neo4j(neo4j_smoke):
    store, graph_name = neo4j_smoke
    await _run_smoke_pipeline(store, graph_name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smoke_memgraph(memgraph_smoke):
    store, graph_name = memgraph_smoke
    await _run_smoke_pipeline(store, graph_name)
