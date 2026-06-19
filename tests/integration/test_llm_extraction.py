"""Integration test for real LLM entity extraction.

This test exercises the full extraction path with a live LLM. It does not need a
running graph database because it uses an in-memory fake graph store and disables
entity resolution/embedding.

Run with:
    RUN_LLM_EXTRACTION_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py -v

Required environment variables depend on LLM_PROVIDER (default: openai):
    - openai: OPENAI_API_KEY, OPENAI_LLM_MODEL
    - azure_openai: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
                    AZURE_OPENAI_LLM_DEPLOYMENT_NAME
    - openrouter: OPENROUTER_API_KEY, OPENROUTER_LLM_MODEL
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from recon_graphrag import (
    GraphBuilderPipeline,
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from examples.config import get_llm
from tests.integration.support import PROVIDER_REQUIREMENTS, require_integration_env

RUN_FLAG = "RUN_LLM_EXTRACTION_INTEGRATION_TESTS"


class FakeGraphStore:
    """In-memory graph store that captures the assembled GraphDocument."""

    def __init__(self):
        self.graph_document = None

    def write_graph_document(self, graph_document):
        self.graph_document = graph_document
        return {
            "documents": 1,
            "chunks": len(graph_document.chunks),
            "entities": len(graph_document.entities),
            "relationships": len(graph_document.relationships),
            "evidence_links": len(graph_document.evidence_links),
        }

    def backfill_descriptions(self):
        pass

    async def resolve_entities(self, **kwargs):
        return {"skipped": True, "reason": "disabled in test"}

    def get_unembedded_entities(self, limit=500):
        return []

    def upsert_vectors(self, ids, property_name, vectors):
        pass

    def validate_graph_build(self):
        return {}


@pytest.fixture
def movie_schema():
    return GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                properties=[PropertyType(name="name", type="STRING")],
            ),
            NodeType(
                label="Movie",
                properties=[PropertyType(name="title", type="STRING")],
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED"),
            RelationshipType(label="ACTED_IN"),
        ],
        patterns=[
            ("Person", "DIRECTED", "Movie"),
            ("Person", "ACTED_IN", "Movie"),
        ],
    )


def _llm_env_or_skip():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    required = PROVIDER_REQUIREMENTS.get(provider, {}).get("llm", [])
    require_integration_env(
        RUN_FLAG,
        required,
        f"{provider} LLM extraction integration test",
    )
    return provider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_llm_extracts_movie_entities(movie_schema):
    _llm_env_or_skip()
    llm = get_llm()

    try:
        pipeline = GraphBuilderPipeline(
            graph_store=FakeGraphStore(),
            llm=llm,
            embedder=MagicMock(),
            schema=movie_schema,
            chunk_size=500,
            chunk_overlap=50,
            extraction_concurrency=2,
            perform_entity_resolution=False,
            embed_entities=False,
        )

        text = (
            "Christopher Nolan directed the movie Inception in 2010. "
            "Leonardo DiCaprio starred as Dom Cobb, and the film won several Academy Awards. "
            "Nolan also directed The Dark Knight, which starred Christian Bale as Batman."
        )

        result = await pipeline.build_from_text(
            text,
            metadata={"source": "integration-test"},
        )

        assert result["extraction"]["chunks"] >= 1

        graph_document = pipeline.graph_store.graph_document
        assert graph_document is not None
        assert len(graph_document.entities) >= 2
        assert len(graph_document.relationships) >= 1

        person_names = {
            entity.properties.get("name", "").lower()
            for entity in graph_document.entities
            if entity.type == "Person"
        }
        movie_titles = {
            entity.properties.get("title", "").lower()
            for entity in graph_document.entities
            if entity.type == "Movie"
        }

        assert any(
            "christopher nolan" in name or name.endswith("nolan")
            for name in person_names
            if name
        ), f"Expected Christopher Nolan among people, got: {person_names}"
        assert any(
            "inception" in title for title in movie_titles if title
        ), f"Expected Inception among movies, got: {movie_titles}"

        relationship_types = {rel.type for rel in graph_document.relationships}
        assert "DIRECTED" in relationship_types, (
            f"Expected DIRECTED relationship, got: {relationship_types}"
        )
    finally:
        await llm.aclose()
