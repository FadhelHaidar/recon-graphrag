"""Integration test for real LLM entity extraction.

This test exercises the full extraction path with a live LLM. It does not need a
running graph database because it uses an in-memory fake graph store and disables
entity resolution/embedding.

Run with:
    RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py -v

Required environment variables depend on LLM_PROVIDER (default: azure_openai):
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

from recon_graphrag import GraphBuilderPipeline
from tests.integration.factories import get_llm
from tests.integration.support import PROVIDER_REQUIREMENTS, require_integration_env
from tests.integration.synthetic_e2e_support import (
    SYNTHETIC_PAGES,
    SYNTHETIC_SCHEMA,
    assert_extraction_invariants,
)


RUN_FLAG = "RUN_PROVIDER_INTEGRATION_TESTS"


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


def _llm_env_or_skip():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    required = PROVIDER_REQUIREMENTS.get(provider, {}).get("llm", [])
    require_integration_env(
        RUN_FLAG,
        required,
        f"{provider} LLM extraction integration test",
    )
    return provider


@pytest.mark.integration
@pytest.mark.provider
async def test_real_llm_extracts_entities_structurally():
    """Live LLM extracts entities and relationships matching the schema.

    Assertions are structural only — no exact names, titles, or wording.
    """
    provider = _llm_env_or_skip()
    llm = get_llm(provider)

    try:
        pipeline = GraphBuilderPipeline(
            graph_store=FakeGraphStore(),
            llm=llm,
            embedder=MagicMock(),
            schema=SYNTHETIC_SCHEMA,
            extraction_concurrency=2,
            perform_entity_resolution=False,
            embed_entities=False,
        )

        combined_text = "\n".join(page["text"] for page in SYNTHETIC_PAGES)

        result = await pipeline.build_from_text(
            combined_text,
            metadata={"source": "synthetic-e2e-test"},
            chunk_size=500,
            chunk_overlap=50,
        )

        assert result["extraction"]["chunks"] >= 1

        graph_document = pipeline.graph_store.graph_document
        assert_extraction_invariants(graph_document, SYNTHETIC_SCHEMA)
    finally:
        await llm.aclose()
