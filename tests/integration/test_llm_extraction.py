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

    def execute_query(self, query, parameters=None):
        return []

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

    def get_entities_needing_summary(self, graph_name, limit=500):
        return []

    def get_relationships_needing_summary(self, graph_name, limit=500):
        return []

    def persist_entity_summaries(self, graph_name, summaries):
        pass

    def persist_relationship_summaries(self, graph_name, summaries):
        pass

    async def resolve_entities_exact(self, graph_name="entity-graph", resolve_property="name", dry_run=False, **kwargs):
        return {"skipped": True, "reason": "disabled in test"}

    async def resolve_entities_normalized(self, graph_name="entity-graph", resolve_property="name", dry_run=False, **kwargs):
        return {"skipped": True, "reason": "disabled in test"}

    async def resolve_entities_fuzzy(self, graph_name="entity-graph", resolve_property="name", dry_run=False,
                                     merge_threshold=95.0, review_threshold=85.0, max_candidates_per_entity=20, **kwargs):
        return {"skipped": True, "reason": "disabled in test"}

    async def resolve_entities_hybrid(self, graph_name="entity-graph", resolve_property="name", dry_run=False,
                                      merge_threshold=95.0, review_threshold=85.0, max_candidates_per_entity=20,
                                      aliases=None, embedder=None, llm=None, llm_guidance=None,
                                      allow_ai_auto_merge=False, context_properties=None, conflict_properties=None,
                                      context_mode="safe_defaults", **kwargs):
        return {"skipped": True, "reason": "disabled in test"}

    def get_unembedded_entities(self, graph_name, limit=500, model_fingerprint=None):
        return []

    def upsert_vectors(self, node_ids, embedding_property, vectors, **kwargs):
        pass

    def validate_graph_build(self):
        return {}

    def create_indexes(self, config=None, embedding_dim=1536):
        pass

    def drop_indexes(self, config=None):
        pass

    def create_vector_index(self, name, label, embedding_property, dimensions, similarity_fn="cosine"):
        pass

    def create_fulltext_index(self, name, label, node_properties):
        pass

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        return []

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        return []

    def fetch_entity_context(self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None):
        return []

    def detect_communities(self, graph_name, relationship_types=None, max_levels=3, gamma=1.0,
                           theta=0.01, tolerance=1e-4, relationship_weight_property=None,
                           random_seed=None, entity_label="__Entity__", community_label="Community"):
        return []

    def get_communities(self, graph_name, level=None):
        return []

    def get_community_stats(self, graph_name):
        return []

    def store_community_report(self, report, graph_name):
        pass

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        pass

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return []

    def get_child_community_reports(self, graph_name, community_id, level, child_level):
        return []

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return []

    def get_community_reports_by_keys(self, graph_name, keys, top_k):
        return []

    def get_unembedded_community_reports(self, graph_name, limit=500):
        return []

    def upsert_community_report_vectors(self, node_ids, vectors, graph_name, levels):
        pass

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        return []

    def get_entity_count(self):
        return 0

    def get_chunk_count(self):
        return 0

    def get_evidence_link_count(self):
        return 0

    def get_relationship_count(self):
        return 0


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
            perform_entity_resolution=False,
            embed_entities=False,
        )

        combined_text = "\n".join(page["text"] for page in SYNTHETIC_PAGES)

        result = await pipeline.build_from_text(
            combined_text,
            metadata={"source": "synthetic-e2e-test"},
            schema=SYNTHETIC_SCHEMA,
            extraction_concurrency=2,
            chunk_size=500,
            chunk_overlap=50,
        )

        assert result["extraction"]["chunks"] >= 1

        graph_document = pipeline.graph_store.graph_document  # type: ignore[attr-defined]
        assert_extraction_invariants(graph_document, SYNTHETIC_SCHEMA)
    finally:
        await llm.aclose()  # type: ignore[attr-defined]
