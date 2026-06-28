"""Tests for the community pipeline wrapper."""

import pytest

from recon_graphrag.communities.pipeline import CommunityPipeline


class FakeGraphStore:
    def __init__(self):
        self.detect_kwargs = None

    def detect_communities(self, **kwargs):
        self.detect_kwargs = kwargs
        return []


class FakeGraphStoreWithCommunities:
    def __init__(self, communities):
        self._communities = communities
        self.detect_kwargs = None
        self.summaries: list[tuple] = []

    def detect_communities(self, **kwargs):
        self.detect_kwargs = kwargs
        return self._communities

    def get_communities(self, graph_name, level=None):
        return [c for c in self._communities if c["level"] == level]

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return [
            {
                "e_id": "person:alice",
                "e_name": "Alice",
                "e_description": "CEO",
                "e_labels": ["Person"],
                "e_degree": 1,
                "rel_type": "WORKS_AT",
                "rel_description": "Alice works at Acme",
                "observation_count": 1,
                "combined_degree": 2,
                "other_id": "org:acme",
                "other_name": "Acme",
                "other_description": "Company",
                "other_labels": ["Organization"],
                "other_degree": 1,
            }
        ]

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def store_community_summary(self, community_id, level, summary, graph_name):
        self.summaries.append((graph_name, community_id, level, summary))


@pytest.mark.asyncio
async def test_build_forwards_relationship_weight_property():
    store = FakeGraphStore()
    pipeline = CommunityPipeline(
        graph_store=store,
        llm=object(),
        relationship_types=["ACTED_IN"],
        relationship_weight_property="weight",
    )

    result = await pipeline.build()

    assert result["communities"] == 0
    assert result["summaries"] == 0
    assert result["levels"] == []
    assert store.detect_kwargs["relationship_weight_property"] == "weight"


@pytest.mark.asyncio
async def test_build_forwards_graph_name():
    store = FakeGraphStore()
    pipeline = CommunityPipeline(
        graph_store=store,
        llm=object(),
        relationship_types=["ACTED_IN"],
        graph_name="my-graph",
    )

    await pipeline.build()

    assert store.detect_kwargs["graph_name"] == "my-graph"


@pytest.mark.asyncio
async def test_build_filters_levels():
    communities = [
        {"id": "c1", "level": 0, "entity_count": 1},
        {"id": "c2", "level": 1, "entity_count": 1},
    ]
    store = FakeGraphStoreWithCommunities(communities)

    class FakeLLM:
        async def ainvoke(self, prompt):
            from recon_graphrag.llm.base import LLMResponse
            return LLMResponse(content="Test summary.")

    pipeline = CommunityPipeline(
        graph_store=store,
        llm=FakeLLM(),
        relationship_types=["ACTED_IN"],
        graph_name="test-graph",
    )

    result = await pipeline.build(level=0)

    assert result["levels"] == [0]
    assert result["communities"] == 2
    assert result["summaries"] == 1
    assert result["level_stats"][0]["level"] == 0
