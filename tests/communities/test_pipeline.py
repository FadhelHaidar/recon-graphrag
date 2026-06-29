"""Tests for the community pipeline wrapper."""

import json

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
        self.reports: list[tuple] = []

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

    def get_child_community_reports(self, graph_name, community_id, level, child_level):
        return []

    def get_community_reports_by_keys(self, graph_name, keys, top_k):
        return []

    def store_community_report(self, report, graph_name):
        self.reports.append((graph_name, report.community_id, report.level, report.summary))

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        pass


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
    assert result["reports"] == 0
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
            return LLMResponse(content=json.dumps({
                "title": "Alice and Acme",
                "summary": "Alice works at Acme.",
                "rating": 7,
                "rating_explanation": "Important relationship.",
                "findings": [{
                    "description": "Alice is connected to Acme.",
                    "references": [{
                        "target_id": "person:alice",
                        "target_type": "entity",
                    }],
                }],
            }))

    pipeline = CommunityPipeline(
        graph_store=store,
        llm=FakeLLM(),
        relationship_types=["ACTED_IN"],
        graph_name="test-graph",
    )

    result = await pipeline.build(level=0)

    # After reversal: levels processed descending, filtered by lvl >= level
    assert result["levels"] == [1, 0]
    assert result["communities"] == 2
    assert result["reports"] == 2
    assert result["level_stats"][0]["level"] == 1
    assert result["level_stats"][1]["level"] == 0
