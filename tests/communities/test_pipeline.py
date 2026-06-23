"""Tests for the community pipeline wrapper."""

import pytest

from recon_graphrag.communities.pipeline import CommunityPipeline


class FakeGraphStore:
    def __init__(self):
        self.detect_kwargs = None

    def detect_communities(self, **kwargs):
        self.detect_kwargs = kwargs
        return []


@pytest.mark.asyncio
async def test_build_forwards_relationship_weight_property():
    store = FakeGraphStore()
    pipeline = CommunityPipeline(
        graph_store=store,
        llm=object(),
        embedder=object(),
        relationship_types=["ACTED_IN"],
        relationship_weight_property="weight",
    )

    result = await pipeline.build()

    assert result["communities"] == 0
    assert result["summaries"] == 0
    assert result["levels"] == []
    assert store.detect_kwargs["relationship_weight_property"] == "weight"
