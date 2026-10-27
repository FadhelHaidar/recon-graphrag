"""Tests for the community pipeline wrapper."""

import json
from typing import Any, cast

import pytest

from recon_graphrag.communities.pipeline import CommunityPipeline


class FakeGraphStore:
    def __init__(self):
        self.detect_kwargs: dict = {}

    def count_communities_with_reports(self, graph_name):
        return 0

    def detect_communities(self, **kwargs):
        self.detect_kwargs = kwargs
        return []


class FakeGraphStoreWithCommunities:
    def __init__(self, communities):
        self._communities = communities
        self.detect_kwargs: dict = {}
        self.reports: list[tuple] = []

    def count_communities_with_reports(self, graph_name):
        return 0

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
        graph_store=cast(Any, store),
        llm=cast(Any, object()),
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
        graph_store=cast(Any, store),
        llm=cast(Any, object()),
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
        graph_store=cast(Any, store),
        llm=cast(Any, FakeLLM()),
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


@pytest.mark.asyncio
async def test_build_forwards_report_prompt_to_summarizer():
    communities = [
        {"id": "c1", "level": 0, "entity_count": 1},
    ]
    store = FakeGraphStoreWithCommunities(communities)

    class CapturingLLM:
        def __init__(self):
            self.prompts = []

        async def ainvoke(self, prompt):
            from recon_graphrag.llm.base import LLMResponse
            self.prompts.append(prompt)
            return LLMResponse(content=json.dumps({
                "title": "T",
                "summary": "S",
                "findings": [{
                    "description": "F",
                    "references": [{
                        "target_id": "person:alice",
                        "target_type": "entity",
                    }],
                }],
            }))

    llm = CapturingLLM()
    pipeline = CommunityPipeline(
        graph_store=cast(Any, store),
        llm=cast(Any, llm),
        relationship_types=["ACTED_IN"],
        graph_name="test-graph",
        report_prompt="You are a security analyst. Identify risks.",
    )

    await pipeline.build()

    assert llm.prompts
    assert "You are a security analyst. Identify risks." in llm.prompts[0]



class FailingReportLLM:
    """Fails the report for one community, succeeds for the other."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        from recon_graphrag.llm.base import LLMResponse
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider said: key=sk-secret-value")
        return LLMResponse(content=json.dumps({
            "title": "Alice and Acme",
            "summary": "Alice works at Acme.",
            "rating": 7,
            "rating_explanation": "Important.",
            "findings": [{
                "description": "Alice is connected to Acme.",
                "references": [{"target_id": "person:alice", "target_type": "entity"}],
            }],
        }))


@pytest.mark.asyncio
async def test_build_reports_completed_when_every_report_succeeds():
    store = FakeGraphStoreWithCommunities([{"id": "c0", "level": 0, "entity_count": 2}])

    class OkLLM:
        async def ainvoke(self, prompt):
            from recon_graphrag.llm.base import LLMResponse
            return LLMResponse(content=json.dumps({
                "title": "T", "summary": "S", "rating": 7,
                "rating_explanation": "E",
                "findings": [{"description": "D", "references": [
                    {"target_id": "person:alice", "target_type": "entity"},
                ]}],
            }))

    pipeline = CommunityPipeline(
        graph_store=cast(Any, store), llm=cast(Any, OkLLM()),
        relationship_types=["ACTED_IN"], graph_name="test-graph",
    )

    result = await pipeline.build()

    assert result["status"] == "completed"
    assert result["maintenance_warnings"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("include_error_details", [False, True])
@pytest.mark.parametrize("community_count", [1, 2], ids=["all-failed", "mixed"])
async def test_build_reports_degraded_when_a_report_fails(include_error_details, community_count):
    store = FakeGraphStoreWithCommunities([
        {"id": f"c{i}", "level": 0, "entity_count": 2}
        for i in range(community_count)
    ])
    pipeline = CommunityPipeline(
        graph_store=cast(Any, store), llm=cast(Any, FailingReportLLM()),
        relationship_types=["ACTED_IN"], graph_name="test-graph",
        include_error_details=include_error_details,
    )

    result = await pipeline.build()

    assert result["status"] == "degraded"
    warnings = result["maintenance_warnings"]
    assert warnings == [{
        "code": "community_report_generation_failed", "level": 0, "failed": 1,
        "errors": [{
            "community_id": "c0", "error_type": "RuntimeError",
            "message": (
                "provider said: key=sk-secret-value" if include_error_details
                else "Community report generation or storage failed."
            ),
        }],
    }]
    assert result["level_stats"][0]["failed"] == 1
    assert result["level_stats"][0]["succeeded"] == community_count - 1
    assert result["reports"] == len(store.reports) == community_count - 1
    if not include_error_details:
        assert "sk-secret-value" not in json.dumps(warnings)
    else:
        assert "sk-secret-value" in json.dumps(warnings)


@pytest.mark.asyncio
async def test_build_forwards_report_structured_output_with_usage_wrapper():
    from recon_graphrag.communities.reports import COMMUNITY_REPORT_RESPONSE_FORMAT
    from recon_graphrag.llm.base import LLMResponse

    class LLM:
        supports_structured_output = True

        def __init__(self):
            self.kwargs = []

        async def ainvoke(self, prompt, **kwargs):
            self.kwargs.append(kwargs)
            return LLMResponse(json.dumps({
                "title": "Report", "summary": "Summary", "rating": None,
                "rating_explanation": None, "findings": [{
                    "description": "Finding", "references": [
                        {"target_id": "person:alice", "target_type": "entity"},
                    ],
                }],
            }))

    store = FakeGraphStoreWithCommunities([{"id": "community:1", "level": 0}])
    llm = LLM()
    result = await CommunityPipeline(
        cast(Any, store), cast(Any, llm), report_structured_output=True,
    ).build()
    assert result["reports"] == 1
    assert llm.kwargs == [{"response_format": COMMUNITY_REPORT_RESPONSE_FORMAT}]
    assert result["token_usage"]["totals"]["calls"] == 1


def test_pipeline_rejects_unsupported_report_adapter_before_build():
    class LLM:
        supports_structured_output = False

    with pytest.raises(ValueError, match="does not support report structured output"):
        CommunityPipeline(cast(Any, FakeGraphStore()), cast(Any, LLM()), report_structured_output=True)
