"""Tests for the community pipeline wrapper."""

import asyncio
import json
import logging
import threading
from typing import Any, cast

import pytest

from recon_graphrag.communities.pipeline import CommunityPipeline
from recon_graphrag.utils.tokens import TiktokenTokenCounter
from tests.pipelines.test_async_offload import BlockingCalls, with_heartbeat


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


class GuardedPipelineStore(FakeGraphStoreWithCommunities):
    """Guarded store double with scripted generation reads."""

    def __init__(
        self,
        communities,
        *,
        detected_generation: str = "generation-a",
        generation_reads=None,
    ):
        super().__init__(communities)
        self.detected_generation = detected_generation
        self.generation_reads = list(generation_reads or [])
        self.read_levels: list[int] = []
        self.community_levels_read: list[int | None] = []
        self.public_detect_calls = 0
        self.public_store_calls = 0
        self.guarded_reports: list[tuple[str, str, str | None]] = []
        self.guarded_failures: list[tuple[str, str, int, str, str | None]] = []
        self.report_results: list[bool] = []

    def detect_communities(self, **kwargs):
        self.public_detect_calls += 1
        return self._communities

    def _detect_communities_with_generation(self, **kwargs):
        self.detect_kwargs = kwargs
        return self._communities, self.detected_generation

    def _get_community_generations(self, graph_name, level):
        self.read_levels.append(level)
        if not self.generation_reads:
            return [self.detected_generation]
        observed = self.generation_reads.pop(0)
        return observed if isinstance(observed, list) else [observed]

    def _store_community_report_guarded(
        self, report, graph_name, *, expected_generation
    ):
        self.guarded_reports.append(
            (graph_name, report.community_id, expected_generation)
        )
        return self.report_results.pop(0) if self.report_results else True

    def _mark_community_report_failed_guarded(
        self,
        graph_name,
        community_id,
        level,
        error,
        *,
        expected_generation,
    ):
        self.guarded_failures.append(
            (graph_name, community_id, level, error, expected_generation)
        )
        return True

    def _get_unembedded_community_reports_for_embedding(self, *args, **kwargs):
        return []

    def _upsert_community_report_vectors_guarded(self, *args, **kwargs):
        return 0

    def _mark_community_report_embedding_failed_guarded(self, *args, **kwargs):
        return True

    def get_communities(self, graph_name, level=None):
        self.community_levels_read.append(level)
        return super().get_communities(graph_name, level=level)

    def store_community_report(self, report, graph_name):
        self.public_store_calls += 1
        super().store_community_report(report, graph_name)


class PipelineReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt, **kwargs):
        from recon_graphrag.llm.base import LLMResponse

        self.calls += 1
        return LLMResponse(json.dumps({
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


class PipelineFakeEmbedder:
    async def async_embed_query(self, text):
        return [0.1, 0.2]


def _pipeline(store, llm=None, **kwargs):
    return CommunityPipeline(
        cast(Any, store),
        cast(Any, llm or PipelineReportLLM()),
        relationship_types=["ACTED_IN"],
        graph_name="test-graph",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_guarded_build_binds_private_detection_and_counts_acknowledged_reports():
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-a"],
    )

    result = await _pipeline(store).build()

    assert result["status"] == "completed"
    assert result["maintenance_warnings"] == []
    assert result["reports"] == 1
    assert store.public_detect_calls == 0
    assert store.public_store_calls == 0
    assert store.guarded_reports == [
        ("test-graph", "c0", "generation-a")
    ]
    assert store.read_levels == [0, 0]


@pytest.mark.asyncio
async def test_guarded_build_stops_before_the_first_replaced_level():
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-b"],
    )
    llm = PipelineReportLLM()

    result = await _pipeline(store, llm).build()

    assert result["status"] == "degraded"
    assert result["maintenance_warnings"] == [
        {"code": "community_generation_changed", "level": 0}
    ]
    assert result["reports"] == 0
    assert result["level_stats"] == []
    assert store.community_levels_read == []
    assert store.guarded_reports == []
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_guarded_build_stops_before_the_next_replaced_level():
    store = GuardedPipelineStore(
        [
            {"id": "c1", "level": 1, "entity_count": 2},
            {"id": "c0", "level": 0, "entity_count": 2},
        ],
        generation_reads=["generation-a", "generation-b"],
    )
    llm = PipelineReportLLM()

    result = await _pipeline(store, llm).build()

    assert result["status"] == "degraded"
    assert result["levels"] == [1, 0]
    assert [stats["level"] for stats in result["level_stats"]] == [1]
    assert result["reports"] == 1
    assert result["maintenance_warnings"] == [
        {"code": "community_generation_changed", "level": 0}
    ]
    assert store.community_levels_read == [1]
    assert store.guarded_reports == [
        ("test-graph", "c1", "generation-a")
    ]
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_guarded_build_stops_after_a_stale_report_rejection():
    store = GuardedPipelineStore(
        [
            {"id": "c1", "level": 1, "entity_count": 2},
            {"id": "c0", "level": 0, "entity_count": 2},
        ],
        generation_reads=["generation-a"],
    )
    store.report_results = [False]
    llm = PipelineReportLLM()

    result = await _pipeline(store, llm).build()

    assert result["status"] == "degraded"
    assert result["reports"] == 0
    assert [stats["level"] for stats in result["level_stats"]] == [1]
    assert store.community_levels_read == [1]
    assert store.guarded_failures == []
    codes = [warning["code"] for warning in result["maintenance_warnings"]]
    assert codes == [
        "community_report_generation_failed",
        "community_generation_changed",
    ]
    assert result["maintenance_warnings"][1] == {
        "code": "community_generation_changed",
        "level": 1,
    }


@pytest.mark.asyncio
async def test_guarded_build_detects_supersession_after_the_last_report():
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-b"],
    )

    result = await _pipeline(store).build()

    assert result["status"] == "degraded"
    assert result["reports"] == 1
    assert result["maintenance_warnings"] == [
        {"code": "community_generation_changed", "level": 0}
    ]


@pytest.mark.asyncio
async def test_guarded_build_skips_embedding_when_replaced_before_embedding(
    monkeypatch,
):
    from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder

    calls: list[tuple[str, str | None]] = []

    async def fake_embed(self, graph_name, expected_generation):
        calls.append((graph_name, expected_generation))
        return 2

    monkeypatch.setattr(
        CommunityReportEmbedder,
        "_embed_reports_for_generation",
        fake_embed,
        raising=False,
    )
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-b"],
    )

    result = await _pipeline(store, embedder=PipelineFakeEmbedder()).build()

    assert result["status"] == "degraded"
    assert result["embedded_reports"] == 0
    assert calls == []
    assert result["maintenance_warnings"] == [
        {"code": "community_generation_changed", "level": 0}
    ]


@pytest.mark.asyncio
async def test_guarded_build_carries_the_bound_generation_to_embedding(
    monkeypatch,
):
    from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder

    calls: list[tuple[str, str | None]] = []

    async def fake_embed(self, graph_name, expected_generation):
        calls.append((graph_name, expected_generation))
        return 2

    monkeypatch.setattr(
        CommunityReportEmbedder,
        "_embed_reports_for_generation",
        fake_embed,
        raising=False,
    )
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-a", "generation-a"],
    )

    result = await _pipeline(store, embedder=PipelineFakeEmbedder()).build()

    assert result["status"] == "completed"
    assert result["embedded_reports"] == 2
    assert calls == [("test-graph", "generation-a")]
    assert result["maintenance_warnings"] == []


@pytest.mark.asyncio
async def test_guarded_build_detects_supersession_after_the_last_vector(
    monkeypatch,
):
    from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder

    async def fake_embed(self, graph_name, expected_generation):
        return 2

    monkeypatch.setattr(
        CommunityReportEmbedder,
        "_embed_reports_for_generation",
        fake_embed,
        raising=False,
    )
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-a", "generation-b"],
    )

    result = await _pipeline(store, embedder=PipelineFakeEmbedder()).build()

    assert result["status"] == "degraded"
    assert result["embedded_reports"] == 2
    assert result["maintenance_warnings"] == [
        {"code": "community_generation_changed", "level": 0}
    ]


@pytest.mark.asyncio
async def test_custom_store_without_guards_warns_and_keeps_the_public_path(caplog):
    store = FakeGraphStoreWithCommunities(
        [{"id": "c0", "level": 0, "entity_count": 2}]
    )

    with caplog.at_level(logging.WARNING):
        result = await _pipeline(store).build()

    assert result["status"] == "completed"
    assert result["reports"] == 1
    assert store.detect_kwargs["graph_name"] == "test-graph"
    assert "not provide the private community generation guards" in caplog.text


@pytest.mark.asyncio
async def test_store_without_private_detection_keeps_the_public_unguarded_path(
    caplog, monkeypatch
):
    monkeypatch.delattr(
        GuardedPipelineStore,
        "_detect_communities_with_generation",
        raising=False,
    )
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a"],
    )

    with caplog.at_level(logging.WARNING):
        result = await _pipeline(store).build()

    assert result["status"] == "completed"
    assert result["reports"] == 1
    assert store.public_detect_calls == 1
    assert store.public_store_calls == 1
    assert store.guarded_reports == []
    assert "does not provide _detect_communities_with_generation" in caplog.text


@pytest.mark.asyncio
async def test_guarded_build_skips_an_embedder_without_generation_support(
    monkeypatch, caplog
):
    from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder

    async def fail_public_embedding(self, batch_size=500):
        raise AssertionError("public embedding path used")

    monkeypatch.setattr(
        CommunityReportEmbedder, "_embed_reports_for_generation", None
    )
    monkeypatch.setattr(
        CommunityReportEmbedder, "embed_reports", fail_public_embedding
    )
    store = GuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-a"],
    )

    with caplog.at_level(logging.WARNING):
        result = await _pipeline(store, embedder=PipelineFakeEmbedder()).build()

    assert result["embedded_reports"] == 0
    assert result["status"] == "completed"
    assert "report embeddings are skipped" in caplog.text


class BlockingGuardedPipelineStore(GuardedPipelineStore, BlockingCalls):
    """Guarded pipeline store whose reads and writes block their callers."""

    def __init__(self, communities, **kwargs):
        GuardedPipelineStore.__init__(self, communities, **kwargs)
        BlockingCalls.__init__(self)

    def count_communities_with_reports(self, graph_name):
        self.block("preflight")
        return super().count_communities_with_reports(graph_name)

    def _detect_communities_with_generation(self, **kwargs):
        self.block("detection")
        return super()._detect_communities_with_generation(**kwargs)

    def _get_community_generations(self, graph_name, level):
        self.block("generation")
        return super()._get_community_generations(graph_name, level)

    def get_communities(self, graph_name, level=None):
        self.block("communities")
        return super().get_communities(graph_name, level=level)

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        self.block("context")
        return super().get_community_ranked_context(graph_name, community_id, level)

    def get_claims_for_entities(self, graph_name, entity_ids):
        self.block("claims")
        return super().get_claims_for_entities(graph_name, entity_ids)

    def get_child_community_reports(
        self, graph_name, community_id, level, child_level
    ):
        self.block("children")
        return super().get_child_community_reports(
            graph_name, community_id, level, child_level
        )

    def _store_community_report_guarded(
        self, report, graph_name, *, expected_generation
    ):
        self.block("report_write")
        return super()._store_community_report_guarded(
            report, graph_name, expected_generation=expected_generation
        )


@pytest.mark.asyncio
async def test_pipeline_preflight_detection_and_report_reads_run_off_loop():
    # Loading the BPE vocabulary is a one-time ~200ms cost, not a store read.
    TiktokenTokenCounter().count("warm")
    store = BlockingGuardedPipelineStore(
        [{"id": "c0", "level": 0, "entity_count": 2}],
        generation_reads=["generation-a", "generation-a"],
    )

    result = await with_heartbeat(_pipeline(store).build())

    assert result["reports"] == 1
    store.assert_offloaded(
        "preflight",
        "detection",
        "generation",
        "communities",
        "context",
        "claims",
        "children",
        "report_write",
    )


class GatedDetectionStore:
    """Detection blocks on a test gate so cancellation timing is deterministic."""

    def __init__(self):
        self.detection_started = threading.Event()
        self.release_detection = threading.Event()
        self.detection_finished = threading.Event()
        self.detection_calls = 0
        self.follow_on_calls: list[str] = []

    def count_communities_with_reports(self, graph_name):
        return 0

    def _detect_communities_with_generation(self, **kwargs):
        self.detection_calls += 1
        self.detection_started.set()
        self.release_detection.wait(5)
        self.detection_finished.set()
        return [{"id": "c0", "level": 0, "entity_count": 1}], "generation-a"

    def _get_community_generations(self, graph_name, level):
        self.follow_on_calls.append("generation")
        return ["generation-a"]

    def get_communities(self, graph_name, level=None):
        self.follow_on_calls.append("communities")
        return []

    def _store_community_report_guarded(
        self, report, graph_name, *, expected_generation
    ):
        self.follow_on_calls.append("report_write")
        return True

    def _mark_community_report_failed_guarded(
        self, graph_name, community_id, level, error, *, expected_generation
    ):
        self.follow_on_calls.append("failure_write")
        return True


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_detection_and_starts_no_follow_on():
    store = GatedDetectionStore()
    pipeline = _pipeline(store)
    task = asyncio.create_task(pipeline.build())
    assert await asyncio.to_thread(store.detection_started.wait, 2)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    store.release_detection.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert task.cancelled()
    assert store.detection_finished.is_set()
    assert store.detection_calls == 1
    assert store.follow_on_calls == []
