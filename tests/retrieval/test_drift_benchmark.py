"""Deterministic invariants for the manual fake-only DRIFT benchmark."""
import asyncio
from collections import Counter

import pytest

from recon_graphrag.retrieval import search_drift
from recon_graphrag.utils.tokens import ApproximateTokenCounter
from tests.manual import benchmark_drift
from tests.manual.benchmark_drift import instrumentation, run_benchmark


def test_drift_benchmark_attribution_and_shared_search_isolation():
    original_thread = asyncio.to_thread
    original_count = ApproximateTokenCounter.count
    original_citations = search_drift.resolve_chunk_citations
    report = asyncio.run(run_benchmark(db_latency=0, llm_latency=0, embed_latency=0,
                                     runs=2, concurrency=2, workers=1,
                                     repair=True, graph=True, shared=True))
    assert asyncio.to_thread is original_thread
    assert ApproximateTokenCounter.count is original_count
    assert search_drift.resolve_chunk_citations is original_citations
    assert report["fake_only"]
    rows = report["searches"]
    assert len(rows) == 4
    assert len({row["fingerprint"] for row in rows}) == 1
    for row in rows:
        counts = row["counts"]
        # Two rounds select two questions each from the shared pending pool.
        assert row["actions"] == 5
        assert row["llm_calls"] == 8
        assert counts["llm.hyde"] == counts["llm.primer_repair"] == counts["llm.reduce"] == 1
        assert counts["llm.action"] == 4
        assert counts["embed"] == 6
        assert counts["context.mixed"] == 4
        assert counts["db.fetch_community_member_graph"] == 1
        assert counts["db.keyword_search"] == counts["db.vector_search"] == 4
        assert row["store_calls"] == row["offloaded_calls"] == 41
        assert row["repeated_db_calls"] == 27
        assert sum(group["count"] for group in row["db_argument_groups"]) == row["store_calls"]
        assert sum(n for name, n in counts.items() if name.startswith("llm.")) == row["llm_calls"]
        assert counts["executor.queue"] == counts["executor.execution"] == counts["executor.offload"]
        assert counts["tokens.ApproximateTokenCounter.count"] > 0
        assert row["failures"] == []
        assert row["semantic"]["citations"]
        assert row["semantic"]["graph"]["edges"]
        assert set(row["phase_ms"]) == {"primer", "traversal", "reduce", "total"}
        assert row["phase_ms"]["total"] <= row["total_ms"]
        spans = row["spans"]
        assert Counter(span["name"] for span in spans) == counts
        for name, total in row["inclusive_span_ms"].items():
            assert total == pytest.approx(sum(s["ms"] for s in spans if s["name"] == name))
        for span in spans:
            assert span["ms"] >= 0
            if span["parent"] is not None:
                parent = spans[span["parent"]]
                assert parent["start_ms"] <= span["start_ms"]
                assert span["start_ms"] + span["ms"] <= parent["start_ms"] + parent["ms"] + .1


def test_drift_benchmark_fingerprint_independent_of_scheduling_and_latencies():
    fast = asyncio.run(run_benchmark(db_latency=0, llm_latency=0, embed_latency=0))["searches"][0]
    slow = asyncio.run(run_benchmark(db_latency=.001, llm_latency=.001,
                                   embed_latency=.001, workers=1))["searches"][0]
    assert fast["fingerprint"] == slow["fingerprint"]
    assert fast["actions"] == 5
    assert fast["llm_calls"] == 7
    assert fast["counts"].get("llm.primer_repair", 0) == 0
    assert fast["counts"].get("db.fetch_community_member_graph", 0) == 0


def test_drift_benchmark_patches_restore_on_error():
    original = asyncio.to_thread
    with pytest.raises(RuntimeError), instrumentation():
        raise RuntimeError("stop")
    assert asyncio.to_thread is original


@pytest.mark.parametrize("cancel", [False, True])
def test_drift_benchmark_drains_siblings_before_restoring_patches(monkeypatch, cancel):
    original_thread = asyncio.to_thread
    tasks = []
    cleaned = []

    async def scenario():
        ready = asyncio.Event()

        async def search(*args, **kwargs):
            index = len(tasks)
            tasks.append(asyncio.current_task())
            if index == 0:
                await ready.wait()
                if not cancel:
                    raise RuntimeError("failed search")
                await asyncio.Event().wait()
            else:
                ready.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    assert asyncio.to_thread is not original_thread
                    # Cleanup may still need the executor before it is shut down.
                    cleaned.append(await asyncio.to_thread(lambda: "drained"))

        monkeypatch.setattr(benchmark_drift, "measure", search)
        batch = asyncio.create_task(run_benchmark(concurrency=2, workers=1))
        if cancel:
            await ready.wait()
            batch.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await batch
        assert cleaned == ["drained"]
        assert len(tasks) == 2 and all(task.done() for task in tasks)
        assert asyncio.to_thread is original_thread

    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [{"runs": 0}, {"workers": 0}, {"concurrency": 0},
                                   {"db_latency": -1}, {"llm_latency": float("nan")},
                                   {"embed_latency": float("inf")}])
def test_drift_benchmark_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        asyncio.run(run_benchmark(**kwargs))
