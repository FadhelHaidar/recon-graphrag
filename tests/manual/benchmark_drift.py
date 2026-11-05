"""Fake-only DRIFT attribution harness; see DRIFT_BENCHMARK.md. No providers."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval import mixed_context, search_drift
from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.utils.tokens import ApproximateTokenCounter, TiktokenTokenCounter
from tests.retrieval.test_drift import MixedFakeGraphStore
from tests.retrieval.test_graph_evidence import ExpansionStore

LLM_LATENCY = 0.30
EMBED_LATENCY = 0.10
QUERY = "How does Alice connect to Acme?"
STORE_METHODS = (
    "vector_search", "keyword_search", "fetch_entity_context",
    "vector_search_community_reports", "execute_query", "get_claims_for_entities",
    "resolve_chunk_citations", "fetch_community_member_graph",
)
_active: ContextVar[Any] = ContextVar("benchmark_search", default=None)
_parent: ContextVar[int | None] = ContextVar("benchmark_parent", default=None)
_phase: ContextVar[str] = ContextVar("benchmark_llm_phase", default="unknown")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Measurement:
    def __init__(self):
        self.start = time.perf_counter()
        self.loop_thread = threading.get_ident()
        self.spans: list[dict] = []
        self.lock = threading.Lock()
        self.prompts: list[dict] = []

    @contextmanager
    def span(self, name: str, **details: Any):
        with self.lock:
            start = time.perf_counter()
            row = {"id": len(self.spans), "parent": _parent.get(), "name": name,
                   "start_ms": (start - self.start) * 1000,
                   "thread": threading.get_ident(), **details}
            self.spans.append(row)
        token = _parent.set(row["id"])
        try:
            yield row
        finally:
            row["ms"] = (time.perf_counter() - start) * 1000
            _parent.reset(token)


def timed(original, name):
    if inspect.iscoroutinefunction(original):
        async def wrapped(*args, **kwargs):
            measurement = _active.get()
            if measurement is None:
                return await original(*args, **kwargs)
            with measurement.span(name):
                return await original(*args, **kwargs)
    else:
        def sync_wrapped(*args, **kwargs):
            measurement = _active.get()
            if measurement is None:
                return original(*args, **kwargs)
            with measurement.span(name):
                return original(*args, **kwargs)
        return sync_wrapped
    return wrapped


@contextmanager
def instrumentation():
    """Process-scoped patches: run only in a dedicated benchmark/test loop."""
    original_to_thread = asyncio.to_thread

    async def to_thread(func, /, *args, **kwargs):
        measurement = _active.get()
        if measurement is None:
            return await original_to_thread(func, *args, **kwargs)
        with measurement.span("executor.offload", target=getattr(func, "__name__", type(func).__name__)):
            submitted = time.perf_counter()
            parent = _parent.get()

            def execute():
                # Queue delay includes submission/dispatch overhead, not just pool contention.
                with measurement.span("executor.queue", submitted_parent=parent) as row:
                    row["start_ms"] = (submitted - measurement.start) * 1000
                    queue_ms = (time.perf_counter() - submitted) * 1000
                row["ms"] = queue_ms
                with measurement.span("executor.execution"):
                    return func(*args, **kwargs)
            return await original_to_thread(execute)

    with ExitStack() as stack:
        stack.enter_context(patch.object(asyncio, "to_thread", to_thread))
        for cls in (ApproximateTokenCounter, TiktokenTokenCounter):
            for method in ("count", "truncate"):
                stack.enter_context(patch.object(cls, method, timed(getattr(cls, method), f"tokens.{cls.__name__}.{method}")))
        for module in (search_drift, mixed_context):
            for name in ("resolve_chunk_citations", "resolve_reference_citations", "pack_items"):
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, timed(getattr(module, name), name)))
        stack.enter_context(patch.object(mixed_context.MixedContextBuilder, "build_context", timed(mixed_context.MixedContextBuilder.build_context, "context.mixed")))
        yield


class SlowStore(ExpansionStore, MixedFakeGraphStore):
    def __init__(self, db_latency: float):
        super().__init__()
        for name in STORE_METHODS:
            original = getattr(self, name)

            def call(*args, _name=name, _original=original, **kwargs):
                measurement = _active.get()
                with measurement.span(f"db.{_name}", args_key=digest([args, kwargs])):
                    time.sleep(db_latency)
                    return _original(*args, **kwargs)
            setattr(self, name, call)


class SlowLLM:
    def __init__(self, latency: float, repair: bool):
        self.latency = latency
        self.repair = repair

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        measurement = _active.get()
        phase = _phase.get()
        with measurement.span(f"llm.{phase}"):
            measurement.prompts.append({"phase": phase, "prompt": prompt})
            await asyncio.sleep(self.latency)
            if phase == "hyde":
                return LLMResponse(content="Alice leads Acme Corp.")
            if phase == "reduce":
                return LLMResponse(content="Alice directed Inception.")
            if phase.endswith("_repair"):
                raw = prompt.split("## Your previous response\n", 1)[1].split("\n\nReturn ONLY", 1)[0]
                data = json.loads(raw)
                data["score"] = 90
            else:
                # Prompt-keyed, not call-order-keyed: scheduling cannot change the tree.
                key = digest(prompt)[:12]
                data = {"answer": "Alice directed Inception.", "score": 90,
                        "follow_ups": [f"Question {key} {i}?" for i in range(2)]}
                if phase == "primer" and self.repair:
                    data["score"] = "invalid"
            return LLMResponse(content=json.dumps(data))


class SlowEmbedder:
    def __init__(self, latency: float):
        self.latency = latency

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        with _active.get().span("embed"):
            await asyncio.sleep(self.latency)
            return [0.1, 0.2, 0.3]


def make_retriever(db_latency, llm_latency, embed_latency, repair):
    retriever = DriftSearchRetriever(
        cast(Any, SlowStore(db_latency)), cast(Any, SlowLLM(llm_latency, repair)),
        cast(Any, SlowEmbedder(embed_latency)),
        config=DriftSearchConfig(max_rounds=2, actions_per_round=2, action_concurrency=3),
    )
    original = retriever._invoke_llm

    async def invoke(prompt, state, max_llm_calls, call_lock, phase):
        phase_token = _phase.set(phase)
        try:
            with _active.get().span(f"invoke.{phase}"):
                return await original(prompt, state, max_llm_calls, call_lock, phase)
        finally:
            _phase.reset(phase_token)
    retriever._invoke_llm = invoke
    for name in ("_build_action_mixed_context", "_format_primer_reports",
                 "_action_graph_evidence", "_reduction_graph_evidence", "_parse_json_strict"):
        setattr(retriever, name, timed(getattr(retriever, name), name))
    return retriever


async def measure(db_latency: float, *, llm_latency: float = LLM_LATENCY,
                  embed_latency: float = EMBED_LATENCY, repair: bool = False,
                  graph: bool = False, retriever=None) -> dict:
    """One search; caller supplies scoped instrumentation and bounded executor."""
    measurement = Measurement()
    token = _active.set(measurement)
    try:
        retriever = retriever or make_retriever(db_latency, llm_latency, embed_latency, repair)
        with measurement.span("search"):
            result = await retriever.search(QUERY, top_k=2, include_graph_evidence=graph,
                                            synthesize_citation_metadata=True)
        trace = result.metadata["drift_trace"]
        db = [s for s in measurement.spans if s["name"].startswith("db.")]
        calls = Counter(s["name"] for s in measurement.spans)
        span_ms = Counter()
        for span in measurement.spans:
            span_ms[span["name"]] += span["ms"]
        db_keys = Counter((s["name"], s["args_key"]) for s in db)
        semantic = {"query": result.query, "answer": result.answer, "context": result.context,
                    "citations": [asdict(c) for c in result.citations], "graph": result.graph,
                    "actions": trace["actions"],
                    "prompts": sorted(measurement.prompts, key=lambda p: (p["phase"], p["prompt"]))}
        return {"total_ms": measurement.spans[0]["ms"], "phase_ms": trace["phase_ms"],
                "actions": len(trace["actions"]), "llm_calls": trace["total_llm_calls"],
                "store_calls": len(db), "offloaded_calls": sum(s["thread"] != measurement.loop_thread for s in db),
                "repeated_db_calls": sum(n - 1 for n in db_keys.values()),
                "db_argument_groups": [{"method": name, "args_key": key, "count": n}
                                       for (name, key), n in sorted(db_keys.items())],
                "counts": dict(sorted(calls.items())),
                "inclusive_span_ms": dict(sorted(span_ms.items())), "failures": trace["failures"],
                "fingerprint": digest(semantic), "semantic": semantic, "spans": measurement.spans}
    finally:
        _active.reset(token)


async def run_benchmark(*, db_latency: float = .05, llm_latency: float = LLM_LATENCY,
                        embed_latency: float = EMBED_LATENCY, runs: int = 1,
                        concurrency: int = 1, workers: int = 4,
                        repair: bool = False, graph: bool = False, shared: bool = False) -> dict:
    if any(not math.isfinite(x) or x < 0 for x in (db_latency, llm_latency, embed_latency)):
        raise ValueError("latencies must be finite and non-negative")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (runs, concurrency, workers)):
        raise ValueError("runs, concurrency and workers must be positive integers")
    # Own this loop: setting a default executor is intentionally not a library API.
    executor = ThreadPoolExecutor(max_workers=workers)
    asyncio.get_running_loop().set_default_executor(executor)
    TiktokenTokenCounter().count("warm")
    rows = []
    gaps = []
    started = asyncio.Event()

    async def heartbeat():
        previous = time.perf_counter()
        started.set()
        while True:
            await asyncio.sleep(.005)
            now = time.perf_counter()
            gaps.append((now - previous) * 1000)
            previous = now

    heartbeat_task = asyncio.create_task(heartbeat())
    await started.wait()
    start = time.perf_counter()
    try:
        with instrumentation():
            try:
                retriever = make_retriever(db_latency, llm_latency, embed_latency, repair) if shared else None
                for _ in range(runs):
                    tasks = [asyncio.create_task(measure(db_latency, llm_latency=llm_latency,
                        embed_latency=embed_latency, repair=repair, graph=graph, retriever=retriever))
                        for _ in range(concurrency)]
                    try:
                        rows.extend(await asyncio.gather(*tasks))
                    finally:
                        for task in tasks:
                            if not task.done() and not task.cancelling():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # Cancelling an await does not stop an already-running store read.
                executor.shutdown(wait=True)
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
    return {"fake_only": True, "settings": {"db_latency": db_latency, "llm_latency": llm_latency,
            "embed_latency": embed_latency, "runs": runs, "concurrency": concurrency,
            "workers": workers, "repair": repair, "graph": graph, "shared": shared},
            "wall_ms": (time.perf_counter() - start) * 1000,
            "max_stall_ms": max(gaps, default=0), "searches": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("db-latency", .05), ("llm-latency", LLM_LATENCY), ("embed-latency", EMBED_LATENCY)):
        parser.add_argument(f"--{name}", type=float, default=default)
    for name, default in (("runs", 1), ("concurrency", 1), ("workers", 4)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name in ("repair", "graph", "shared", "json"):
        parser.add_argument(f"--{name}", action="store_true")
    args = vars(parser.parse_args())
    detailed = args.pop("json")
    report = asyncio.run(run_benchmark(**args))
    if not detailed:
        for row in report["searches"]:
            row.pop("spans")
            row.pop("semantic")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
