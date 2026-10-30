"""Manual DRIFT timing harness with simulated database latency.

Not collected by pytest (no ``test_`` prefix) and imposes no thresholds. It
reports, for one fixed DRIFT workload, the wall time and the longest event-loop
scheduling gap observed during the search. Store calls are labelled with the
thread they ran on, so a gap only points at blocking reads when they share the
loop thread. Numbers describe the mechanical cost of blocking reads; they are
not production latencies.

Run from the repository root:

    uv run python tests/manual/benchmark_drift.py
    uv run python tests/manual/benchmark_drift.py --db-latency 0.01 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.drift_types import DriftSearchConfig
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.utils.tokens import TiktokenTokenCounter
from tests.retrieval.test_drift import MixedFakeGraphStore

LLM_LATENCY = 0.30
EMBED_LATENCY = 0.10


class SlowStore(MixedFakeGraphStore):
    """Every synchronous store call costs ``db_latency`` on the calling thread."""

    def __init__(self, db_latency: float):
        super().__init__()
        self.db_latency = db_latency
        self.loop_thread = threading.get_ident()
        self.call_threads: list[tuple[str, int]] = []

    def _block(self, label: str) -> None:
        self.call_threads.append((label, threading.get_ident()))
        time.sleep(self.db_latency)

    def vector_search(self, *args: Any, **kwargs: Any):
        self._block("vector_search")
        return super().vector_search(*args, **kwargs)

    def keyword_search(self, *args: Any, **kwargs: Any):
        self._block("keyword_search")
        return super().keyword_search(*args, **kwargs)

    def fetch_entity_context(self, *args: Any, **kwargs: Any):
        self._block("fetch_entity_context")
        return super().fetch_entity_context(*args, **kwargs)

    def vector_search_community_reports(self, *args: Any, **kwargs: Any):
        self._block("vector_search_community_reports")
        return super().vector_search_community_reports(*args, **kwargs)

    def execute_query(self, *args: Any, **kwargs: Any):
        self._block("execute_query")
        return super().execute_query(*args, **kwargs)

    def get_claims_for_entities(self, *args: Any, **kwargs: Any):
        self._block("get_claims_for_entities")
        return super().get_claims_for_entities(*args, **kwargs)

    def resolve_chunk_citations(self, *args: Any, **kwargs: Any):
        self._block("resolve_chunk_citations")
        return super().resolve_chunk_citations(*args, **kwargs)


class SlowLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(LLM_LATENCY)
        if "Partial Answers" in prompt or "synthesizing" in prompt.lower():
            return LLMResponse(content="Alice directed Inception.")
        self.calls_for_follow_up = getattr(self, "calls_for_follow_up", 0) + 1
        base = self.calls_for_follow_up * 10
        return LLMResponse(
            content=json.dumps(
                {
                    "answer": "Alice directed Inception.",
                    "score": 90,
                    "follow_ups": [f"Question {base}?", f"Question {base + 1}?"],
                }
            )
        )


class SlowEmbedder:
    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        await asyncio.sleep(EMBED_LATENCY)
        return [0.1, 0.2, 0.3]


async def measure(db_latency: float) -> dict:
    store = SlowStore(db_latency)
    retriever = DriftSearchRetriever(
        cast(Any, store),
        cast(Any, SlowLLM()),
        cast(Any, SlowEmbedder()),
        config=DriftSearchConfig(max_depth=2, max_followups=2, action_concurrency=3),
    )

    # The tokenizer loads its BPE vocabulary once (~200ms); keep that one-time
    # cost out of the measured stall, it is not a store read.
    TiktokenTokenCounter().count("warm")
    gaps: list[float] = []
    started = asyncio.Event()

    async def heartbeat() -> None:
        previous = time.monotonic()
        started.set()
        while True:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    task = asyncio.create_task(heartbeat())
    await started.wait()
    start = time.monotonic()
    try:
        result = await retriever.search("How does Alice connect to Acme?", top_k=2)
    finally:
        elapsed = time.monotonic() - start
        task.cancel()

    return {
        "total_ms": round(elapsed * 1000),
        "max_stall_ms": round(max(gaps) * 1000),
        "actions": len(result.metadata["drift_trace"]["actions"]),
        "store_calls": len(store.call_threads),
        "offloaded_calls": sum(
            1 for _, thread in store.call_threads if thread != store.loop_thread
        ),
        "llm_calls": result.metadata["drift_trace"]["total_llm_calls"],
        "answer": result.answer[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-latency", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    row = asyncio.run(measure(args.db_latency))
    if args.json:
        print(json.dumps({"db_latency": args.db_latency, **row}, indent=2))
        return
    print(f"db_latency={args.db_latency}s")
    for key, value in row.items():
        print(f"  {key:16} {value}")


if __name__ == "__main__":
    main()
