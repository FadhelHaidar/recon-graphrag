"""Manual F9 measurement harness: bounded work, not just bounded concurrency.

Not collected by pytest (no ``test_`` prefix) and imposes **no** thresholds.
It prints one table of deterministic measurements per scenario at two or more
input sizes, so before/after numbers are described rather than assumed.

Run from the repository root:

    uv run python tests/manual/benchmark_f9.py
    uv run python tests/manual/benchmark_f9.py --only global,fuzzy
    uv run python tests/manual/benchmark_f9.py --json

Measurements are produced with instrumented fakes and ``tracemalloc``. They say
nothing about production capacity, provider cost, or latency: live-backend
numbers require the disposable Podman/Compose setup documented in
``docs/10-testing.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tracemalloc
from pathlib import Path
from typing import Any, cast

# Run as a file, so the repository root is not on sys.path automatically.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.graphdb.entity_resolution import (
    _EntityRecord,
    _FuzzyCandidateIndex,
    _normalize_name,
    _scan_candidates,
)
from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.utils.tokens import ApproximateTokenCounter
from tests.integration.synthetic_e2e_support import SYNTHETIC_PAGES, SYNTHETIC_SCHEMA

NAP = 0.002
SIZES = (8, 32)


class CountingLLM:
    """LLM fake that records calls, peak concurrency, and peak live tasks."""

    def __init__(self, content: str = '{"nodes": [], "relationships": []}'):
        self.content = content
        self.calls = 0
        self.map_calls = 0
        self.map_input_tokens = 0
        self.reduce_calls = 0
        self.active = 0
        self.peak_active = 0
        self.peak_tasks = 0

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if "Synthesize" in prompt or "Partial Answers" in prompt:
            self.reduce_calls += 1
        else:
            self.map_calls += 1
            self.map_input_tokens += ApproximateTokenCounter().count(prompt)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.peak_tasks = max(self.peak_tasks, len(asyncio.all_tasks()))
        try:
            await asyncio.sleep(NAP)
            return LLMResponse(content=self.content)
        finally:
            self.active -= 1

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError("async only")


class NullEmbedder:
    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:  # pragma: no cover
        return [0.0, 0.0, 0.0]


class NullWriter:
    def __init__(self):
        self.writes = 0

    def write_graph_document(self, *args: Any, **kwargs: Any) -> dict:
        self.writes += 1
        return {}

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


class NullStore:
    """Graph store fake: no rows, every write is a no-op."""

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        return []

    def validate_graph_build(self) -> dict:
        return {"entity_count": 0, "entity_relationship_count": 0}

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


def _measure(factory) -> tuple[Any, Any, int]:
    """Run a coroutine factory twice; report the second run's peak traced bytes.

    The first run warms one-time caches (imports, tqdm, token tables) that would
    otherwise dominate the peak of the smallest input size.
    """
    first = asyncio.run(factory())
    tracemalloc.start()
    try:
        result = asyncio.run(factory())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return first, result, peak


# --- Scenario 1+2: ingestion task materialization ---------------------------


def _documents(count: int) -> list[dict]:
    documents = []
    for index in range(count):
        page = SYNTHETIC_PAGES[index % len(SYNTHETIC_PAGES)]
        documents.append(
            {
                "text": page["text"],
                "metadata": {"record_id": f"synthetic-page-{index:03d}"},
            }
        )
    return documents


def bench_ingestion(documents: int, concurrency: int) -> dict:
    async def run() -> dict:
        llm = CountingLLM()
        writer = NullWriter()
        pipeline = GraphBuilderPipeline(
            graph_store=cast(Any, NullStore()),
            llm=llm,
            embedder=cast(Any, NullEmbedder()),
            graph_writer=cast(Any, writer),
            perform_entity_resolution=False,
            embed_entities=False,
        )
        results = await pipeline.build_from_documents(
            _documents(documents),
            schema=SYNTHETIC_SCHEMA,
            document_concurrency=concurrency,
            extraction_concurrency=1,
            max_gleanings=0,
            chunk_unit="characters",
            chunk_size=400,
            chunk_overlap=0,
        )
        return {
            "input_documents": documents,
            "document_concurrency": concurrency,
            "llm_calls": llm.calls,
            "peak_concurrent_llm": llm.peak_active,
            "peak_live_tasks": llm.peak_tasks,
            "returned_results": len(results),
            "graph_writes": writer.writes,
        }

    _, outcome, peak = _measure(run)
    outcome["peak_traced_kib"] = round(peak / 1024)
    return outcome


# --- Scenario 3: fuzzy candidate discovery ----------------------------------


def _pool(size: int, adversarial: bool = False) -> list[_EntityRecord]:
    """Singleton pool for candidate-discovery measurement.

    ``adversarial`` keeps every value on one leading character and one length,
    which is the case the index cannot improve on.
    """
    records = []
    for index in range(size):
        if adversarial:
            value = f"entity{index:04d}"
        else:
            value = f"{chr(65 + index % 26)}{index:04d}{'x' * (index % 9)}"
        records.append(
            _EntityRecord(
                node_id=index,
                entity_id=f"e{index}",
                graph_name="bench",
                domain_label="Person",
                resolve_value=value,
                normalized_value=_normalize_name(value),
            )
        )
    return records


def bench_fuzzy(
    pool_size: int, max_candidates: int, adversarial: bool = False
) -> dict:
    pool = _pool(pool_size, adversarial=adversarial)

    def never_match(_left, _right) -> bool:
        return False

    index = _FuzzyCandidateIndex(pool)
    index_inspections = 0
    for anchor in pool:
        index.candidates(anchor, max_candidates, set(), never_match)
        index_inspections = index.inspections

    scan_inspections = 0
    for anchor in pool:
        _scan_candidates(anchor, pool, max_candidates, set(), never_match)
        scan_inspections += pool_size

    return {
        "pool_size": pool_size,
        "pool_shape": "one-leading-char" if adversarial else "varied-leading-char",
        "max_candidates_per_entity": max_candidates,
        "full_scan_inspections": scan_inspections,
        "indexed_inspections": index_inspections,
        "inspection_ratio": (
            round(scan_inspections / index_inspections, 1) if index_inspections else None
        ),
    }


# --- Scenario 4: global search whole-query work -----------------------------


def bench_global(report_count: int, **limits: int | None) -> dict:
    reports = [
        {
            "id": f"report:{index}:0",
            "level": 0,
            "report_text": f"Community {index} summary. " + "detail " * 30,
        }
        for index in range(report_count)
    ]

    class Store(NullStore):
        def execute_query(self, query, params=None):
            params = params or {}
            if "MATCH (c:Community" in query:
                return reports[: params["report_limit"]]
            return []

    llm = CountingLLM(
        content=json.dumps({"answer": "a", "helpfulness": 50, "report_ids": []})
    )
    search = GlobalSearchRetriever(
        cast(Any, Store()),
        llm,
        map_budget_tokens=600,
        map_concurrency=limits.get("map_concurrency") or 5,
        max_map_calls=limits.get("max_map_calls"),
        max_reports=limits.get("max_reports"),
        max_map_tokens=limits.get("max_map_tokens"),
        track_token_usage=False,
    )

    result = asyncio.run(search.search("benchmark question", community_level=0, random_seed=42))
    metadata = result.metadata or {}
    if search.max_map_tokens is not None:
        assert llm.map_input_tokens <= search.max_map_tokens
    if search.max_map_calls is not None:
        assert llm.map_calls <= search.max_map_calls
    return {
        "reports_available": report_count,
        "max_reports": limits.get("max_reports"),
        "max_map_calls": limits.get("max_map_calls"),
        "max_map_tokens": limits.get("max_map_tokens"),
        "map_concurrency": search.map_concurrency,
        "reports_used": metadata.get("reports_used"),
        "map_batches": metadata.get("map_batches"),
        "map_calls": llm.map_calls,
        "map_input_tokens_estimated": llm.map_input_tokens,
        "reduce_calls": llm.reduce_calls,
        "peak_concurrent_llm": llm.peak_active,
        "peak_live_tasks": llm.peak_tasks,
    }


# --- Scenario 5: community reporting ---------------------------------------


def bench_communities(count: int, concurrency: int) -> dict:
    class Store(NullStore):
        def get_communities(self, graph_name, level=None):
            return [
                {"id": f"community:{index}", "level": 0, "entity_count": 2}
                for index in range(count)
            ]

        def get_community_ranked_context(self, graph_name, community_id, level=0):
            return [
                {
                    "e_id": f"person:{community_id}",
                    "e_name": community_id,
                    "e_description": "Bench entity.",
                    "e_labels": ["Person"],
                    "e_degree": 1,
                    "rel_type": "KNOWS",
                    "rel_description": "Bench relationship.",
                    "observation_count": 1,
                    "combined_degree": 2,
                    "other_id": "org:bench",
                    "other_name": "Bench",
                    "other_description": "Bench org.",
                    "other_labels": ["Organization"],
                    "other_degree": 1,
                }
            ]

        def get_claims_for_entities(self, graph_name, entity_ids):
            return []

        def get_community_reports_by_keys(self, graph_name, keys, top_k):
            return []

    llm = CountingLLM(
        content=json.dumps(
            {
                "title": "Bench",
                "summary": "Bench summary.",
                "rating": 5.0,
                "rating_explanation": "ok",
                "findings": [
                    {
                        "description": "Bench finding.",
                        "references": [
                            {"target_id": "person:bench", "target_type": "entity"}
                        ],
                    }
                ],
            }
        )
    )
    summarizer = CommunitySummarizer(
        cast(Any, Store()), llm=cast(Any, llm), concurrency=concurrency
    )
    results, stats = asyncio.run(summarizer.generate_all(level=0))
    return {
        "communities": count,
        "concurrency": concurrency,
        "reports_returned": len(results),
        "stats_succeeded": stats.succeeded,
        "llm_calls": llm.calls,
        "peak_concurrent_llm": llm.peak_active,
        "peak_live_tasks": llm.peak_tasks,
    }


SCENARIOS = {
    "ingestion": lambda: [
        bench_ingestion(documents, concurrency=2) for documents in SIZES
    ],
    "fuzzy": lambda: (
        [bench_fuzzy(pool_size, max_candidates=10) for pool_size in (200, 800)]
        + [bench_fuzzy(800, max_candidates=10, adversarial=True)]
    ),
    "global": lambda: (
        [bench_global(reports) for reports in (16, 64)]
        + [bench_global(64, max_reports=8), bench_global(64, max_map_calls=2)]
        + [bench_global(64, max_map_tokens=400), bench_global(64, max_map_tokens=1200)]
    ),
    "communities": lambda: [
        bench_communities(count, concurrency=2) for count in SIZES
    ],
}


def _print_table(name: str, rows: list[dict]) -> None:
    print(f"\n### {name}")
    if not rows:
        print("(no rows)")
        return
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), *(len(str(row.get(column))) for row in rows))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=",".join(SCENARIOS),
        help=f"comma-separated subset of {','.join(SCENARIOS)}",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    output = {name: SCENARIOS[name]() for name in selected}
    if args.json:
        print(json.dumps(output, indent=2))
        return 0
    for name, rows in output.items():
        _print_table(name, rows)
    print(
        "\nno thresholds: these are deterministic fake-LLM/fake-store measurements, "
        "not production capacity, latency, or cost claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
