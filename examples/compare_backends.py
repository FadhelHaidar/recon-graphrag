"""Compare Neo4j and Memgraph retrieval quality on the movie example.

Run this after both databases have been built from the same movie corpus.

Usage:
  python compare_backends.py --limit 5
  python compare_backends.py --modes local drift
  python compare_backends.py --dedup-strategy normalized
"""

from __future__ import annotations

import argparse
import asyncio
import os
import traceback

from recon_graphrag import GraphRAG

try:
    from .common import SEARCH_OPTIONS, configure_movie_rag, get_backend_targets
    from .config import get_embedder, get_llm
    from .query_suite import MOVIE_QUERY_SUITE
except ImportError:
    from common import SEARCH_OPTIONS, configure_movie_rag, get_backend_targets
    from config import get_embedder, get_llm
    from query_suite import MOVIE_QUERY_SUITE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Neo4j and Memgraph movie-example retrieval outputs."
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "azure_openai"),
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["openrouter", "azure_openai", "openai", "sentence-transformer"],
        default=os.getenv("EMBEDDER_PROVIDER", "azure_openai"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of query cases to run.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["local", "global", "drift"],
        default=None,
        help="Restrict comparison to specific retrieval modes.",
    )
    parser.add_argument(
        "--dedup-strategy",
        choices=["exact", "normalized", "fuzzy", "hybrid"],
        default="normalized",
        help="Dry-run dedup strategy to include in the report.",
    )
    parser.add_argument(
        "--tracebacks",
        action="store_true",
        help="Print full tracebacks for retrieval failures.",
    )
    return parser.parse_args()


async def _dedup_dry_run(store, strategy: str, embedder, llm) -> dict:
    try:
        return await store.resolve_entities(
            strategy=strategy,
            dry_run=True,
            embedder=embedder if strategy == "hybrid" else None,
            llm=llm if strategy == "hybrid" else None,
        )
    except Exception as exc:
        return {"error": str(exc)}


def _safe_stats(store) -> dict:
    try:
        stats = store.validate_graph_build()
    except Exception as exc:
        stats = {"error": str(exc)}

    try:
        community_stats = store.get_community_stats("entity-graph")
        stats["community_count"] = len(community_stats)
        stats["community_levels"] = sorted({row["level"] for row in community_stats})
        stats["empty_communities"] = sum(
            1 for row in community_stats if row.get("entity_count", 0) == 0
        )
    except Exception as exc:
        stats["community_error"] = str(exc)

    return stats


def _context_headings(context: str) -> list[str]:
    headings = []
    for line in context.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Finding:", "Report Segment", "Segment ")):
            headings.append(stripped)
    return headings[:8]


async def _run_one(
    graph_rag: GraphRAG,
    query: str,
    mode: str,
    tracebacks: bool = False,
) -> dict:
    try:
        result = await graph_rag.search(query, mode=mode, **SEARCH_OPTIONS[mode])
        return {
            "ok": True,
            "answer": result.answer,
            "context": result.context,
            "headings": _context_headings(result.context),
        }
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        if tracebacks:
            error = f"{error}\n{traceback.format_exc().rstrip()}"
        return {
            "ok": False,
            "error": error,
            "answer": "",
            "context": "",
            "headings": [],
        }


def _print_stats(label: str, stats: dict, dedup: dict):
    print(f"\n[{label}] graph stats")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"[{label}] dedup dry-run")
    for key in ("strategy", "candidate_groups", "merged_groups", "merged_nodes", "error"):
        if key in dedup:
            print(f"  {key}: {dedup[key]}")


def _print_mode_diff(query: str, mode: str, neo4j: dict, memgraph: dict):
    print(f"\n--- {mode.upper()} :: {query}")
    if not neo4j["ok"] or not memgraph["ok"]:
        print(f"  neo4j_error: {neo4j.get('error', '')}")
        print(f"  memgraph_error: {memgraph.get('error', '')}")
        return

    neo4j_headings = set(neo4j["headings"])
    memgraph_headings = set(memgraph["headings"])
    missing = sorted(neo4j_headings - memgraph_headings)
    extra = sorted(memgraph_headings - neo4j_headings)
    print(f"  neo4j_answer_chars: {len(neo4j['answer'])}")
    print(f"  memgraph_answer_chars: {len(memgraph['answer'])}")
    print(f"  neo4j_context_items: {len(neo4j['headings'])}")
    print(f"  memgraph_context_items: {len(memgraph['headings'])}")
    if not memgraph["context"].strip():
        print("  memgraph_issue: empty context")
    if missing:
        print(f"  missing_from_memgraph: {missing[:5]}")
    if extra:
        print(f"  memgraph_extra: {extra[:5]}")


async def main():
    args = parse_args()
    llm = get_llm(args.llm_provider)
    embedder = get_embedder(args.embedder_provider)

    targets = {name: store for name, store, _ in get_backend_targets("all")}
    neo4j_store = targets["neo4j"]
    memgraph_store = targets["memgraph"]
    neo4j_rag = configure_movie_rag(GraphRAG(neo4j_store, llm, embedder))
    memgraph_rag = configure_movie_rag(GraphRAG(memgraph_store, llm, embedder))

    neo4j_stats = _safe_stats(neo4j_store)
    memgraph_stats = _safe_stats(memgraph_store)
    neo4j_dedup = await _dedup_dry_run(neo4j_store, args.dedup_strategy, embedder, llm)
    memgraph_dedup = await _dedup_dry_run(memgraph_store, args.dedup_strategy, embedder, llm)
    _print_stats("neo4j", neo4j_stats, neo4j_dedup)
    _print_stats("memgraph", memgraph_stats, memgraph_dedup)

    suite = MOVIE_QUERY_SUITE
    if args.limit is not None:
        suite = suite[: args.limit]

    for item in suite:
        query = item["query"]
        item_modes = item.get("modes", ["local", "global", "drift"])
        modes = [mode for mode in item_modes if args.modes is None or mode in args.modes]
        for mode in modes:
            neo4j_result, memgraph_result = await asyncio.gather(
                _run_one(neo4j_rag, query, mode, tracebacks=args.tracebacks),
                _run_one(memgraph_rag, query, mode, tracebacks=args.tracebacks),
            )
            _print_mode_diff(query, mode, neo4j_result, memgraph_result)


if __name__ == "__main__":
    asyncio.run(main())
