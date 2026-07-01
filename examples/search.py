"""Search the movie graph in Neo4j or Memgraph.

Run this after ingesting the shared movie graph artifact and building communities.

Usage:
  python search.py --backend neo4j
  python search.py --backend neo4j --limit 5
  python search.py --backend memgraph --modes local drift
"""

from __future__ import annotations

import argparse
import asyncio
import os

from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever

from common import SEARCH_OPTIONS, get_backend_targets
from config import get_embedder, get_llm
from prompts import (
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
    LOCAL_ANSWER_PROMPT,
)
from query_suite import MOVIE_QUERY_SUITE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval test queries against a movie graph backend."
    )
    parser.add_argument(
        "--backend",
        choices=["neo4j", "memgraph"],
        required=True,
        help="Graph backend to search.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
        help="LLM provider (defaults to LLM_PROVIDER env var, then openrouter).",
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["openrouter", "azure_openai", "openai", "sentence-transformer"],
        default=os.getenv("EMBEDDER_PROVIDER", "openrouter"),
        help="Embedder provider (defaults to EMBEDDER_PROVIDER env var, then openrouter).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["local", "global", "drift"],
        default=None,
        help="Restrict the suite to specific retrieval modes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of query cases to run.",
    )
    return parser.parse_args()


async def run_search_suite(
    backend: str,
    llm_provider: str,
    embedder_provider: str,
    modes: list[str] | None = None,
    limit: int | None = None,
):
    _, store, _ = get_backend_targets(backend)[0]
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)

    # Build the three retriever instances directly
    local = LocalSearchRetriever(store, llm, embedder, graph_name="entity-graph")
    local.answer_prompt = LOCAL_ANSWER_PROMPT

    global_search = GlobalSearchRetriever(store, llm, graph_name="entity-graph")
    global_search.map_prompt = GLOBAL_MAP_PROMPT
    global_search.reduce_prompt = GLOBAL_REDUCE_PROMPT

    drift = DriftSearchRetriever(store, llm, embedder, graph_name="entity-graph")
    drift.reduce_prompt = DRIFT_ANSWER_PROMPT

    search_instances = {
        "local": local,
        "global": global_search,
        "drift": drift,
    }

    suite = MOVIE_QUERY_SUITE
    if limit is not None:
        suite = suite[:limit]

    for item in suite:
        item_modes = item.get("modes", ["local", "global", "drift"])
        if modes:
            item_modes = [mode for mode in item_modes if mode in modes]

        print("\n" + "=" * 60)
        print(f"QUERY: {item['query']}")
        print(f"MODES: {', '.join(item_modes)}")
        print(f"OBJECTIVE: {item['test_objective']}")
        print("=" * 60)

        for mode in item_modes:
            search = search_instances[mode]
            try:
                result = await search.search(item["query"], **SEARCH_OPTIONS[mode])
                print(f"\n>>> [{mode.upper()} ANSWER]:\n{result.answer}")
            except Exception as exc:
                print(f"\n>>> [{mode.upper()} ERROR]: {exc}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_search_suite(
            args.backend,
            args.llm_provider,
            args.embedder_provider,
            modes=args.modes,
            limit=args.limit,
        )
    )
