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

try:
    from .common import (
        configure_movie_rag,
        get_backend_targets,
        run_movie_search_suite,
    )
    from .config import get_embedder, get_llm
    from .query_suite import MOVIE_QUERY_SUITE
except ImportError:
    from common import (
        configure_movie_rag,
        get_backend_targets,
        run_movie_search_suite,
    )
    from config import get_embedder, get_llm
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
        default=os.getenv("LLM_PROVIDER", "azure_openai"),
        help="LLM provider (defaults to LLM_PROVIDER env var, then azure_openai).",
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["openrouter", "azure_openai", "openai", "sentence-transformer"],
        default=os.getenv("EMBEDDER_PROVIDER", "azure_openai"),
        help="Embedder provider (defaults to EMBEDDER_PROVIDER env var, then azure_openai).",
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
    search_instances = configure_movie_rag(store, llm, embedder)
    suite = MOVIE_QUERY_SUITE
    if limit is not None:
        suite = suite[:limit]
    await run_movie_search_suite(search_instances, suite, modes_filter=modes)


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
