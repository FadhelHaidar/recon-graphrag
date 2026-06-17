"""Search the movie graph in Neo4j or Memgraph.

Run this after ingesting the shared movie graph artifact and building communities.

Usage:
  python search.py --backend neo4j
  python search.py --backend memgraph --modes local drift
"""

from __future__ import annotations

import argparse
import asyncio
import os

from recon_graphrag import GraphRAG

try:
    from .common import configure_movie_rag, run_movie_search_suite
    from .config import get_embedder, get_llm, get_memgraph_store, get_neo4j_store
    from .query_suite import MOVIE_QUERY_SUITE
except ImportError:
    from common import configure_movie_rag, run_movie_search_suite
    from config import get_embedder, get_llm, get_memgraph_store, get_neo4j_store
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
    return parser.parse_args()


def _get_store(backend: str):
    if backend == "neo4j":
        return get_neo4j_store()
    if backend == "memgraph":
        return get_memgraph_store()
    raise ValueError(f"Unknown backend: {backend}")


async def run_search_suite(
    backend: str,
    llm_provider: str,
    embedder_provider: str,
    modes: list[str] | None = None,
):
    store = _get_store(backend)
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)
    graph_rag = configure_movie_rag(GraphRAG(store, llm, embedder))
    await run_movie_search_suite(graph_rag, MOVIE_QUERY_SUITE, modes_filter=modes)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_search_suite(
            args.backend,
            args.llm_provider,
            args.embedder_provider,
            modes=args.modes,
        )
    )
