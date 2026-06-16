"""Neo4j graph building pipeline: ingestion + community detection.

Run this to populate the Neo4j graph and build communities.

Usage:
  python build_neo4j.py                              # Full pipeline
  python build_neo4j.py --indexes                    # Drop and recreate indexes only
  python build_neo4j.py --llm-provider openrouter    # Use OpenRouter for LLM
  python build_neo4j.py --embedder-provider openai   # Use OpenAI for embeddings
"""

import argparse
import asyncio
import os
import sys

from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, IndexManager

from config import EMBEDDING_DIM, get_embedder, get_llm, get_neo4j_store
from data import MOVIE_EXAMPLE_PAGES
from prompts import COMMUNITY_SUMMARY_PROMPT
from schema import COMMUNITY_RELATIONSHIP_TYPES, MOVIE_SCHEMA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the movie-industry graph in Neo4j."
    )
    parser.add_argument(
        "--indexes",
        action="store_true",
        help="Drop and recreate indexes only, then exit.",
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
    return parser.parse_args()


async def rebuild_indexes():
    store = get_neo4j_store()
    IndexManager(store, embedding_dim=EMBEDDING_DIM).create_indexes()
    print("Indexes recreated.")


async def build(llm_provider: str, embedder_provider: str):
    store = get_neo4j_store()
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)

    IndexManager(store, embedding_dim=EMBEDDING_DIM).create_indexes()

    pipeline = GraphBuilderPipeline(store, llm, embedder, schema=MOVIE_SCHEMA)
    result = await pipeline.build_from_pages(
        MOVIE_EXAMPLE_PAGES,
        metadata={"source": "example"},
        window_size=2,
        window_overlap=1,
    )
    print(f"Ingestion result: {result}")

    community = CommunityPipeline(
        store,
        llm,
        embedder,
        relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
        summary_prompt=COMMUNITY_SUMMARY_PROMPT,
    )
    comm_result = await community.build()
    print(f"Community result: {comm_result}")


if __name__ == "__main__":
    args = parse_args()
    if args.indexes:
        asyncio.run(rebuild_indexes())
    else:
        asyncio.run(build(args.llm_provider, args.embedder_provider))
