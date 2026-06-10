"""Graph building pipeline: ingestion + community detection.

Run this to populate the graph and build communities.

Usage:
  python build_graph.py              # Full pipeline
  python build_graph.py --indexes    # Drop and recreate indexes only
"""

import asyncio
import sys

from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, IndexManager

from config import get_embedder, get_llm, get_neo4j_store
from data import MOVIE_EXAMPLE_TEXT
from prompts import COMMUNITY_SUMMARY_PROMPT
from schema import COMMUNITY_RELATIONSHIP_TYPES, MOVIE_SCHEMA


async def rebuild_indexes():
    store = get_neo4j_store()
    IndexManager(store, embedding_dim=1536).create_indexes()
    print("Indexes recreated.")


async def build():
    store = get_neo4j_store()
    llm = get_llm()
    embedder = get_embedder()

    IndexManager(store, embedding_dim=1536).create_indexes()

    pipeline = GraphBuilderPipeline(store, llm, embedder, schema=MOVIE_SCHEMA)
    result = await pipeline.build_from_text(
        MOVIE_EXAMPLE_TEXT,
        metadata={"source": "example"},
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
    if "--indexes" in sys.argv:
        asyncio.run(rebuild_indexes())
    else:
        asyncio.run(build())
