"""Movie Industry app using recon-graphrag SDK.

Demonstrates how to wire the SDK with domain-specific schema and prompts.
"""

import asyncio
import os

from neo4j import GraphDatabase

# --- SDK imports ---
from recon_graphrag import (
    Neo4jGraphStore,
    GraphBuilderPipeline,
    IndexManager,
    CommunityPipeline,
    GraphRAG,
    create_llm,
    create_embedder,
)

# --- Domain imports ---
from schema import MOVIE_SCHEMA, COMMUNITY_RELATIONSHIP_TYPES
from prompts import (
    COMMUNITY_SUMMARY_PROMPT,
    LOCAL_ANSWER_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
)


def main():
    # --- Config (from env vars or .env) ---
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    # --- Create graph store ---
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    store = Neo4jGraphStore(driver, database=neo4j_database)

    # --- Create LLM + Embedder ---
    # Azure OpenAI:
    llm = create_llm(
        "azure_openai",
        model_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model_params={"temperature": 0},
    )
    embedder = create_embedder(
        "azure_openai",
        model=os.getenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME", "text-embedding-3-small"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # --- Create indexes (run once) ---
    index_mgr = IndexManager(store)
    index_mgr.create_indexes()

    # --- Ingestion pipeline ---
    pipeline = GraphBuilderPipeline(store, llm, embedder, schema=MOVIE_SCHEMA)

    # --- Community pipeline ---
    community = CommunityPipeline(
        store, llm, embedder,
        relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
        summary_prompt=COMMUNITY_SUMMARY_PROMPT,
    )

    # --- Search (with domain prompts injected) ---
    graph_rag = GraphRAG(store, llm, embedder)
    # Override default prompts with domain-specific ones:
    graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
    graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT
    graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
    graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT

    # --- Example usage ---
    async def run():
        # Ingest
        result = await pipeline.build_from_text(
            "Christopher Nolan directed Inception (2010), a sci-fi thriller exploring "
            "dream manipulation, starring Leonardo DiCaprio. Warner Bros produced the film, "
            "which won an Oscar for Best Cinematography. Parasite (2019), directed by Bong Joon-ho, "
            "explores social inequality and won the Palme d'Or and Oscar for Best Picture.",
            metadata={"source": "example"},
        )
        print(f"Ingestion result: {result}")

        # Build communities (normally run on schedule)
        comm_result = await community.build()
        print(f"Community result: {comm_result}")

        # Search
        for mode in ["local", "global", "drift"]:
            result = await graph_rag.search(
                "What themes does Inception explore?", mode=mode
            )
            print(f"\n[{mode.upper()}] {result.answer}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
