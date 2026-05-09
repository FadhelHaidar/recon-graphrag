"""Retrieval pipeline: query the graph with domain-specific prompts.

Run this after build_graph.py has populated the graph.
"""

import asyncio

from recon_graphrag import GraphRAG

from config import get_neo4j_store, get_llm, get_embedder
from prompts import (
    LOCAL_ANSWER_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
)


async def run_test_suite(test_queries: list):
    """Iterates through a list of questions and runs all RAG modes for each."""
    store = get_neo4j_store()
    llm = get_llm()
    embedder = get_embedder()

    graph_rag = GraphRAG(store, llm, embedder)
    
    # Configure prompts
    graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
    graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT
    graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
    graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT

    for item in test_queries:
        print(f"\n" + "="*60)
        print(f"TESTING: {item['search_type']}")
        print(f"QUERY: {item['query']}")
        print(f"OBJECTIVE: {item['test_objective']}")
        print("="*60)

        for mode in ["local", "global", "drift"]:
            try:
                result = await graph_rag.search(item['query'], mode=mode)
                print(f"\n>>> [{mode.upper()} ANSWER]:\n{result.answer}")
            except Exception as e:
                print(f"\n>>> [{mode.upper()} ERROR]: {str(e)}")

if __name__ == "__main__":
    # Your dictionary list of test cases
    test_suite = [
        {
            "search_type": "Local",
            "query": "Which movies in the database were directed by Christopher Nolan and feature Cillian Murphy?",
            "test_objective": "Verify 'DIRECTED' and 'ACTED_IN' relationship accuracy."
        },
        {
            "search_type": "Global",
            "query": "What are the most common themes across high-budget sci-fi films in this collection?",
            "test_objective": "Assess community summary quality for the 'Sci-Fi' cluster."
        },
        {
            "search_type": "Drift",
            "query": "How does the work of Hans Zimmer connect the movie Inception to the movie Dune?",
            "test_objective": "Test pathfinding between two different directors via a shared technical node."
        }
    ]

    asyncio.run(run_test_suite(test_suite))
