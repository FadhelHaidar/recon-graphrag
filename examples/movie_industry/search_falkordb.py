"""FalkorDB retrieval pipeline: query the graph with domain-specific prompts.

Run this after build_falkordb.py has populated the graph.

Usage:
  python search_falkordb.py
  python search_falkordb.py --llm-provider openrouter
  python search_falkordb.py --embedder-provider openai
"""

import argparse
import asyncio
import os

from recon_graphrag import GraphRAG

from config import get_embedder, get_falkordb_store, get_llm
from prompts import (
    LOCAL_ANSWER_PROMPT,
    DRIFT_ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval test queries against the FalkorDB movie graph."
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


async def run_test_suite(test_queries: list, llm_provider: str, embedder_provider: str):
    """Iterates through test questions and runs the specified RAG mode(s) for each."""
    store = get_falkordb_store()
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)

    graph_rag = GraphRAG(store, llm, embedder)

    # Configure prompts
    graph_rag.local.answer_prompt = LOCAL_ANSWER_PROMPT
    graph_rag.drift.answer_prompt = DRIFT_ANSWER_PROMPT
    graph_rag.global_.map_prompt = GLOBAL_MAP_PROMPT
    graph_rag.global_.reduce_prompt = GLOBAL_REDUCE_PROMPT

    search_options = {
        "local": {},
        "global": {
            "top_k": 5,
            # Community levels: "finest" == level 0, "coarsest" == highest level,
            # "all" == no level filter, or pass an integer for exact old behavior.
            "community_level": "coarsest",
        },
        "drift": {
            "top_k": 10,
            "community_top_k": 3,
            # DRIFT defaults to the finest level for backward compatibility.
            "community_level": "finest",
        },
    }

    for item in test_queries:
        modes = item.get("modes", ["local", "global", "drift"])

        print(f"\n" + "="*60)
        print(f"QUERY: {item['query']}")
        print(f"MODES: {', '.join(modes)}")
        print(f"OBJECTIVE: {item['test_objective']}")
        print("="*60)

        for mode in modes:
            if mode not in search_options:
                print(f"\n>>> [{mode.upper()} ERROR]: Unknown mode. Use local, global, or drift.")
                continue
            try:
                result = await graph_rag.search(
                    item["query"],
                    mode=mode,
                    **search_options[mode],
                )
                print(f"\n>>> [{mode.upper()} ANSWER]:\n{result.answer}")
            except Exception as e:
                print(f"\n>>> [{mode.upper()} ERROR]: {str(e)}")

if __name__ == "__main__":
    args = parse_args()

    # Graph RAG test cases for the movie industry corpus.
    # Each case can specify a "modes" list; defaults to ["local", "global", "drift"].
    test_suite = [
        {
            "query": (
                "Which movies were directed by Christopher Nolan and feature "
                "Cillian Murphy?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify DIRECTED and ACTED_IN relationships between Person and Movie."
            ),
        },
        {
            "query": (
                "Which films did Hans Zimmer compose music for in this collection?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify COMPOSED_MUSIC relationships and composer-to-movie extraction."
            ),
        },
        {
            "query": (
                "Which movies were shot by Hoyte van Hoytema?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify SHOT_BY relationships for cinematographer-to-movie links."
            ),
        },
        {
            "query": (
                "How does Hans Zimmer connect Inception to Dune?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test multi-hop traversal from Movie to Person to Movie using "
                "COMPOSED_MUSIC relationships."
            ),
        },
        {
            "query": (
                "Find the connection path between Interstellar and Dune."
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test whether the graph can connect movies through shared actors, "
                "composer, genre, and sci-fi themes."
            ),
        },
        {
            "query": (
                "How is Cillian Murphy connected to Michelle Yeoh?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test cross-film pathfinding through Sunshine, Oppenheimer, "
                "Everything Everywhere All At Once, and actor bridges."
            ),
        },
        {
            "query": (
                "How are Inception, The Revenant, and Birdman connected?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test multi-hop traversal through Leonardo DiCaprio, Tom Hardy, "
                "Alejandro G. Inarritu, and award relationships."
            ),
        },
        {
            "query": (
                "Which characters were played by actors in Christopher Nolan films?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify PLAYED_CHARACTER extraction and connections between Person, "
                "Character, and Movie context."
            ),
        },
        {
            "query": (
                "How does Tom Hardy connect Nolan films to non-Nolan films?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test actor bridge reasoning across Inception, The Dark Knight Rises, "
                "The Revenant, and Peaky Blinders."
            ),
        },
        {
            "query": (
                "Which actors create bridges between Nolan films and other directors' films?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Evaluate whether local and DRIFT retrieval can identify bridge actors "
                "such as Tom Hardy, Leonardo DiCaprio, Cillian Murphy, Anne Hathaway, "
                "and Timothee Chalamet."
            ),
        },
        {
            "query": (
                "Which movies in this corpus won the Oscar for Best Picture?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify WON_AWARD relationship extraction for Movie to Award."
            ),
        },
        {
            "query": (
                "How are Parasite, Everything Everywhere All At Once, Birdman, "
                "and Oppenheimer connected through awards?"
            ),
            "modes": ["global", "drift"],
            "test_objective": (
                "Test global community retrieval over shared award relationships."
            ),
        },
        {
            "query": (
                "How does Parasite connect to 1917?"
            ),
            "modes": ["global", "drift"],
            "test_objective": (
                "Verify award competition reasoning using WON_AWARD and NOMINATED_FOR "
                "relationships."
            ),
        },
        {
            "query": (
                "Which movies explore time, memory, or moral responsibility?"
            ),
            "modes": ["local", "global"],
            "test_objective": (
                "Test EXPLORES relationships and semantic retrieval over Theme nodes."
            ),
        },
        {
            "query": (
                "What themes connect Interstellar, Inception, Dunkirk, and Oppenheimer?"
            ),
            "modes": ["global", "drift"],
            "test_objective": (
                "Assess global and DRIFT retrieval over Nolan-related thematic clusters."
            ),
        },
        {
            "query": (
                "Which films in the corpus explore class inequality or social conflict?"
            ),
            "modes": ["local", "global"],
            "test_objective": (
                "Verify Theme extraction for Parasite, Oppenheimer, and other "
                "social-conflict-related films."
            ),
        },
        {
            "query": (
                "Which films are connected through IMAX cinematography?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify USES_TECHNIQUE relationships and retrieval over Technique nodes."
            ),
        },
        {
            "query": (
                "How are Nolan and Villeneuve connected through large-format filmmaking?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test semantic traversal across directors, cinematographers, movies, "
                "and Technique nodes."
            ),
        },
        {
            "query": (
                "Which films use Shepard tones or sound design to create tension?"
            ),
            "modes": ["local"],
            "test_objective": (
                "Verify extraction of musical and sound-related Technique nodes."
            ),
        },
        {
            "query": (
                "Which movies are connected to Warner Bros, A24, or Universal Pictures?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Verify PRODUCED and DISTRIBUTED relationships involving Studio nodes."
            ),
        },
        {
            "query": (
                "How did Christopher Nolan's studio relationships change from "
                "Warner Bros to Universal Pictures?"
            ),
            "modes": ["local", "drift"],
            "test_objective": (
                "Test studio-level reasoning over Nolan movies and Studio relationships."
            ),
        },
        {
            "query": (
                "Which movies belong to or are connected with major franchises or series?"
            ),
            "modes": ["local", "global"],
            "test_objective": (
                "Verify BELONGS_TO relationships for The Dark Knight trilogy, "
                "Dune series, and Marvel Cinematic Universe references."
            ),
        },
        {
            "query": (
                "What are the major communities or clusters in this movie graph?"
            ),
            "modes": ["global"],
            "test_objective": (
                "Assess global search quality using community summaries."
            ),
        },
        {
            "query": (
                "Summarize the Nolan-related community in this graph."
            ),
            "modes": ["global"],
            "test_objective": (
                "Test whether global retrieval can summarize the Nolan subgraph involving "
                "movies, actors, composer, cinematographer, studio, and themes."
            ),
        },
        {
            "query": (
                "Summarize the Oscar-winning film community in this graph."
            ),
            "modes": ["global"],
            "test_objective": (
                "Test community detection around Best Picture winners and award links."
            ),
        },
        {
            "query": (
                "Compare the connections of Parasite and Oppenheimer in the graph."
            ),
            "modes": ["local", "global", "drift"],
            "test_objective": (
                "Test comparative reasoning across themes, awards, directors, and "
                "historical or social context."
            ),
        },
        {
            "query": (
                "Compare the sci-fi network around Interstellar, Dune, Sunshine, "
                "and Everything Everywhere All At Once."
            ),
            "modes": ["global", "drift"],
            "test_objective": (
                "Test cross-community retrieval across sci-fi movies, actors, studios, "
                "themes, and composers."
            ),
        },
        {
            "query": (
                "Which people are the most important bridge nodes in this graph?"
            ),
            "modes": ["local", "global", "drift"],
            "test_objective": (
                "Evaluate whether the system can identify central connecting people "
                "such as Christopher Nolan, Hans Zimmer, Cillian Murphy, Tom Hardy, "
                "Leonardo DiCaprio, and Michelle Yeoh."
            ),
        },
    ]

    asyncio.run(run_test_suite(test_suite, args.llm_provider, args.embedder_provider))
