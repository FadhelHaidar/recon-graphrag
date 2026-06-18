"""Run the movie community pipeline for Neo4j or Memgraph.

This assumes the shared movie graph artifact has already been ingested.

Usage:
  python communities.py --backend neo4j
  python communities.py --backend memgraph --community-gamma 3.0
  python communities.py --backend all
"""

from __future__ import annotations

import argparse
import asyncio
import os

from recon_graphrag import CommunityPipeline

try:
    from .common import BACKEND_CHOICES, get_backend_targets
    from .config import (
        EMBEDDING_DIM,
        get_embedder,
        get_llm,
    )
    from .prompts import COMMUNITY_SUMMARY_PROMPT
    from .schema import COMMUNITY_RELATIONSHIP_TYPES
except ImportError:
    from common import BACKEND_CHOICES, get_backend_targets
    from config import (
        EMBEDDING_DIM,
        get_embedder,
        get_llm,
    )
    from prompts import COMMUNITY_SUMMARY_PROMPT
    from schema import COMMUNITY_RELATIONSHIP_TYPES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build movie-industry communities for one or all graph backends."
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        required=True,
        help="Graph backend to process.",
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
        "--level",
        type=int,
        default=None,
        help="Highest community level to summarize/embed. Defaults to all levels.",
    )
    parser.add_argument(
        "--community-gamma",
        type=float,
        default=float(os.getenv("MEMGRAPH_COMMUNITY_GAMMA", "3.0")),
        help="Leiden gamma (resolution). Higher values create smaller communities.",
    )
    parser.add_argument(
        "--community-max-levels",
        type=int,
        default=int(os.getenv("MEMGRAPH_COMMUNITY_MAX_LEVELS", "3")),
        help="Maximum community hierarchy levels to write.",
    )
    parser.add_argument(
        "--community-theta",
        type=float,
        default=float(os.getenv("MEMGRAPH_COMMUNITY_THETA", "0.01")),
        help="Leiden theta parameter.",
    )
    parser.add_argument(
        "--community-tolerance",
        type=float,
        default=float(os.getenv("MEMGRAPH_COMMUNITY_TOLERANCE", "0.0001")),
        help="Leiden tolerance parameter.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=int(os.getenv("COMMUNITY_RANDOM_SEED", "42")),
        help="Random seed for deterministic community detection (Neo4j only; MAGE does not expose a seed).",
    )
    return parser.parse_args()


async def build_communities(
    backend: str,
    llm_provider: str,
    embedder_provider: str,
    level: int | None = None,
    community_gamma: float = 3.0,
    community_max_levels: int = 3,
    community_theta: float = 0.01,
    community_tolerance: float = 0.0001,
    random_seed: int = 42,
) -> dict:
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)
    results = {}

    for name, store, index_manager_cls in get_backend_targets(backend):
        print(f"\n=== Building {name} communities ===")
        index_manager_cls(store, embedding_dim=EMBEDDING_DIM).create_indexes()

        kwargs = {
            "relationship_types": COMMUNITY_RELATIONSHIP_TYPES,
            "summary_prompt": COMMUNITY_SUMMARY_PROMPT,
            "max_levels": community_max_levels,
            "gamma": community_gamma,
            "theta": community_theta,
            "tolerance": community_tolerance,
            "random_seed": random_seed,
        }

        community = CommunityPipeline(store, llm, embedder, **kwargs)
        results[name] = await community.build(level=level)
        print(f"{name} community result: {results[name]}")

    return results


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        build_communities(
            args.backend,
            args.llm_provider,
            args.embedder_provider,
            level=args.level,
            community_gamma=args.community_gamma,
            community_max_levels=args.community_max_levels,
            community_theta=args.community_theta,
            community_tolerance=args.community_tolerance,
            random_seed=args.random_seed,
        )
    )
