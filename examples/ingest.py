"""Ingest a movie graph JSON artifact into one or all graph backends.

Usage:
  python ingest.py --backend all
  python ingest.py --backend neo4j --input artifacts/movie_graph.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from recon_graphrag.embeddings import EntityEmbedder
from recon_graphrag.extraction.artifacts import load_graph_document_json

from common import BACKEND_CHOICES, DEFAULT_ARTIFACT_PATH, get_backend_targets
from config import EMBEDDING_DIM, get_embedder, get_llm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a neutral movie graph artifact into graph databases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Input JSON artifact path.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="all",
        help="Database backend to ingest.",
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["openrouter", "azure_openai", "openai", "sentence-transformer"],
        default=os.getenv("EMBEDDER_PROVIDER", "openrouter"),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
        help="LLM provider used for hybrid entity-resolution review.",
    )
    return parser.parse_args()


async def ingest_artifact(
    input_path: Path,
    backend: str,
    embedder_provider: str,
    llm_provider: str,
) -> dict:
    graph_document = load_graph_document_json(input_path)
    embedder = get_embedder(embedder_provider)
    llm = get_llm(llm_provider)

    results = {}
    for name, store, index_manager_cls in get_backend_targets(backend):
        print(f"\n=== Ingesting {input_path} into {name} ===")

        # 1. Create indexes
        index_manager_cls(store, embedding_dim=EMBEDDING_DIM).create_indexes()

        # 2. Write graph document
        write_stats = store.write_graph_document(graph_document)
        print(f"Write complete: {write_stats}")

        # 3. Backfill missing descriptions
        store.backfill_descriptions()

        # 4. Resolve duplicate entities
        print("Resolving duplicate entities ...")
        resolution = await store.resolve_entities_normalized(
            graph_name=graph_document.document.graph_name,
        )
        print(f"Entity resolution result: {resolution}")

        # 5. Embed entity nodes
        await EntityEmbedder(store, embedder).embed_entities()

        # 6. Validate graph build
        validation = store.validate_graph_build()
        print(f"Validation: {validation}")

        results[name] = {
            "write_stats": write_stats,
            "entity_resolution": resolution,
            "validation": validation,
        }

    return results


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        ingest_artifact(
            args.input,
            args.backend,
            args.embedder_provider,
            args.llm_provider,
        )
    )
