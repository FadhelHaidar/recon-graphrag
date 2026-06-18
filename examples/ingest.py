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

from recon_graphrag.extraction.artifacts import load_graph_document_json

try:
    from .common import (
        BACKEND_CHOICES,
        DEFAULT_ARTIFACT_PATH,
        finalize_graph_ingest,
        get_backend_targets,
        write_graph_document_for_ingest,
    )
    from .config import get_embedder, get_llm
except ImportError:
    from common import (
        BACKEND_CHOICES,
        DEFAULT_ARTIFACT_PATH,
        finalize_graph_ingest,
        get_backend_targets,
        write_graph_document_for_ingest,
    )
    from config import get_embedder, get_llm


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
        default=os.getenv("EMBEDDER_PROVIDER", "azure_openai"),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
        help="LLM provider used for hybrid entity-resolution review.",
    )
    parser.add_argument(
        "--entity-resolution-strategy",
        choices=["exact", "normalized", "fuzzy", "hybrid"],
        default="normalized",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Do not recreate indexes before ingesting.",
    )
    parser.add_argument(
        "--skip-entity-embeddings",
        action="store_true",
        help="Write graph data without embedding entity nodes.",
    )
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help="Write graph data without resolving duplicate entity nodes.",
    )
    parser.add_argument(
        "--allow-ai-auto-merge",
        action="store_true",
        help=(
            "Allow hybrid entity resolution to merge LLM-approved review "
            "candidates."
        ),
    )
    return parser.parse_args()


async def ingest_artifact(
    input_path: Path,
    backend: str,
    embedder_provider: str,
    llm_provider: str = "openrouter",
    entity_resolution_strategy: str = "normalized",
    create_indexes: bool = True,
    resolve_entities: bool = True,
    embed_entities: bool = True,
    allow_ai_auto_merge: bool = False,
) -> dict:
    graph_document = load_graph_document_json(input_path)
    embedder = get_embedder(embedder_provider)
    llm = get_llm(llm_provider) if entity_resolution_strategy == "hybrid" else None
    targets = get_backend_targets(backend)

    results = {}
    for name, store, index_manager_cls in targets:
        print(f"\n=== Ingesting {input_path} into {name} ===")
        results[name] = {
            "write_stats": write_graph_document_for_ingest(
                store,
                graph_document,
                index_manager_cls=index_manager_cls,
                create_indexes=create_indexes,
            )
        }

    for name, store, _ in targets:
        print(f"\n=== Finalizing {name} ingest ===")
        results[name].update(
            await finalize_graph_ingest(
                store,
                graph_document,
                embedder,
                llm=llm,
                entity_resolution_strategy=entity_resolution_strategy,
                resolve_entities=resolve_entities,
                embed_entities=embed_entities,
                allow_ai_auto_merge=allow_ai_auto_merge,
            )
        )
    return results


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        ingest_artifact(
            args.input,
            args.backend,
            args.embedder_provider,
            llm_provider=args.llm_provider,
            entity_resolution_strategy=args.entity_resolution_strategy,
            create_indexes=not args.skip_indexes,
            resolve_entities=not args.skip_entity_resolution,
            embed_entities=not args.skip_entity_embeddings,
            allow_ai_auto_merge=args.allow_ai_auto_merge,
        )
    )
