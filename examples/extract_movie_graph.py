"""Extract the movie graph once into a database-neutral JSON artifact.

Usage:
  python extract_movie_graph.py
  python extract_movie_graph.py --output artifacts/movie_graph.json
  python extract_movie_graph.py --llm-provider openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from recon_graphrag.extraction.artifacts import save_graph_document_json

from common import DEFAULT_ARTIFACT_PATH, extract_graph_document_from_pages
from config import get_llm
from data import MOVIE_EXAMPLE_PAGES
from schema import MOVIE_SCHEMA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the movie graph to a neutral JSON artifact."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "azure_openai"),
    )
    return parser.parse_args()


async def extract_to_artifact(output: Path, llm_provider: str):
    llm = get_llm(llm_provider)
    graph_document = await extract_graph_document_from_pages(
        MOVIE_EXAMPLE_PAGES,
        llm=llm,
        schema=MOVIE_SCHEMA,
        metadata={"source": "example"},
        window_size=2,
        window_overlap=1,
    )
    save_graph_document_json(graph_document, output)
    print(
        f"Saved graph artifact to {output} "
        f"({len(graph_document.entities)} entities, "
        f"{len(graph_document.relationships)} relationships)"
    )
    return graph_document


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(extract_to_artifact(args.output, args.llm_provider))
