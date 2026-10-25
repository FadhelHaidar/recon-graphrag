"""Extract the movie graph once into a database-neutral JSON artifact.

Usage:
  python extract.py
  python extract.py --output artifacts/movie_graph.json
  python extract.py --llm-provider openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import os
from pathlib import Path

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.artifacts import save_graph_document_json
from recon_graphrag.extraction.chunking import PageWindowBuilder
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.validator import SchemaValidator

from common import DEFAULT_ARTIFACT_PATH
from config import get_llm
from data import MOVIE_EXAMPLE_PAGES
from prompts import EXTRACTION_PROMPT
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
        default=os.getenv("LLM_PROVIDER", "openrouter"),
    )
    return parser.parse_args()


def _make_document_id(pages: list[dict]) -> str:
    sources = "|".join(p.get("metadata", {}).get("source", "") for p in pages)
    digest = hashlib.sha256(sources.encode("utf-8")).hexdigest()[:12]
    return f"doc:movie-example:{digest}"


async def extract_to_artifact(output: Path, llm_provider: str):
    llm = get_llm(llm_provider)
    try:
        document_id = _make_document_id(MOVIE_EXAMPLE_PAGES)
        pages = MOVIE_EXAMPLE_PAGES
        page_texts = [item["text"] for item in MOVIE_EXAMPLE_PAGES]
        metadata = {
            "source": "movie-example",
            "page_count": len(MOVIE_EXAMPLE_PAGES),
        }

        # Passing dicts preserves per-page metadata.
        chunks = PageWindowBuilder(
            window_size=2,
            window_overlap=1,
        ).build_windows(
            pages=pages,
            document_id=document_id,
            metadata=metadata,
        )

        # The backend appends the schema, rules, and JSON format sections.
        prompt_builder = SchemaPromptBuilder(extraction_prompt=EXTRACTION_PROMPT)
        extractor = LLMGraphExtractor(llm, prompt_builder=prompt_builder)
        validator = SchemaValidator()
        chunk_extractions = {}

        print(f"Extracting {len(chunks)} chunks ...")
        results = await extractor.extract_all(
            chunks,
            MOVIE_SCHEMA,
            concurrency=5,
        )
        for chunk, (chunk_id, raw_extraction, _) in zip(chunks, results):
            validated = validator.validate(raw_extraction, MOVIE_SCHEMA)
            chunk_extractions[chunk.id] = validated
            node_count = len(validated.nodes)
            rel_count = len(validated.relationships)
            page_range = (
                f"{chunk.metadata.get('page_start', '?')}-"
                f"{chunk.metadata.get('page_end', '?')}"
            )
            print(
                f"  Chunk {chunk_id} | pages {page_range} | "
                f"extracted ({node_count} nodes, {rel_count} rels)"
            )

        text_hash = hashlib.sha256("\n\n".join(page_texts).encode("utf-8")).hexdigest()
        graph_document = GraphDocumentAssembler().assemble(
            document_id=document_id,
            text_hash=text_hash,
            chunks=chunks,
            chunk_extractions=chunk_extractions,
            metadata=metadata,
            graph_name="entity-graph",
        )

        all_sources = set()
        for chunk in graph_document.chunks:
            all_sources.update(chunk.metadata.get("source_ids", []))

        save_graph_document_json(graph_document, output)
        print(
            f"Saved graph artifact to {output} "
            f"({len(graph_document.entities)} entities, "
            f"{len(graph_document.relationships)} relationships, "
            f"{len(graph_document.chunks)} chunks, "
            f"{len(all_sources)} sources)"
        )
        for source in sorted(all_sources):
            print(f"  - {source}")
        return graph_document
    finally:
        close = getattr(llm, "aclose", None)
        if inspect.iscoroutinefunction(close):
            await close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(extract_to_artifact(args.output, args.llm_provider))
