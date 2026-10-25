"""Auto-analyze a graph schema from sample documents.

For when you have no idea what schema fits your use case: the LLM
proposes one from sample text, which you inspect, tweak, and pass to
the build methods (build_from_text/build_from_documents require a schema).

Usage:
  python analyze_schema.py
  python analyze_schema.py --llm-provider openai
  python analyze_schema.py --hint "movie industry news"
  python analyze_schema.py --output artifacts/proposed_schema.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
from pathlib import Path

from recon_graphrag import aanalyze_schema, save_schema_json

from config import get_llm
from data import MOVIE_EXAMPLE_PAGES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Propose a graph schema from sample documents via LLM."
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
    )
    parser.add_argument(
        "--hint",
        default="",
        help="Optional domain description to guide the analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the proposed schema as JSON.",
    )
    return parser.parse_args()


async def main(llm_provider: str, hint: str, output: Path | None):
    llm = get_llm(llm_provider)
    try:
        texts = [page["text"] for page in MOVIE_EXAMPLE_PAGES]
        schema = await aanalyze_schema(llm, texts, hint=hint)

        print("Proposed node types:")
        for node in schema.node_types:
            props = ", ".join(sorted(node.property_names))
            print(f"  {node.label}: {node.description} [{props}]")

        print("\nProposed relationship types:")
        for rel in schema.relationship_types:
            print(f"  {rel.label}: {rel.description}")

        print("\nProposed patterns:")
        for source, rel, target in schema.patterns:
            print(f"  {source} -[{rel}]-> {target}")

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            save_schema_json(schema, output)
            print(f"\nSaved schema to {output} (edit it, then load_schema_json).")

        print(
            "\nInspect/tweak the schema, then pass it to "
            "build_from_text(..., schema=schema)."
        )
        return schema
    finally:
        close = getattr(llm, "aclose", None)
        if inspect.iscoroutinefunction(close):
            await close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.llm_provider, args.hint, args.output))
