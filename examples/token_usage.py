"""Track token usage across graph building and search.

The pipelines and retrievers below track token usage by default.
This example explicitly wraps the LLM with ``track_usage(llm, ledger=...)``.
The custom ``TokenUsageLedger`` and ``JsonlUsageSink`` persist events across calls
for later analysis. The default ledger retains only the current run's events.

Usage:
  python token_usage.py --backend neo4j --llm-provider openai
  python token_usage.py --backend memgraph --llm-provider openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from recon_graphrag import (
    CommunityPipeline,
    GraphBuilderPipeline,
    JsonlUsageSink,
    TokenUsageLedger,
    load_usage_events,
    render_usage_table,
    summarize_events,
    track_usage,
)
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
)
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.utils.tokens import ApproximateTokenCounter

from common import get_backend_targets
from config import get_embedder, get_llm


SAMPLE_TEXT = """\
Alice is the CEO of Acme Corp. Bob is the CTO of Acme Corp.
Alice and Bob founded Acme Corp in 2010. Acme Corp makes widgets.
"""

SCHEMA = GraphSchema(
    node_types=[
        NodeType(label="Person", description="A person"),
        NodeType(label="Company", description="A company"),
    ],
    relationship_types=[
        RelationshipType(label="WORKS_AT", description="Person works at a company"),
        RelationshipType(label="FOUNDED", description="Person founded a company"),
    ],
    patterns=[
        ("Person", "WORKS_AT", "Company"),
        ("Person", "FOUNDED", "Company"),
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo token usage observability across build and search."
    )
    parser.add_argument(
        "--backend",
        choices=["neo4j", "memgraph"],
        required=True,
        help="Graph backend to use.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
        help="LLM provider (defaults to LLM_PROVIDER env var, then openrouter).",
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["openrouter", "azure_openai", "openai", "sentence-transformer"],
        default=os.getenv("EMBEDDER_PROVIDER", "openrouter"),
        help="Embedder provider (defaults to EMBEDDER_PROVIDER env var, then openrouter).",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Path to persist JSONL usage events. Defaults to a temporary file.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    _, store, _ = get_backend_targets(args.backend)[0]
    embedder = get_embedder(args.embedder_provider)

    if args.jsonl_path:
        jsonl_path = args.jsonl_path
    else:
        fd, tmp = tempfile.mkstemp(suffix="_usage.jsonl")
        os.close(fd)
        jsonl_path = Path(tmp)
    print(f"Usage events will be written to: {jsonl_path}")

    ledger = TokenUsageLedger(sinks=[JsonlUsageSink(jsonl_path)])
    llm = track_usage(get_llm(args.llm_provider), ledger=ledger)

    builder = GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=embedder,
        graph_name="token-usage-demo",
        summarize_descriptions=True,
        summarization_concurrency=2,
    )

    print("\n--- Build ---")
    build_result = await builder.build_from_text(
        text=SAMPLE_TEXT,
        schema=SCHEMA,
        extraction_concurrency=2,
        max_gleanings=1,
    )
    if "token_usage" in build_result:
        print(render_usage_table(build_result["token_usage"]))

    print("\n--- Communities ---")
    community_pipeline = CommunityPipeline(
        graph_store=store,
        llm=llm,
        graph_name="token-usage-demo",
        embedder=embedder,
        embed_community_reports=True,
        summarize_concurrency=2,
        max_context_tokens=4000,
        token_counter=ApproximateTokenCounter(),
    )
    community_result = await community_pipeline.build()
    if "token_usage" in community_result:
        print(render_usage_table(community_result["token_usage"]))

    print("\n--- DRIFT search ---")
    drift = DriftSearchRetriever(
        store, llm, embedder, graph_name="token-usage-demo"
    )
    drift_result = await drift.search("What does Acme Corp make?", top_k=5)
    print(drift_result.answer)
    if "token_usage" in drift_result.metadata:
        print(render_usage_table(drift_result.metadata["token_usage"]))

    print("\n--- Rollup from JSONL ---")
    events = load_usage_events(jsonl_path)
    rollup = summarize_events(events)
    print(
        f"Total runs: {rollup['runs']}, calls: {rollup['totals']['calls']}, "
        f"tokens: {rollup['totals']['total_tokens']:,}"
    )
    for stage, totals in rollup["stages"].items():
        print(
            f"  {stage}: {totals['calls']} calls, "
            f"{totals['input_tokens']:,} in, {totals['output_tokens']:,} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
