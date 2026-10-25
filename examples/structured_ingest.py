"""Ingest structured data (CSV) into the graph: direct mapping + LLM notes pass.

Columns map deterministically to entities/relationships (no LLM per row); the
free-text ``notes`` column gets an LLM extraction pass whose entities anchor to
the row's Product. Run with a Neo4j from docker-compose up.

Usage:
  python structured_ingest.py
  python structured_ingest.py --llm-provider openai
  python structured_ingest.py --no-notes   # zero LLM calls end to end
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import os
import tempfile
from pathlib import Path

from neo4j import GraphDatabase

from recon_graphrag import (
    ColumnEntity,
    GraphBuilderPipeline,
    Neo4jGraphStore,
    RowMapping,
    RowRelationship,
    TextColumn,
    build_schema,
    create_embedder,
)

from config import get_llm

ORDERS = [
    {
        "order_id": "O-1",
        "supplier_name": "Acme Corp",
        "product_name": "Widget",
        "unit_price": "9.99",
        "region": "EU",
        "notes": "The Widget uses the AX-9 chip manufactured by ChipCo in Taiwan.",
    },
    {
        "order_id": "O-2",
        "supplier_name": "Acme Corp",
        "product_name": "Gadget",
        "unit_price": "24.50",
        "region": "US",
        "notes": "",
    },
    {
        "order_id": "O-3",
        "supplier_name": "Globex",
        "product_name": "Widget",
        "unit_price": "9.49",
        "region": "EU",
        "notes": "Competing Widget batch; ChipCo supplies the same AX-9 chip.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Structured-data ingestion example.")
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "azure_openai", "openai"],
        default=os.getenv("LLM_PROVIDER", "openrouter"),
    )
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Skip the notes column (direct mapping only, zero LLM extraction calls).",
    )
    return parser.parse_args()


# The notes prose mentions a Component — a label the table doesn't model.
# The default extraction schema (mapping.to_schema()) would drop it, so pass
# a richer one; keep the table's labels so resolution merges direct+extracted.
NOTES_SCHEMA = build_schema(
    node_types=[
        {"label": "Company"},
        {"label": "Product"},
        {"label": "Component", "description": "A part used in a product"},
    ],
    relationship_types=[
        {"label": "MENTIONS"},
        {"label": "MADE_BY", "description": "Component is manufactured by a company"},
    ],
    patterns=[],
)


def make_mapping(include_notes: bool) -> RowMapping:
    return RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="supplier_name"),
            ColumnEntity(
                label="Product",
                identity_column="product_name",
                properties={"unit_price": "unit_price"},
                types={"unit_price": "FLOAT"},
                description_template="{product_name}, supplied by {supplier_name} at {unit_price} USD",
            ),
        ],
        relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
        text_columns=(
            [TextColumn("notes", anchor_label="Product")] if include_notes else []
        ),
        record_id_column="order_id",
        metadata_columns=["region"],
    )


async def main():
    args = parse_args()
    mapping = make_mapping(include_notes=not args.no_notes)

    # Write the sample rows to a CSV to demonstrate build_from_csv end to end.
    csv_path = Path(tempfile.gettempdir()) / "recon_graphrag_orders.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ORDERS[0]))
        writer.writeheader()
        writer.writerows(ORDERS)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7688"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )
    store = Neo4jGraphStore(driver)
    llm = get_llm(args.llm_provider)
    embedder = create_embedder("sentence_transformers")

    try:
        pipeline = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            # Typed columns give the hybrid resolution LLM judge real signal.
            entity_resolution_context_properties=mapping.resolution_context(),
        )
        result = await pipeline.build_from_csv(
            csv_path, mapping, extraction_schema=NOTES_SCHEMA
        )
        print(f"Ingested {result['extraction']['chunks']} rows "
              f"({result['extraction']['llm_rows']} with LLM notes pass).")
        print(f"Validation: {result.get('validation')}")

        print("\nEntities in the graph:")
        for record in store.execute_query(
            "MATCH (e:__Entity__ {graph_name: 'entity-graph'}) "
            "RETURN e.type AS type, e.name AS name ORDER BY type, name"
        ):
            print(f"  {record['type']}: {record['name']}")

        print("\nRow provenance for the AX-9 chip:")
        for record in store.execute_query(
            "MATCH (c:Chunk)-[:FROM_CHUNK]->(e:__Entity__ {name: 'AX-9 chip'}) "
            "RETURN c.record_id AS record_id, c.region AS region"
        ):
            print(f"  cited from order {record['record_id']} ({record['region']})")
    finally:
        close = getattr(llm, "aclose", None)
        if inspect.iscoroutinefunction(close):
            await close()
        driver.close()


if __name__ == "__main__":
    asyncio.run(main())
