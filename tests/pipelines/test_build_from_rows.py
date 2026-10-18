"""Tests for GraphBuilderPipeline structured-data ingestion."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
)
from recon_graphrag.extraction.structured import (
    ColumnEntity,
    RowMapping,
    RowRelationship,
    TextColumn,
)
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline

from tests.pipelines.test_graph_builder import FakeGraphStore


ROWS = [
    {
        "order_id": "O-1",
        "supplier_name": "Acme Corp",
        "product_name": "Widget",
        "price": "9.99",
        "notes": "Widget uses the AX-9 chip made by ChipCo.",
    },
    {
        "order_id": "O-2",
        "supplier_name": "Acme Corp",
        "product_name": "Widget",
        "price": "9.99",
        "notes": "",
    },
]


@pytest.fixture
def mapping():
    return RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="supplier_name"),
            ColumnEntity(
                label="Product",
                identity_column="product_name",
                properties={"price": "unit_price"},
                types={"unit_price": "FLOAT"},
            ),
        ],
        relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
        record_id_column="order_id",
    )


@pytest.fixture
def fake_writer():
    writer = MagicMock()
    writer.write_graph_document = MagicMock(
        return_value={"documents": 1, "chunks": 2, "entities": 2, "relationships": 1}
    )
    return writer


def _make_pipeline(llm, fake_writer):
    return GraphBuilderPipeline(
        graph_store=FakeGraphStore(),
        llm=llm,
        embedder=MagicMock(),
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )


@pytest.mark.asyncio
async def test_direct_only_makes_zero_llm_calls(mapping, fake_writer):
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    pipeline = _make_pipeline(llm, fake_writer)

    result = await pipeline.build_from_rows(
        [dict(row, notes="") for row in ROWS],
        mapping,
        metadata={"source": "orders"},
    )

    llm.ainvoke.assert_not_called()
    assert result["extraction"]["chunks"] == 2
    assert result["extraction"]["llm_rows"] == 0
    assert "validation" in result

    doc = fake_writer.write_graph_document.call_args.args[0]
    assert doc.document.id == "doc:orders"
    assert [chunk.metadata["record_id"] for chunk in doc.chunks] == ["O-1", "O-2"]

    entities = {e.canonical_key: e for e in doc.entities}
    assert set(entities) == {"Company:Acme Corp", "Product:Widget"}
    product = entities["Product:Widget"]
    assert product.type == "Product"
    assert product.properties["name"] == "Widget"
    assert product.properties["unit_price"] == 9.99
    assert product.properties["description"]  # deterministic, non-empty

    # Same relationship across both rows dedups to one record with a bumped count.
    (rel,) = doc.relationships
    assert rel.type == "SUPPLIES"
    assert rel.observation_count == 2


@pytest.mark.asyncio
async def test_cross_label_same_name_stays_separate(fake_writer):
    mapping = RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="brand"),
            ColumnEntity(label="Product", identity_column="item"),
        ],
    )
    pipeline = _make_pipeline(MagicMock(), fake_writer)

    await pipeline.build_from_rows(
        [{"brand": "Mercury", "item": "Mercury"}],
        mapping,
        metadata={"source": "collide"},
        finalize=False,
    )

    doc = fake_writer.write_graph_document.call_args.args[0]
    assert len(doc.entities) == 2
    assert len({e.id for e in doc.entities}) == 2  # distinct uuids


@pytest.mark.asyncio
async def test_text_columns_merge_llm_extraction_and_anchor(mapping, fake_writer):
    mapping = RowMapping(
        entities=mapping.entities,
        relationships=mapping.relationships,
        text_columns=[TextColumn("notes", anchor_label="Product")],
        record_id_column=mapping.record_id_column,
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "nodes": [
            {"id": "Widget", "label": "Product", "properties": {"name": "Widget"}},
            {"id": "ChipCo", "label": "Company", "properties": {"name": "ChipCo"}}
        ],
        "relationships": []
    }
    '''))
    pipeline = _make_pipeline(llm, fake_writer)

    result = await pipeline.build_from_rows(
        ROWS,
        mapping,
        metadata={"source": "orders"},
        max_gleanings=0,
        finalize=False,
    )

    # Only the row with non-empty notes gets an LLM pass.
    assert llm.ainvoke.await_count == 1
    assert result["extraction"]["llm_rows"] == 1

    doc = fake_writer.write_graph_document.call_args.args[0]
    entities = {e.canonical_key for e in doc.entities}
    # Direct scoped keys plus LLM name-keyed entities (merged later by resolution).
    assert entities == {"Company:Acme Corp", "Product:Widget", "Widget", "ChipCo"}

    mentions = [r for r in doc.relationships if r.type == "MENTIONS"]
    # Anchor edge to ChipCo; the self-link to "Widget" is skipped.
    assert len(mentions) == 1
    (mention,) = mentions
    assert mention.properties["source_canonical_key"] == "Product:Widget"
    assert mention.properties["target_canonical_key"] == "ChipCo"


@pytest.mark.asyncio
async def test_custom_extraction_schema_constrains_llm_pass(mapping, fake_writer):
    mapping = RowMapping(
        entities=mapping.entities,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "nodes": [
            {"id": "AX-9", "label": "Component", "properties": {"name": "AX-9"}},
            {"id": "ChipCo", "label": "Company", "properties": {"name": "ChipCo"}}
        ],
        "relationships": []
    }
    '''))
    pipeline = _make_pipeline(llm, fake_writer)

    companies_only = GraphSchema(
        node_types=[NodeType(label="Company")],
        relationship_types=[RelationshipType(label="MENTIONS")],
    )
    await pipeline.build_from_rows(
        [ROWS[0]],
        mapping,
        metadata={"source": "orders"},
        extraction_schema=companies_only,
        max_gleanings=0,
        finalize=False,
    )

    doc = fake_writer.write_graph_document.call_args.args[0]
    keys = {e.canonical_key for e in doc.entities}
    # Component "AX-9" was dropped by the validator against the custom schema.
    assert "AX-9" not in keys
    assert "ChipCo" in keys


@pytest.mark.asyncio
async def test_build_from_csv_and_sql(tmp_path, mapping, fake_writer):
    pipeline = _make_pipeline(MagicMock(), fake_writer)

    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,supplier_name,product_name,price\nO-1,Acme Corp,Widget,9.99\n",
        encoding="utf-8",
    )
    result = await pipeline.build_from_csv(path, mapping, finalize=False)
    assert result["extraction"]["document_id"] == "doc:orders-csv"
    assert result["extraction"]["chunks"] == 1

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE orders (order_id TEXT, supplier_name TEXT, product_name TEXT, price REAL)"
    )
    connection.execute("INSERT INTO orders VALUES ('O-1', 'Acme Corp', 'Widget', 9.99)")
    result = await pipeline.build_from_sql(
        connection, "SELECT * FROM orders", mapping, finalize=False
    )
    assert result["extraction"]["chunks"] == 1
    connection.close()

    doc = fake_writer.write_graph_document.call_args.args[0]
    product = next(e for e in doc.entities if e.canonical_key == "Product:Widget")
    assert product.properties["unit_price"] == 9.99  # already-typed value passes through


@pytest.mark.asyncio
async def test_invalid_mapping_raises_before_any_work(mapping, fake_writer):
    pipeline = _make_pipeline(MagicMock(), fake_writer)
    bad = RowMapping(
        entities=[ColumnEntity(label="P", identity_column="a")],
        relationships=[RowRelationship("P", "REL", "Missing")],
    )
    with pytest.raises(ValueError, match="unknown label"):
        await pipeline.build_from_rows([{"a": "x"}], bad)
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["initial", "validation", "continuation"])
async def test_row_text_failure_prevents_direct_and_extracted_row_writes(
    mapping, fake_writer, failure
):
    mapping = RowMapping(
        entities=mapping.entities, relationships=mapping.relationships,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    valid = MagicMock(content='{"nodes": [], "relationships": []}')
    invalid = MagicMock(content='{"nodes": [{"id": "p", "label": "Product", "properties": {"name": []}}]}')
    responses = {
        "initial": [valid, RuntimeError("private provider error")],
        "validation": [valid, invalid],
        "continuation": [
            valid, MagicMock(content="no"), valid,
            MagicMock(content="yes"), MagicMock(content="invalid json"),
        ],
    }[failure]
    llm = MagicMock(ainvoke=AsyncMock(side_effect=responses))
    pipeline = _make_pipeline(llm, fake_writer)
    error = ValueError if failure == "validation" else RuntimeError
    with pytest.raises(error, match=r"document=.*chunk=3"):
        # Neither the direct-only nor the successfully extracted text row may
        # be written when the last row's required text extraction fails.
        await pipeline.build_from_rows(
            [ROWS[1], ROWS[0], ROWS[0]], mapping, metadata={"source": "private-source"},
            extraction_concurrency=1,
            max_gleanings=1 if failure == "continuation" else 0,
        )
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["failure", "cancel", "provider-cancel"])
async def test_row_text_tasks_are_drained_without_queued_provider_calls(
    mapping, fake_writer, stop
):
    import asyncio

    mapping = RowMapping(
        entities=mapping.entities,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    started = asyncio.Event()
    trigger = asyncio.Event()
    cleaned = asyncio.Event()
    provider_tasks = []

    async def invoke(prompt):
        provider_tasks.append(asyncio.current_task())
        if len(provider_tasks) == 1:
            await trigger.wait()
            if stop == "provider-cancel":
                raise asyncio.CancelledError
            raise RuntimeError("private provider error")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    pipeline = _make_pipeline(MagicMock(ainvoke=AsyncMock(side_effect=invoke)), fake_writer)
    build = asyncio.create_task(pipeline.build_from_rows(
        [ROWS[0]] * 3, mapping, extraction_concurrency=2, max_gleanings=0,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    if stop == "cancel":
        build.cancel()
    else:
        trigger.set()
    error = RuntimeError if stop == "failure" else asyncio.CancelledError
    with pytest.raises(error):
        await asyncio.wait_for(build, timeout=2)
    assert cleaned.is_set()
    assert len(provider_tasks) == 2
    assert all(task.done() for task in provider_tasks)
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nodes", "relationships"])
async def test_row_text_parser_property_failure_preserves_safe_field_context(
    mapping, fake_writer, kind
):
    import json

    mapping = RowMapping(
        entities=mapping.entities, relationships=mapping.relationships,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    item = (
        {"id": "private-id", "label": "Product"} if kind == "nodes" else
        {"source_id": "private-id", "target_id": "private-id", "type": "SUPPLIES"}
    )
    invalid = MagicMock(content=json.dumps({kind: [{**item, "properties": []}]}))
    llm = MagicMock(ainvoke=AsyncMock(side_effect=[
        MagicMock(content='{"nodes": []}'), invalid,
    ]))
    pipeline = _make_pipeline(llm, fake_writer)
    with pytest.raises(ValueError) as caught:
        await pipeline.build_from_rows(
            [ROWS[1], ROWS[0], ROWS[0]], mapping,
            metadata={"source": "private-source-name"},
            extraction_concurrency=1, max_gleanings=0,
        )
    message = str(caught.value)
    assert "Validation failed (document=" in message
    assert f"chunk=3 field={kind}[0].properties: expected an object" in message
    assert "private-id" not in message
    assert "private-source-name" not in message
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
async def test_row_text_provider_value_error_cannot_supply_parser_field_context(
    mapping, fake_writer
):
    import traceback

    mapping = RowMapping(
        entities=mapping.entities,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    provider_message = "nodes[0].properties: expected an object"
    llm = MagicMock(ainvoke=AsyncMock(side_effect=ValueError(provider_message)))
    pipeline = _make_pipeline(llm, fake_writer)
    with pytest.raises(RuntimeError, match=r"Extraction failed .*document=.*chunk=1") as caught:
        await pipeline.build_from_rows([ROWS[0]], mapping, max_gleanings=0)
    assert provider_message not in "".join(traceback.format_exception(caught.value))
    assert "field=" not in str(caught.value)
    fake_writer.write_graph_document.assert_not_called()
