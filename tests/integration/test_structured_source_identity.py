"""Live checks for structured source identity and citation continuity.

These tests use deterministic structured imports against disposable databases.
They do not call an LLM or embedder provider. Run with::

    RUN_DATABASE_INTEGRATION_TESTS=1 uv run pytest \
        tests/integration/test_structured_source_identity.py -m database
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from recon_graphrag.extraction.structured import (
    ColumnEntity,
    RowMapping,
    RowRelationship,
)
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_build_from_rows import (
    _file_doc_id,
    _hashed_doc_id,
    _write_orders,
)

pytestmark = [pytest.mark.integration, pytest.mark.database]

ORDERS_HEADER = ["order_id", "supplier_name", "product_name", "price"]


@pytest.fixture
def orders_mapping():
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


@pytest.fixture(scope="module", params=["neo4j", "memgraph"])
def store(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "structured source identity regressions",
        fail_on_missing=True,
    )
    live_store = (
        get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    )
    try:
        yield live_store
    finally:
        live_store.driver.close()


@pytest.fixture
def graph(store):
    graph_name = f"stage5-source-identity-{uuid4().hex}"
    try:
        yield graph_name
    finally:
        cleanup_graph(store, graph_name)


def _structured_pipeline(store, graph_name):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=AssertionError("structured imports must not call the LLM")
    )
    return GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=MagicMock(),
        graph_name=graph_name,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )


def _document_id(result):
    return result["extraction"]["document_id"]


def _document_sources(store, graph_name):
    rows = store.execute_query(
        """
        MATCH (d:Document {graph_name: $graph_name})
        RETURN d.id AS id, d.source AS source
        ORDER BY d.id
        """,
        {"graph_name": graph_name},
    )
    return {row["id"]: row["source"] for row in rows}


def _citations(store, graph_name, document_id):
    chunk_ids = [
        row["id"]
        for row in store.execute_query(
            """
            MATCH (c:Chunk {graph_name: $graph_name})-[:PART_OF]->
                  (d:Document {id: $document_id, graph_name: $graph_name})
            RETURN c.id AS id
            ORDER BY id
            """,
            {"graph_name": graph_name, "document_id": document_id},
        )
    ]
    rows = store.resolve_chunk_citations(graph_name, chunk_ids)
    return {row["chunk_id"]: row for row in rows}


def _assert_citation(citation, *, document_id, document_name, record_id, value):
    assert citation["document_id"] == document_id
    assert citation["document_name"] == document_name
    assert citation["document_metadata"]["source"] == document_name
    assert citation["chunk_metadata"]["record_id"] == record_id
    assert f"product_name: {value}" in citation["excerpt"]


@pytest.mark.asyncio
async def test_same_basename_csv_sources_keep_citations_isolated(
    store, graph, tmp_path, orders_mapping
):
    pipeline = _structured_pipeline(store, graph)
    first_path = tmp_path / "first" / "orders.csv"
    second_path = tmp_path / "second" / "orders.csv"
    _write_orders(first_path, product_name="Widget")
    _write_orders(
        second_path,
        order_id="O-2",
        supplier_name="Globex",
        product_name="Gadget",
    )

    first_id = _document_id(
        await pipeline.build_from_csv(first_path, orders_mapping, finalize=False)
    )
    second_id = _document_id(
        await pipeline.build_from_csv(second_path, orders_mapping, finalize=False)
    )

    assert first_id == _file_doc_id(first_path)
    assert second_id == _file_doc_id(second_path)
    assert first_id != second_id
    assert _document_sources(store, graph) == {
        first_id: "orders.csv",
        second_id: "orders.csv",
    }
    second_citations = _citations(store, graph, second_id)
    _assert_citation(
        second_citations[f"{second_id}:row:0"],
        document_id=second_id,
        document_name="orders.csv",
        record_id="O-2",
        value="Gadget",
    )

    _write_orders(first_path, product_name="Widget Mark II")
    repeat_id = _document_id(
        await pipeline.build_from_csv(first_path, orders_mapping, finalize=False)
    )

    assert repeat_id == first_id
    assert _document_sources(store, graph) == {
        first_id: "orders.csv",
        second_id: "orders.csv",
    }
    first_citations = _citations(store, graph, first_id)
    assert list(first_citations) == [f"{first_id}:row:0"]
    _assert_citation(
        first_citations[f"{first_id}:row:0"],
        document_id=first_id,
        document_name="orders.csv",
        record_id="O-1",
        value="Widget Mark II",
    )
    assert _citations(store, graph, second_id) == second_citations


@pytest.mark.asyncio
async def test_same_workbook_sheets_keep_citations_isolated(
    store, graph, tmp_path, orders_mapping
):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "orders.xlsx"
    workbook = openpyxl.Workbook()
    first = workbook.active
    first.title = "First"
    first.append(ORDERS_HEADER)
    first.append(["O-1", "Acme Corp", "Widget", "9.99"])
    second = workbook.create_sheet("Second")
    second.append(ORDERS_HEADER)
    second.append(["O-2", "Globex", "Gadget", "1.50"])
    workbook.save(path)

    pipeline = _structured_pipeline(store, graph)
    default_id = _document_id(
        await pipeline.build_from_excel(path, orders_mapping, finalize=False)
    )
    explicit_id = _document_id(
        await pipeline.build_from_excel(
            path, orders_mapping, sheet="First", finalize=False
        )
    )
    second_id = _document_id(
        await pipeline.build_from_excel(
            path, orders_mapping, sheet="Second", finalize=False
        )
    )

    assert default_id == _file_doc_id(path, "First")
    assert explicit_id == default_id
    assert second_id == _file_doc_id(path, "Second")
    assert _document_sources(store, graph) == {
        default_id: "orders.xlsx",
        second_id: "orders.xlsx",
    }
    second_citations = _citations(store, graph, second_id)
    _assert_citation(
        second_citations[f"{second_id}:row:0"],
        document_id=second_id,
        document_name="orders.xlsx",
        record_id="O-2",
        value="Gadget",
    )

    workbook = openpyxl.load_workbook(path)
    workbook["First"]["C2"] = "Widget Mark II"
    workbook.save(path)

    repeat_id = _document_id(
        await pipeline.build_from_excel(path, orders_mapping, finalize=False)
    )

    assert repeat_id == default_id
    assert _document_sources(store, graph) == {
        default_id: "orders.xlsx",
        second_id: "orders.xlsx",
    }
    first_citations = _citations(store, graph, default_id)
    assert list(first_citations) == [f"{default_id}:row:0"]
    _assert_citation(
        first_citations[f"{default_id}:row:0"],
        document_id=default_id,
        document_name="orders.xlsx",
        record_id="O-1",
        value="Widget Mark II",
    )
    assert _citations(store, graph, second_id) == second_citations


@pytest.mark.asyncio
async def test_explicit_legacy_id_updates_only_that_record(
    store, graph, tmp_path, orders_mapping
):
    pipeline = _structured_pipeline(store, graph)
    path = tmp_path / "legacy" / "orders.csv"
    _write_orders(path, product_name="Widget")
    legacy_id = _hashed_doc_id("orders.csv")

    path_id = _document_id(
        await pipeline.build_from_csv(path, orders_mapping, finalize=False)
    )
    legacy_result = await pipeline.build_from_csv(
        path,
        orders_mapping,
        metadata={"document_id": legacy_id},
        finalize=False,
    )

    assert path_id != legacy_id
    assert legacy_result["extraction"]["document_id"] == legacy_id
    assert _document_sources(store, graph) == {
        legacy_id: "orders.csv",
        path_id: "orders.csv",
    }
    path_citations = _citations(store, graph, path_id)
    legacy_citations = _citations(store, graph, legacy_id)
    assert legacy_citations[f"{legacy_id}:row:0"]["document_metadata"][
        "document_id"
    ] == legacy_id
    _assert_citation(
        path_citations[f"{path_id}:row:0"],
        document_id=path_id,
        document_name="orders.csv",
        record_id="O-1",
        value="Widget",
    )

    _write_orders(path, product_name="Widget Mark II")
    repeat = await pipeline.build_from_csv(
        path,
        orders_mapping,
        metadata={"document_id": legacy_id},
        finalize=False,
    )

    assert repeat["extraction"]["document_id"] == legacy_id
    assert _document_sources(store, graph) == {
        legacy_id: "orders.csv",
        path_id: "orders.csv",
    }
    updated = _citations(store, graph, legacy_id)
    assert updated[f"{legacy_id}:row:0"]["document_metadata"][
        "document_id"
    ] == legacy_id
    _assert_citation(
        updated[f"{legacy_id}:row:0"],
        document_id=legacy_id,
        document_name="orders.csv",
        record_id="O-1",
        value="Widget Mark II",
    )
    assert _citations(store, graph, path_id) == path_citations
