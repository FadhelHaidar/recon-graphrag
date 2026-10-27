"""Tests for structured-data mapping, conversion, coercion, and row sources."""

import sqlite3

import pytest

from recon_graphrag.extraction.structured import (
    ColumnEntity,
    RowMapping,
    RowRelationship,
    TextColumn,
    coerce,
    iter_csv,
    iter_excel,
    iter_sql,
    row_text,
    rows_to_chunks_and_extractions,
)


@pytest.fixture
def mapping():
    return RowMapping(
        entities=[
            ColumnEntity(
                label="Company",
                identity_column="supplier_name",
                properties={"country": "country"},
            ),
            ColumnEntity(
                label="Product",
                identity_column="product_name",
                properties={"price": "unit_price"},
                types={"unit_price": "FLOAT"},
            ),
        ],
        relationships=[
            RowRelationship("Company", "SUPPLIES", "Product", properties={"order_id": "order_id"}),
        ],
        text_columns=[TextColumn("notes", anchor_label="Product")],
        record_id_column="order_id",
        metadata_columns=["region"],
    )


ROW = {
    "order_id": "O-1",
    "supplier_name": "Acme Corp",
    "product_name": "Widget",
    "price": "9.99",
    "country": "DE",
    "region": "EU",
    "notes": "Widget uses the AX-9 chip.",
}


def test_rows_to_chunks_and_extractions(mapping):
    chunks, extractions, llm_rows = rows_to_chunks_and_extractions(
        [ROW], mapping, document_id="doc:orders"
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "doc:orders:row:0"
    assert "supplier_name: Acme Corp" in chunk.text
    assert chunk.metadata["record_id"] == "O-1"
    assert chunk.metadata["region"] == "EU"
    assert chunk.metadata["row_index"] == 0

    extraction = extractions[chunk.id]
    nodes = {node.id: node for node in extraction.nodes}
    assert set(nodes) == {"Company:Acme Corp", "Product:Widget"}

    product = nodes["Product:Widget"]
    assert product.label == "Product"
    assert product.properties["name"] == "Widget"
    assert product.properties["canonical_key"] == "Product:Widget"
    assert product.properties["unit_price"] == 9.99  # coerced FLOAT
    assert product.properties["description"] == "Product Widget — unit_price: 9.99"

    (rel,) = extraction.relationships
    assert rel.source_id == "Company:Acme Corp"
    assert rel.target_id == "Product:Widget"
    assert rel.type == "SUPPLIES"
    assert rel.properties == {"order_id": "O-1"}

    assert llm_rows == {chunk.id: [("Product:Widget", "Widget", "MENTIONS")]}


def test_empty_identity_skips_entity_and_relationships(mapping):
    row = {**ROW, "supplier_name": "  ", "notes": ""}
    chunks, extractions, llm_rows = rows_to_chunks_and_extractions(
        [row], mapping, document_id="doc:orders"
    )

    extraction = extractions[chunks[0].id]
    assert [node.id for node in extraction.nodes] == ["Product:Widget"]
    assert extraction.relationships == []
    assert llm_rows == {}  # empty notes -> no LLM pass


def test_cross_label_same_name_stays_separate():
    mapping = RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="col_a"),
            ColumnEntity(label="Person", identity_column="col_b"),
        ],
    )
    _, extractions, _ = rows_to_chunks_and_extractions(
        [{"col_a": "Mercury", "col_b": "Mercury"}], mapping, document_id="doc:x"
    )
    ids = {node.id for node in next(iter(extractions.values())).nodes}
    assert ids == {"Company:Mercury", "Person:Mercury"}


def test_description_template_uses_full_row():
    mapping = RowMapping(
        entities=[
            ColumnEntity(
                label="Product",
                identity_column="product_name",
                description_template="{product_name} sold in {region}",
            ),
        ],
    )
    _, extractions, _ = rows_to_chunks_and_extractions(
        [{"product_name": "Widget", "region": "EU"}], mapping, document_id="doc:x"
    )
    (node,) = next(iter(extractions.values())).nodes
    assert node.properties["description"] == "Widget sold in EU"


def test_record_id_defaults_to_row_index():
    mapping = RowMapping(entities=[ColumnEntity(label="P", identity_column="name")])
    chunks, _, _ = rows_to_chunks_and_extractions(
        [{"name": "a"}, {"name": "b"}], mapping, document_id="doc:x"
    )
    assert [chunk.metadata["record_id"] for chunk in chunks] == [0, 1]


@pytest.mark.parametrize(
    ("value", "ptype", "expected"),
    [
        ("42", "INTEGER", 42),
        ("9.99", "FLOAT", 9.99),
        ("yes", "BOOLEAN", True),
        ("No", "BOOLEAN", False),
        (True, "BOOLEAN", True),
        ("2026-07-09", "DATE", "2026-07-09"),
        ("2026-07-09T10:30:00", "DATETIME", "2026-07-09T10:30:00"),
        ("a, b , c", "LIST", ["a", "b", "c"]),
        (7, "STRING", "7"),
        ("", "FLOAT", None),
        (None, "INTEGER", None),
    ],
)
def test_coerce(value, ptype, expected):
    assert coerce(value, ptype, column="col", row_index=0) == expected


def test_coerce_failure_names_column_and_row():
    with pytest.raises(ValueError, match=r"column 'price' value 'oops' to FLOAT \(row 3\)"):
        coerce("oops", "FLOAT", column="price", row_index=3)


def test_to_schema(mapping):
    schema = mapping.to_schema()
    assert schema.node_labels() == {"Company", "Product"}
    assert schema.relationship_labels() == {"SUPPLIES", "MENTIONS"}
    product = schema.get_node_type("Product")
    assert {p.name for p in product.properties} >= {"unit_price", "name", "description"}
    assert next(p for p in product.properties if p.name == "unit_price").type == "FLOAT"
    assert schema.patterns == []


def test_resolution_context(mapping):
    assert mapping.resolution_context() == {
        "Company": ["country"],
        "Product": ["unit_price"],
    }


def test_validate_rejects_bad_mappings():
    with pytest.raises(ValueError, match="at least one ColumnEntity"):
        RowMapping(entities=[]).validate()

    with pytest.raises(ValueError, match="Duplicate ColumnEntity labels"):
        RowMapping(
            entities=[
                ColumnEntity(label="P", identity_column="a"),
                ColumnEntity(label="P", identity_column="b"),
            ]
        ).validate()

    with pytest.raises(ValueError, match="unknown label 'Q'"):
        RowMapping(
            entities=[ColumnEntity(label="P", identity_column="a")],
            relationships=[RowRelationship("P", "REL", "Q")],
        ).validate()

    with pytest.raises(ValueError, match="anchors to unknown label"):
        RowMapping(
            entities=[ColumnEntity(label="P", identity_column="a")],
            text_columns=[TextColumn("notes", anchor_label="Q")],
        ).validate()


def test_row_text_skips_empty_cells():
    assert row_text({"a": "1", "b": "", "c": None, "d": " x "}) == "a: 1\nd: x"


def test_iter_csv(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("name,price\nWidget,9.99\nGadget,1.50\n", encoding="utf-8")
    rows = list(iter_csv(path))
    assert rows == [
        {"name": "Widget", "price": "9.99"},
        {"name": "Gadget", "price": "1.50"},
    ]


def test_iter_sql_with_sqlite():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE products (name TEXT, price REAL)")
    connection.execute("INSERT INTO products VALUES ('Widget', 9.99)")
    rows = list(iter_sql(connection, "SELECT name, price FROM products WHERE price > ?", (1,)))
    assert rows == [{"name": "Widget", "price": 9.99}]
    connection.close()


def test_iter_excel(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "rows.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["name", "price"])
    sheet.append(["Widget", 9.99])
    workbook.save(path)

    rows = list(iter_excel(path))
    assert rows == [{"name": "Widget", "price": 9.99}]


def test_custom_identity_schema_and_metadata():
    mapping = RowMapping([ColumnEntity("Thing", "key", identity_property="record_key")])
    schema = mapping.to_schema()
    node_type = schema.get_node_type("Thing")
    assert node_type is not None
    assert node_type.identity_property == "record_key"
    assert "record_key" in node_type.property_names
    _, extractions, _ = rows_to_chunks_and_extractions(
        [{"key": "1"}], mapping, "doc",
    )
    node = next(iter(extractions.values())).nodes[0]
    assert node.properties["record_key"] == "1"
    assert node.canonical_key_trusted is True


@pytest.mark.parametrize("identity_property", ["", " ", None])
def test_invalid_identity_property_rejected(identity_property):
    mapping = RowMapping([ColumnEntity("Thing", "key", identity_property=identity_property)])
    with pytest.raises(ValueError, match="identity_property must be a non-empty string"):
        mapping.validate()


def test_custom_identity_cannot_be_overwritten_by_other_column():
    mapping = RowMapping([ColumnEntity(
        "Thing", "key", {"other": "record_key"}, identity_property="record_key",
    )])
    with pytest.raises(ValueError, match="identity_property must map from identity_column"):
        mapping.validate()


@pytest.mark.parametrize("explicit_mapping", [False, True])
@pytest.mark.parametrize(("ptype", "value", "expected"), [
    ("INTEGER", "0", 0), ("BOOLEAN", "false", False), ("STRING", " 01 ", "01"),
])
def test_custom_identity_honors_declared_type(explicit_mapping, ptype, value, expected):
    mapping = RowMapping([ColumnEntity(
        "Thing", "key", {"key": "record_key"} if explicit_mapping else {},
        types={"record_key": ptype}, identity_property="record_key",
    )])
    mapping.validate()
    node_type = mapping.to_schema().get_node_type("Thing")
    assert node_type is not None
    assert next(prop for prop in node_type.properties if prop.name == "record_key").type == ptype
    _, extractions, _ = rows_to_chunks_and_extractions([{"key": value}], mapping, "doc")
    node = extractions["doc:row:0"].nodes[0]
    assert node.properties["record_key"] == expected
    assert type(node.properties["record_key"]) is type(expected)


def test_custom_identity_rejects_invalid_declared_type_value():
    mapping = RowMapping([ColumnEntity(
        "Thing", "key", types={"record_key": "INTEGER"}, identity_property="record_key",
    )])
    with pytest.raises(ValueError, match="cannot coerce column 'key'.*INTEGER.*row 0"):
        rows_to_chunks_and_extractions([{"key": "invalid"}], mapping, "doc")
