"""Tests for schema validation of extracted graph data."""

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)
from recon_graphrag.extraction.validator import SchemaValidator


def _make_movie_schema():
    return GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                properties=[PropertyType(name="name", type="STRING")],
            ),
            NodeType(
                label="Movie",
                properties=[
                    PropertyType(name="title", type="STRING"),
                    PropertyType(name="year", type="STRING"),
                ],
                identity_property="title",
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED"),
            RelationshipType(label="ACTED_IN"),
        ],
        patterns=[("Person", "DIRECTED", "Movie"), ("Person", "ACTED_IN", "Movie")],
    )


def test_validator_drops_unknown_node_labels():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="x1", label="Unknown", properties={"name": "X"}),
        ],
        relationships=[],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    assert len(result.nodes) == 1
    assert result.nodes[0].label == "Person"


def test_validator_filters_properties():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(
                id="m1",
                label="Movie",
                properties={"title": "Inception", "year": "2010", "extra": "bad"},
            ),
        ],
        relationships=[],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    props = result.nodes[0].properties
    assert "title" in props
    assert "year" in props
    assert "extra" not in props
    assert "name" in props
    assert "description" in props


def test_validator_sets_name_from_identity_property():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(
                id="m1", label="Movie", properties={"title": "Inception", "year": "2010"}
            ),
        ],
        relationships=[],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    assert result.nodes[0].properties["name"] == "Inception"


def test_validator_drops_unknown_relationship_types():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="m1", label="Movie", properties={"title": "Inception"}),
        ],
        relationships=[
            ExtractedRelationship(
                source_id="p1", target_id="m1", type="PRODUCED"
            ),
        ],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    assert len(result.relationships) == 0


def test_validator_drops_relationships_with_missing_endpoints():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
        ],
        relationships=[
            ExtractedRelationship(
                source_id="p1", target_id="m1", type="DIRECTED"
            ),
        ],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    assert len(result.relationships) == 0


def test_validator_drops_invalid_patterns():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
        ],
        relationships=[
            ExtractedRelationship(
                source_id="p1", target_id="p2", type="DIRECTED"
            ),
        ],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    assert len(result.relationships) == 0


def test_validator_adds_defaults_to_relationships():
    schema = _make_movie_schema()
    extraction = GraphExtraction(
        nodes=[
            ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="m1", label="Movie", properties={"title": "Inception"}),
        ],
        relationships=[
            ExtractedRelationship(
                source_id="p1", target_id="m1", type="DIRECTED"
            ),
        ],
    )
    validator = SchemaValidator()
    result = validator.validate(extraction, schema)
    rel = result.relationships[0]
    assert rel.properties["description"] == ""
    assert rel.properties["weight"] == 1.0


# Property contracts apply equally to retained nodes and relationships. IDs and
# values intentionally contain sensitive text to guard against error disclosure.
def _property_extraction(properties, kind):
    nodes = [
        ExtractedNode(id="private source", label="Thing", properties={}),
        ExtractedNode(id="private target", label="Thing", properties={}),
    ]
    relationships = []
    if kind == "nodes":
        nodes = nodes[:1]
        nodes[0].properties = properties
    else:
        relationships.append(ExtractedRelationship(
            source_id="private source", target_id="private target", type="LINK",
            properties=properties,
        ))
    return GraphExtraction(nodes=nodes, relationships=relationships)


def _property_schema(prop, kind):
    return GraphSchema(
        node_types=[NodeType(label="Thing", properties=[prop] if kind == "nodes" else [])],
        relationship_types=[RelationshipType(
            label="LINK", properties=[prop] if kind == "relationships" else [],
        )],
    )


def _validate_property(ptype, value, kind="nodes", required=False):
    extraction = _property_extraction({"value": value}, kind)
    schema = _property_schema(PropertyType("value", ptype, required=required), kind)
    result = SchemaValidator().validate(extraction, schema)
    return getattr(result, kind)[0].properties["value"]


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize(("ptype", "value"), [
    ("STRING", "  unchanged  "), ("STRING", ""),
    ("INTEGER", 0), ("INTEGER", -42), ("FLOAT", 3), ("FLOAT", -1.5),
    ("BOOLEAN", True), ("BOOLEAN", False),
    ("DATE", "2024-02-29"), ("DATETIME", "2024-02-29T12:34:56+00:00"),
    ("LIST", []), ("LIST", ["one", 2, 3.5, True, ""]),
])
def test_validator_preserves_valid_declared_property_types(kind, ptype, value):
    assert _validate_property(ptype, value, kind) == value


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize(("ptype", "value"), [
    ("STRING", 42), ("STRING", True), ("STRING", {"private value": 1}),
    ("INTEGER", "42"), ("INTEGER", 1.5), ("INTEGER", 1.0), ("INTEGER", True),
    ("FLOAT", "3.5"), ("FLOAT", True), ("FLOAT", float("nan")),
    ("FLOAT", float("inf")), ("FLOAT", -float("inf")), ("FLOAT", 10**400),
    ("BOOLEAN", "yes"), ("BOOLEAN", "false"), ("BOOLEAN", 1), ("BOOLEAN", 0),
    ("DATE", "2023-02-29"), ("DATE", "2024-01-01T12:34:56"), ("DATE", 20240101),
    ("DATETIME", "not a date private value"), ("DATETIME", 20240101),
    ("LIST", "one,two"), ("LIST", {"private value": 1}),
    ("LIST", [["private value"]]), ("LIST", [{"private value": 1}]),
    ("LIST", [None]), ("LIST", [float("nan")]), ("LIST", [float("inf")]),
])
def test_validator_rejects_invalid_declared_property_types_safely(kind, ptype, value):
    with pytest.raises(ValueError) as error:
        _validate_property(ptype, value, kind)
    assert str(error.value) == f"{kind}[0].properties.value: expected {ptype}"


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("ptype", ["STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "DATETIME", "LIST"])
def test_validator_preserves_optional_null_properties(kind, ptype):
    assert _validate_property(ptype, None, kind) is None


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("properties", [{}, {"value": None}])
def test_validator_rejects_missing_or_null_required_properties(kind, properties):
    extraction = _property_extraction(properties, kind)
    schema = _property_schema(PropertyType("value", "STRING", required=True), kind)
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(extraction, schema)
    assert str(error.value) == f"{kind}[0].properties.value: required property is missing or null"


@pytest.mark.parametrize(("ptype", "value"), [
    ("STRING", ""), ("INTEGER", 0), ("FLOAT", 0.0), ("BOOLEAN", False), ("LIST", []),
])
def test_validator_required_properties_accept_falsey_valid_values(ptype, value):
    assert _validate_property(ptype, value, required=True) == value


def test_validator_checks_required_properties_after_existing_defaults():
    schema = GraphSchema(
        node_types=[NodeType(label="Thing", properties=[
            PropertyType("name", required=True), PropertyType("description", required=True),
        ])],
        relationship_types=[RelationshipType(label="LINK", properties=[
            PropertyType("description", required=True), PropertyType("weight", "FLOAT", required=True),
        ])],
    )
    result = SchemaValidator().validate(_property_extraction({}, "relationships"), schema)
    assert result.nodes[0].properties == {"name": "private source", "description": ""}
    assert result.relationships[0].properties == {"description": "", "weight": 1.0}


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
def test_validator_filters_undeclared_properties_before_validation_without_mutating_input(kind):
    properties = {"value": "kept", "extra": {"nested": [None, {}]}}
    extraction = _property_extraction(properties, kind)
    schema = _property_schema(PropertyType("value"), kind)
    result = SchemaValidator().validate(extraction, schema)
    assert "extra" not in getattr(result, kind)[0].properties
    assert properties == {"value": "kept", "extra": {"nested": [None, {}]}}
    assert "description" not in properties


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("field", ["name", "description", "weight"])
@pytest.mark.parametrize("value", [{"private value": 1}, [["private value"]]])
def test_validator_rejects_nested_base_properties(kind, field, value):
    schema = GraphSchema([NodeType("Thing")], [RelationshipType("LINK")])
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(_property_extraction({field: value}, kind), schema)
    expected_type = "FLOAT" if field == "weight" else "STRING"
    assert str(error.value) == f"{kind}[0].properties.{field}: expected {expected_type}"


@pytest.mark.parametrize("weight", ["private value", "2.5", True, False, None, float("inf"), float("nan"), [], {}])
def test_validator_rejects_invalid_relationship_weights_before_assembly(weight):
    schema = GraphSchema([NodeType("Thing")], [RelationshipType("LINK")])
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(_property_extraction({"weight": weight}, "relationships"), schema)
    assert str(error.value).startswith("relationships[0].properties.weight: expected ")
    assert "private" not in str(error.value)


@pytest.mark.parametrize("weight", [0, -1.0, 1, 2.5])
def test_validator_preserves_finite_relationship_weights(weight):
    schema = GraphSchema([NodeType("Thing")], [RelationshipType("LINK")])
    result = SchemaValidator().validate(_property_extraction({"weight": weight}, "relationships"), schema)
    assert result.relationships[0].properties["weight"] == weight


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
def test_validator_rejects_nonobject_properties_with_safe_field_path(kind):
    schema = GraphSchema([NodeType("Thing")], [RelationshipType("LINK")])
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(_property_extraction(["private value"], kind), schema)
    assert str(error.value) == f"{kind}[0].properties: expected an object"


def test_validator_reports_original_index_after_filtering_unknown_nodes():
    schema = _property_schema(PropertyType("value", "INTEGER"), "nodes")
    extraction = _property_extraction({"value": "private value"}, "nodes")
    extraction.nodes.insert(0, ExtractedNode("private ID", "Unknown", {}))
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(extraction, schema)
    assert str(error.value) == "nodes[1].properties.value: expected INTEGER"


@pytest.mark.parametrize("value", [0, 7, -42])
@pytest.mark.parametrize("required", [False, True])
def test_validator_preserves_explicit_integer_name_override(value, required):
    schema = GraphSchema([NodeType("Thing", properties=[
        PropertyType("name", "INTEGER", required=required),
        PropertyType("title", "STRING"),
    ], identity_property="title")], [])
    extraction = GraphExtraction(nodes=[ExtractedNode(
        "private source", "Thing", {"name": value, "title": "Other identity"},
    )])
    result = SchemaValidator().validate(extraction, schema)
    assert result.nodes[0].properties == {
        "name": value, "title": "Other identity", "description": "",
    }
    assert type(result.nodes[0].properties["name"]) is int


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("value", [0, 7])
def test_validator_preserves_supplied_nonstring_description_override(kind, required, value):
    schema = _property_schema(PropertyType("description", "INTEGER", required=required), kind)
    extraction = _property_extraction({"description": value}, kind)
    result = SchemaValidator().validate(extraction, schema)
    assert getattr(result, kind)[0].properties["description"] == value
    assert type(getattr(result, kind)[0].properties["description"]) is int


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("properties", [{}, {"description": None}])
def test_validator_does_not_default_optional_nonstring_description(kind, properties):
    schema = _property_schema(PropertyType("description", "INTEGER"), kind)
    extraction = _property_extraction(properties, kind)
    result = SchemaValidator().validate(extraction, schema)
    actual = getattr(result, kind)[0].properties
    assert ("description" in actual) == ("description" in properties)
    if "description" in properties:
        assert actual["description"] is None


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("properties", [{}, {"description": None}])
def test_validator_required_nonstring_description_override_still_enforced(kind, properties):
    schema = _property_schema(PropertyType("description", "INTEGER", required=True), kind)
    extraction = _property_extraction(properties, kind)
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(extraction, schema)
    assert str(error.value) == f"{kind}[0].properties.description: required property is missing or null"


@pytest.mark.parametrize("properties", [{}, {"name": None}])
def test_validator_required_nonstring_name_override_still_enforced(properties):
    schema = _property_schema(PropertyType("name", "INTEGER", required=True), "nodes")
    extraction = _property_extraction(properties, "nodes")
    with pytest.raises(ValueError) as error:
        SchemaValidator().validate(extraction, schema)
    assert str(error.value) == "nodes[0].properties.name: required property is missing or null"


@pytest.mark.parametrize("properties", [{}, {"name": None}])
def test_validator_does_not_default_optional_nonstring_name(properties):
    schema = _property_schema(PropertyType("name", "INTEGER"), "nodes")
    result = SchemaValidator().validate(_property_extraction(properties, "nodes"), schema)
    actual = result.nodes[0].properties
    assert ("name" in actual) == ("name" in properties)
    if "name" in properties:
        assert actual["name"] is None


@pytest.mark.parametrize("required", [False, True])
def test_validator_defaults_omitted_integer_relationship_weight_before_required_check(required):
    schema = _property_schema(PropertyType("weight", "INTEGER", required=required), "relationships")
    result = SchemaValidator().validate(_property_extraction({}, "relationships"), schema)
    weight = result.relationships[0].properties["weight"]
    assert weight == 1
    assert type(weight) is int


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("weight", [1.0, 2.5])
def test_validator_integer_relationship_weight_still_rejects_supplied_float(required, weight):
    schema = _property_schema(PropertyType("weight", "INTEGER", required=required), "relationships")
    with pytest.raises(ValueError, match=r"relationships\[0\].properties.weight: expected INTEGER"):
        SchemaValidator().validate(_property_extraction({"weight": weight}, "relationships"), schema)


def test_validator_preserves_sdk_identity_trust_metadata():
    schema = GraphSchema([NodeType("Thing")], [])
    result = SchemaValidator().validate(GraphExtraction(nodes=[
        ExtractedNode("a", "Thing", {"name": "A"}, canonical_key_trusted=False),
    ]), schema)
    assert result.nodes[0].canonical_key_trusted is False
