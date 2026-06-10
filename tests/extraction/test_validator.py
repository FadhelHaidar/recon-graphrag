"""Tests for schema validation of extracted graph data."""

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
