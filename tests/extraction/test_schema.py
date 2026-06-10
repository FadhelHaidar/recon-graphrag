"""Tests for internal schema classes and build_schema."""

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
    build_schema,
)


def test_direct_schema_construction():
    schema = GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                description="A human being.",
                properties=[
                    PropertyType(name="name", type="STRING"),
                ],
            ),
            NodeType(
                label="Movie",
                description="A film or motion picture.",
                properties=[
                    PropertyType(name="title", type="STRING"),
                    PropertyType(name="year", type="STRING"),
                ],
                identity_property="title",
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED", description="Person directed a movie"),
        ],
        patterns=[("Person", "DIRECTED", "Movie")],
    )
    assert schema.node_labels() == {"Person", "Movie"}
    assert schema.relationship_labels() == {"DIRECTED"}


def test_build_schema_with_string_properties():
    schema = build_schema(
        node_types=[
            {"label": "Company", "description": "A company", "properties": ["name"]},
            {"label": "Product", "description": "A product", "properties": ["name", "brand"]},
        ],
        relationship_types=[
            {"label": "SUPPLIES", "description": "Company supplies product"},
        ],
        patterns=[("Company", "SUPPLIES", "Product")],
    )
    assert schema.node_labels() == {"Company", "Product"}
    company = schema.get_node_type("Company")
    assert company.identity_property == "name"
    assert company.property_names == {"name"}


def test_build_schema_with_dict_properties():
    schema = build_schema(
        node_types=[
            {
                "label": "Movie",
                "properties": [
                    {"name": "title", "type": "STRING", "required": True},
                    {"name": "year", "type": "INTEGER"},
                ],
                "identity_property": "title",
            },
        ],
        relationship_types=[
            {
                "label": "DIRECTED",
                "properties": [
                    {"name": "role", "type": "STRING", "description": "Role in film"},
                ],
            },
        ],
        patterns=[],
    )
    movie = schema.get_node_type("Movie")
    assert movie.identity_property == "title"
    title_prop = next(p for p in movie.properties if p.name == "title")
    assert title_prop.required is True
    year_prop = next(p for p in movie.properties if p.name == "year")
    assert year_prop.type == "INTEGER"

    directed = schema.get_relationship_type("DIRECTED")
    role_prop = next(p for p in directed.properties if p.name == "role")
    assert role_prop.description == "Role in film"


def test_identity_property_defaults():
    node = NodeType(label="Person")
    assert node.identity_property == "name"

    node_explicit = NodeType(label="Movie", identity_property="title")
    assert node_explicit.identity_property == "title"


def test_duplicate_node_labels_rejected():
    with pytest.raises(ValueError, match="Duplicate node labels"):
        GraphSchema(
            node_types=[
                NodeType(label="Movie"),
                NodeType(label="Movie"),
            ],
            relationship_types=[],
        )


def test_duplicate_relationship_labels_rejected():
    with pytest.raises(ValueError, match="Duplicate relationship labels"):
        GraphSchema(
            node_types=[NodeType(label="Movie")],
            relationship_types=[
                RelationshipType(label="DIRECTED"),
                RelationshipType(label="DIRECTED"),
            ],
        )


def test_unknown_pattern_node_rejected():
    with pytest.raises(ValueError, match="unknown source node label"):
        GraphSchema(
            node_types=[NodeType(label="Movie")],
            relationship_types=[RelationshipType(label="DIRECTED")],
            patterns=[("Person", "DIRECTED", "Movie")],
        )


def test_unknown_pattern_relationship_rejected():
    with pytest.raises(ValueError, match="unknown relationship label"):
        GraphSchema(
            node_types=[
                NodeType(label="Person"),
                NodeType(label="Movie"),
            ],
            relationship_types=[RelationshipType(label="ACTED_IN")],
            patterns=[("Person", "DIRECTED", "Movie")],
        )
