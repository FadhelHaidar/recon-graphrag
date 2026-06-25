# Defining a Schema

The schema tells the extraction pipeline which entities, properties, and relationships to look for in your text. A well-defined schema improves extraction quality and keeps the graph focused on your domain.

## Why schemas matter

The LLM extractor uses the schema to:

- Decide which node labels to create.
- Decide which relationship types to create.
- Validate extracted nodes and relationships.
- Filter out entities that do not match your domain.

## Using `GraphSchema` directly

The most explicit way to define a schema is with the `GraphSchema` class:

```python
from recon_graphrag import GraphSchema, NodeType, PropertyType, RelationshipType, build_schema

schema = GraphSchema(
    node_types=[
        NodeType(
            label="Movie",
            description="A film or motion picture",
            identity_property="name",
            properties=[
                PropertyType(name="release_year", type="STRING", description="Year the film was released"),
                PropertyType(name="genre", type="STRING", description="Primary genre of the film", required=False),
                PropertyType(name="plot_summary", type="STRING", description="Brief summary of the film's plot"),
            ],
        ),
        NodeType(
            label="Person",
            description="An individual in the film industry",
            identity_property="name",
            properties=[
                PropertyType(name="nationality", type="STRING", description="Country or countries the person is associated with"),
                PropertyType(name="occupation", type="STRING", description="Primary role such as director, actor, or composer"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(label="DIRECTED", description="Person directed a movie"),
        RelationshipType(
            label="ACTED_IN",
            description="Person acted in a movie",
            properties=[
                PropertyType(name="role", type="STRING", description="Character name"),
            ],
        ),
    ],
    patterns=[
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
    ],
)
```

### Schema components

| Component | Purpose |
| --------- | --------- |
| `NodeType` | A kind of entity. Has a `label`, a `description`, an `identity_property` (defaults to `"name"`), and optional `properties`. |
| `PropertyType` | A property on a node or relationship. Has a `name`, a `type` (defaults to `"STRING"`), an optional `description`, and a `required` flag (defaults to `False`). |
| `RelationshipType` | A kind of relationship. Has a `label`, a `description`, and optional `properties`. |
| `patterns` | Valid source-label / relationship / target-label triples. Used to validate extracted relationships. |

### Property types

Supported property types:

- `STRING`
- `INTEGER`
- `FLOAT`
- `BOOLEAN`
- `DATE`
- `DATETIME`
- `LIST`

Use `STRING` for dates, years, or any value that you do not need to compare numerically.

## Using `build_schema()`

For a more compact definition, use the `build_schema()` helper:

```python
from recon_graphrag import build_schema

schema = build_schema(
    node_types=[
        {
            "label": "Movie",
            "description": "A film or motion picture",
            "properties": [
                {"name": "release_year", "type": "STRING", "description": "Year the film was released"},
                {"name": "genre", "type": "STRING", "description": "Primary genre of the film"},
            ],
        },
        {
            "label": "Person",
            "description": "An individual in the film industry",
            "properties": [
                {"name": "occupation", "type": "STRING", "description": "Primary role such as director or actor"},
            ],
        },
    ],
    relationship_types=[
        {"label": "DIRECTED", "description": "Person directed a movie"},
        {"label": "ACTED_IN", "description": "Person acted in a movie"},
    ],
    patterns=[
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
    ],
)
```

`build_schema()` returns a `GraphSchema` object and is useful when loading schema definitions from JSON or YAML.

## Patterns

`patterns` constrains which relationships are allowed between node labels. Each pattern is a tuple:

```python
(source_label, relationship_label, target_label)
```

For example:

```python
patterns=[
    ("Person", "DIRECTED", "Movie"),
    ("Person", "ACTED_IN", "Movie"),
]
```

During extraction, relationships that do not match a pattern are dropped. This prevents the graph from accumulating low-quality or off-schema edges.

## GraphSchema helpers

`GraphSchema` provides helper methods for inspecting and validating the schema:

| Method | Purpose |
| ------ | ------- |
| `node_labels()` | Return a set of all node labels. |
| `relationship_labels()` | Return a set of all relationship labels. |
| `pattern_set()` | Return the patterns as a set of `(source, relationship, target)` tuples. |
| `get_node_type(label)` | Return the `NodeType` for a label, or `None`. |
| `get_relationship_type(label)` | Return the `RelationshipType` for a label, or `None`. |
| `is_valid_pattern(source, relationship, target)` | Check whether a triple is allowed by the patterns (or by the relationship labels when no patterns are defined). |
| `validate()` | Raise `ValueError` for duplicate labels, unknown pattern labels, or invalid patterns. Called automatically on construction. |

## Entity Identity

Recon-GraphRAG stores entity nodes with Microsoft GraphRAG-style separated
identity fields:

| Property | Purpose |
| -------- | ------- |
| `id` | UUID string used as the persisted entity identity. |
| `canonical_key` | Stable readable extraction key such as `person:alice`. |
| `human_readable_id` | Readable reference used in reports, citations, and prompts. |
| `name` / `title` | Display text for search results and graph inspection. |
| `description` | Consolidated entity description from source observations. |

During extraction, the LLM still emits readable IDs so relationships and claims
can refer to entities in the same chunk. During assembly those readable IDs are
mapped to deterministic UUIDs and preserved as `canonical_key` and
`human_readable_id`. This keeps storage identity stable while keeping report
references debuggable.

In Neo4j Browser, `<id>` is Neo4j's internal element identity. It is separate
from Recon's `id` property.

## Best practices

1. **Keep labels concise.** Short, clear labels are easier for the LLM to produce consistently.
2. **Write descriptive descriptions.** The description is shown to the LLM and strongly influences extraction quality.
3. **Start small.** Begin with 2–5 node types and a few relationships. Expand once the core extraction works.
4. **Use properties sparingly.** Only add properties that you will query or display.
5. **Define patterns for every relationship.** Patterns are the main guard against invalid edges.
6. **Reuse relationship types.** If multiple node pairs connect the same way, use one relationship type with patterns rather than many relationship types.

## Next steps

- Run the schema through a pipeline in [Quick Start](quickstart.md).
- Learn about indexing in [Indexing](indexing.md).
- See a large real-world schema in [Example](example.md).
