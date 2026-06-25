# Recon-GraphRAG

Domain-agnostic GraphRAG SDK for Neo4j and Memgraph, with a pluggable graph-store backend and the [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) philosophy.

Like Microsoft GraphRAG, Recon-GraphRAG uses **community detection with multi-level hierarchical communities** to structure knowledge graphs, and provides the same three search paradigms — **Local**, **Global**, and **DRIFT** search — to answer questions at different levels of specificity.

> **Work in Progress** — This project is under active development and is not yet reliable for production use. APIs may change without notice, and features may be incomplete or unstable.

## What is Recon-GraphRAG?

Recon-GraphRAG is a Python library for building knowledge graphs from unstructured text and querying them with retrieval-augmented generation. It extracts entities and relationships with an LLM, resolves duplicates, detects hierarchical communities, and supports semantic search over entities, chunks, and community summaries.

Learn more about the two-stage pipeline in [docs/pipelines.md](docs/pipelines.md).

## Requirements

- **Python** >= 3.11
- One supported graph database:
  - **Neo4j** with **APOC** for merge helpers and **GDS** for Leiden community detection
  - **Memgraph** with **MAGE** for Leiden community detection

See [docs/installation.md](docs/installation.md) for detailed setup instructions. The included Docker Compose file configures both databases.

### Backend requirements

The `GraphStore` protocol is designed to be backend-agnostic. A backend must support the following server-side capabilities to run the full Recon-GraphRAG pipeline:

- Vector indexes and approximate nearest-neighbor search
- Fulltext indexes
- Multi-level community detection (e.g., Leiden or Louvain)
- Cypher-compatible query language
- Internal node identity-based embedding upsert
- Entity merge / deduplication

Neo4j and Memgraph implement the same `GraphStore` contract. Their index and community-detection implementations use each database's native capabilities.

## Install

The package is not yet on PyPI. Install it directly from GitHub with `uv`:

```bash
uv add git+https://github.com/FadhelHaidar/Recon-GraphRAG.git
uv sync
```

With optional extras:

```bash
uv add "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
uv sync
```

`openai`, `sentence-transformers`, and the Neo4j driver are included in the core package, so the `[all]` extra only adds the Ollama client and any remaining optional providers.

Pin to a specific release:

```bash
uv add git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.4.0
uv sync
```

See [docs/installation.md](docs/installation.md) for more install options (`pip`, editable install, clone-without-install, extras, version pinning, troubleshooting).

## Quick start

```python
from neo4j import GraphDatabase
from recon_graphrag import (
    GraphRAG,
    GraphBuilderPipeline,
    CommunityPipeline,
    Neo4jGraphStore,
    IndexConfig,
    create_llm,
    create_embedder,
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)

# Connect to Neo4j (see the quick start for the Memgraph alternative)
driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "password"))
store = Neo4jGraphStore(driver)

# Create indexes
store.create_indexes(IndexConfig(), embedding_dim=1536)

# Define a schema
schema = GraphSchema(
    node_types=[
        NodeType(
            label="Person",
            description="An individual such as an actor or director",
            properties=[
                PropertyType(name="occupation", type="STRING", description="Primary role or profession"),
            ],
        ),
        NodeType(
            label="Movie",
            description="A film or motion picture",
            properties=[
                PropertyType(name="release_year", type="STRING", description="Year the film was released"),
                PropertyType(name="genre", type="STRING", description="Primary genre"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(label="DIRECTED", description="Person directed a movie"),
    ],
    patterns=[("Person", "DIRECTED", "Movie")],
)

# Create providers
llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")

# Build the graph
pipeline = GraphBuilderPipeline(graph_store=store, llm=llm, embedder=embedder, schema=schema)
await pipeline.build_from_text("Christopher Nolan directed Inception...")

# Build communities
community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    relationship_types=["DIRECTED"],
)
await community.build()

# Search
graph_rag = GraphRAG(store, llm, embedder)
result = await graph_rag.search("What are the key findings?", mode="local")
print(result.answer)
for citation in result.citations:
    print(citation.metadata)  # arbitrary source metadata, e.g. record_id/table/page
```

For a step-by-step walkthrough, see [docs/quickstart.md](docs/quickstart.md).

## Documentation

| Document | Description |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Full installation guide, Docker setup, extras, and troubleshooting |
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step quick start |
| [docs/pipelines.md](docs/pipelines.md) | `GraphBuilderPipeline` and `CommunityPipeline` architecture |
| [docs/workflows.md](docs/workflows.md) | Composable building blocks below the high-level pipelines |
| [docs/schema.md](docs/schema.md) | Defining schemas with `GraphSchema` and `build_schema()` |
| [docs/indexing.md](docs/indexing.md) | Creating and managing Neo4j and Memgraph indexes |
| [docs/providers.md](docs/providers.md) | LLM and embedder providers |
| [docs/search.md](docs/search.md) | Local, global, and DRIFT search modes, citations, and source metadata |
| [docs/example.md](docs/example.md) | Movie industry example walkthrough |
| [docs/testing.md](docs/testing.md) | Running tests and integration test flags |

## Example

A complete movie industry example is available in [examples/](examples/):

```bash
cd examples
python extract.py
python ingest.py --backend all
python communities.py --backend all
python search.py --backend neo4j
python search.py --backend memgraph
```

See [docs/example.md](docs/example.md) for a full walkthrough.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, branch naming, commit conventions, and the pull request process.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

This project is licensed under the [MIT License](LICENSE).
