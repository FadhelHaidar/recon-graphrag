# Quick Start

This guide walks you through building and searching your first knowledge graph with Recon-GraphRAG.

## Before you start

You need:

1. A running Neo4j instance with APOC/GDS or Memgraph instance with MAGE.
2. Recon-GraphRAG installed from GitHub.

See [Installation](installation.md) if you have not set these up yet.

## 1. Connect to a graph database

Both backends use the Bolt-compatible `GraphDatabase` driver. Choose the matching store and index manager.

### Neo4j

```python
from neo4j import GraphDatabase
from recon_graphrag import Neo4jGraphStore

driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "password")
)

store = Neo4jGraphStore(driver)
```

### Memgraph

```python
from neo4j import GraphDatabase
from recon_graphrag import MemgraphGraphStore

driver = GraphDatabase.driver("bolt://localhost:7689")
store = MemgraphGraphStore(driver)
```

## 2. Create indexes

Each backend store creates the vector and fulltext indexes required by the retrievers. Run this once after setting up the store.

```python
from recon_graphrag import IndexConfig

store.create_indexes(IndexConfig(), embedding_dim=1536)
```

The indexes created are:

- `chunk-embeddings` — vector index on `Chunk.embedding`
- `entity-embeddings` — vector index on `__Entity__.embedding`
- `community-embeddings` — vector index on `Community.embedding` for custom community retrieval
- `entity-names` — fulltext index on `__Entity__.name`

Use the backend-specific `IndexManager.verify()` to print the created indexes and node/relationship counts.

## 3. Define a schema

A schema tells the extraction pipeline which entities and relationships to look for. Start with a small schema:

```python
from recon_graphrag import GraphSchema, NodeType, PropertyType, RelationshipType

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
    patterns=[
        ("Person", "DIRECTED", "Movie"),
    ],
)
```

See [Schema](schema.md) for more ways to define schemas, including the `build_schema()` helper.

## 4. Create the LLM and embedder

Use the factory functions to create the providers you installed. This example uses OpenAI:

```python
from recon_graphrag import create_llm, create_embedder

llm = create_llm(
    "openai",
    model_name="gpt-4o",
    api_key="sk-...",
)

embedder = create_embedder(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-...",
)
```

See [Providers](providers.md) for Azure OpenAI, Ollama, OpenRouter, and sentence-transformer examples.

## 5. Build the graph

`GraphBuilderPipeline` extracts entities and relationships, resolves duplicate entities, and embeds the resulting graph:

```python
from recon_graphrag import GraphBuilderPipeline

pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
)

await pipeline.build_from_text(
    "Christopher Nolan directed Inception. Hans Zimmer composed the score.",
    metadata={"source": "quickstart"},
)
```

You can also ingest paginated text with `build_from_pages()` or pre-chunked documents with `build_from_documents()`.

## 6. Build communities

`CommunityPipeline` detects hierarchical communities, generates summaries, and embeds them:

```python
from recon_graphrag import CommunityPipeline

community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    relationship_types=["DIRECTED"],
)

await community.build()
```

`relationship_types` tells the pipeline which relationships form the community structure. Choose types that create meaningful connections between entities.

## 7. Search the graph

`GraphRAG` provides three search modes:

```python
from recon_graphrag import GraphRAG

graph_rag = GraphRAG(store, llm, embedder)

# Specific question about an entity
local_result = await graph_rag.search(
    "Who directed Inception?",
    mode="local",
    top_k=10,
)

# Broad overview using community summaries
global_result = await graph_rag.search(
    "What are the main themes?",
    mode="global",
    community_level="coarsest",
)

# Hybrid detail + context
drift_result = await graph_rag.search(
    "Tell me about Christopher Nolan's work.",
    mode="drift",
    top_k=10,
    community_top_k=3,
)
```

Search results include `answer`, `context`, and structured source data. When a
mode can resolve evidence, citations are available both as a flat list and
grouped by document:

```python
print(local_result.answer)

for source in local_result.sources:
    print(source.document_name or source.document_id)
    for citation in source.chunk_list:
        print(citation.chunk_id, citation.page_start, citation.page_end)
        print(citation.metadata)
```

`citation.metadata` contains the arbitrary metadata you supplied during
ingestion, so it can carry page numbers, database row IDs, API object IDs,
ticket IDs, list-item IDs, or other source-specific keys.

> **Note on community levels:** In Recon-GraphRAG, `level=0` means the **finest / most local** communities. This is the opposite of some Microsoft GraphRAG descriptions. See [Search](search.md) for details.

## Complete Neo4j example

Here is the full script in one block:

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

# Connect
driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "password"))
store = Neo4jGraphStore(driver)

# Indexes
store.create_indexes(IndexConfig(), embedding_dim=1536)

# Schema
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

# Providers
llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")

# Build graph
pipeline = GraphBuilderPipeline(graph_store=store, llm=llm, embedder=embedder, schema=schema)
await pipeline.build_from_text("Christopher Nolan directed Inception.")

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
```

To run the complete example on Memgraph, replace the connection setup with the Memgraph snippet from step 1. The schema, providers, pipelines, and search calls remain unchanged.

## Next steps

- Learn more about the two-stage pipeline in [Pipelines](pipelines.md).
- Explore schema options in [Schema](schema.md).
- Compare search modes in [Search](search.md).
- Run a complete domain example in [Example](example.md).
