# Indexing

Recon-GraphRAG uses native vector and fulltext indexes in Neo4j and Memgraph. Each backend store exposes a backend-neutral `create_indexes()` method that creates the required indexes in one call.

## What `create_indexes` creates

```python
store.create_indexes(embedding_dim=1536)
```

This creates:

| Index name | Type | Indexed data |
| --------- | --------- | --------- |
| `chunk-embeddings` | Vector | `Chunk.embedding` |
| `entity-embeddings` | Vector | `__Entity__.embedding` |
| `entity-names` | Fulltext | `__Entity__.name` |

These indexes power:

- Semantic search over chunks and entities.
- Keyword search over entity names.

---

## When to create indexes

Create indexes once after setting up a new graph database, before running any pipeline:

```python
store.create_indexes(embedding_dim=1536)
```

You do not need to recreate them every time you ingest more text, unless you change the embedding dimension or want to drop and rebuild the graph from scratch.

---

## Embedding dimensions

The `embedding_dim` parameter must match the dimension of your embedder.

| Provider | Typical dimension | Notes |
| --------- | --------- | --------- |
| OpenAI `text-embedding-3-small` | 1536 | Default for OpenAI examples. |
| OpenAI `text-embedding-3-large` | 3072 | Pass `embedding_dim=3072`. |
| Azure OpenAI | 1536 or 3072 | Depends on the deployment. |
| Sentence-Transformers | Auto-detected | Pass the embedder to `create_indexes` to detect the dimension. |
| Ollama / OpenRouter | Varies | Check the model card and pass explicitly. |

For sentence-transformers, detect the dimension and pass it explicitly:

```python
from recon_graphrag import create_embedder
from recon_graphrag.embeddings import detect_embedding_dim

embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
embedding_dim = detect_embedding_dim(embedder)
store.create_indexes(embedding_dim=embedding_dim)
```

For all other providers, pass `embedding_dim` explicitly.

### Reducing dimensions with model_params

Some OpenAI embedding models support a `dimensions` parameter:

```python
from recon_graphrag import create_embedder

embedder = create_embedder(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-...",
    model_params={"dimensions": 512},
)

store.create_indexes(embedding_dim=512)
```

Make sure `embedding_dim` matches the `dimensions` value.

Pass `IndexConfig` only when you need to override the managed index names or labels:

```python
from recon_graphrag import IndexConfig

store.create_indexes(
    IndexConfig(entity_vector_index="custom-entity-embeddings"),
    embedding_dim=1536,
)
```

---

## Verify indexes

Each backend has an internal `IndexManager` with a `verify()` helper that prints the created indexes and node/relationship counts. You can access it through the backend-specific module:

```python
from recon_graphrag.graphdb.neo4j.index_manager import IndexManager

IndexManager(store).verify()
```

This is useful during development to confirm that:

- All required indexes exist.
- The expected nodes and relationships were created.

---

## Rebuilding indexes

If you change the embedding dimension, recreate the managed indexes:

```python
store.create_indexes(embedding_dim=1536)
```

`create_indexes()` replaces its managed indexes. This does not delete graph data, but queries may briefly run without those indexes while they are recreated.

---

## Backend-specific `IndexManager`

Both backends still have an internal `IndexManager` (`recon_graphrag.graphdb.neo4j.index_manager` and `recon_graphrag.graphdb.memgraph.index_manager`) that `create_indexes()` delegates to. You do not need to use it directly unless you are extending a backend.

---

## Next steps

- See the full indexing flow in [Quick Start](02-quickstart.md).
- Learn about pipeline parameters in [Pipelines](05-pipelines.md).
