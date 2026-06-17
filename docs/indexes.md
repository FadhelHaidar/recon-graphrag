# Indexes

Recon-GraphRAG uses Neo4j vector and fulltext indexes for retrieval. The `IndexManager` helper creates the indexes you need in one call.

## What IndexManager creates

```python
from recon_graphrag import IndexManager, Neo4jGraphStore

manager = IndexManager(store, embedding_dim=1536)
manager.create_indexes()
```

This creates:

| Index name | Type | Indexed data |
| --------- | --------- | --------- |
| `chunk-embeddings` | Vector | `Chunk.embedding` |
| `entity-embeddings` | Vector | `__Entity__.embedding` |
| `community-embeddings` | Vector | `Community.embedding` |
| `entity-names` | Fulltext | `__Entity__.name` |

These indexes power:

- Semantic search over chunks and entities.
- Keyword search over entity names.
- Community summary retrieval.

## When to create indexes

Create indexes once after setting up a new Neo4j database, before running any pipeline:

```python
IndexManager(store, embedding_dim=1536).create_indexes()
```

You do not need to recreate them every time you ingest more text, unless you change the embedding dimension or want to drop and rebuild the graph from scratch.

## Embedding dimensions

The `embedding_dim` parameter must match the dimension of your embedder.

| Provider | Typical dimension | Notes |
| --------- | --------- | --------- |
| OpenAI `text-embedding-3-small` | 1536 | Default for OpenAI examples. |
| OpenAI `text-embedding-3-large` | 3072 | Pass `embedding_dim=3072`. |
| Azure OpenAI | 1536 or 3072 | Depends on the deployment. |
| Sentence-Transformers | Auto-detected | `IndexManager` can detect it for sentence-transformers. |
| Ollama / OpenRouter | Varies | Check the model card and pass explicitly. |

For sentence-transformers, `IndexManager` can auto-detect the dimension:

```python
from recon_graphrag import create_embedder

embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
manager = IndexManager(store, embedder=embedder)
manager.create_indexes()
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

manager = IndexManager(store, embedding_dim=512)
manager.create_indexes()
```

Make sure `embedding_dim` matches the `dimensions` value.

## Verify indexes

Use `verify()` to print the existing indexes and node/relationship counts:

```python
manager.verify()
```

This is useful during development to confirm that:

- All required indexes exist.
- The expected nodes and relationships were created.

## Rebuilding indexes

If you change the embedding dimension or want to start fresh, drop the existing indexes and recreate them:

```python
manager.drop_indexes()
manager.create_indexes()
```

> **Warning:** Dropping indexes does not delete graph data, but it will remove the indexes until you recreate them.

## Next steps

- See the full indexing flow in [Quick Start](quickstart.md).
- Learn about pipeline parameters in [Pipelines](pipelines.md).
