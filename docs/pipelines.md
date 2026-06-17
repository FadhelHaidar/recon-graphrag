# Pipelines

Recon-GraphRAG splits indexing into two pipelines: one that builds the entity graph, and one that builds hierarchical communities on top of it.

## Architecture overview

| Stage | Pipeline | Steps |
| --------- | --------- | --------- |
| **Graph Building** | `GraphBuilderPipeline` | 1. Extract entities & relationships via LLM |
| | | 2. Entity resolution (merge duplicates) |
| | | 3. Entity embedding |
| **Community Building** | `CommunityPipeline` | 4. Community detection (GDS Leiden) |
| | | 5. Community summarization (LLM) |
| | | 6. Community embedding |

After both pipelines run, the graph is ready for `GraphRAG.search()` in `local`, `global`, or `drift` mode.

## GraphBuilderPipeline

`GraphBuilderPipeline` turns raw text into a resolved and embedded entity graph.

```python
from recon_graphrag import GraphBuilderPipeline

pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    chunk_size=1000,            # text chunking size (default: 1000)
    chunk_overlap=200,          # overlap between chunks (default: 200)
)
```

### Build methods

#### `build_from_text`

Ingest a single string:

```python
result = await pipeline.build_from_text(
    "Christopher Nolan directed Inception...",
    metadata={"source": "my-doc"},
)
```

#### `build_from_pages`

Ingest a list of pages, for example from a paginated document:

```python
pages = [
    {"text": "Page one text...", "metadata": {"page": 1}},
    {"text": "Page two text...", "metadata": {"page": 2}},
]
result = await pipeline.build_from_pages(pages)
```

#### `build_from_documents`

Ingest pre-chunked documents:

```python
documents = [
    {"text": "Chunk one...", "metadata": {"doc_id": "doc-1"}},
    {"text": "Chunk two...", "metadata": {"doc_id": "doc-1"}},
]
result = await pipeline.build_from_documents(documents)
```

### Key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation, usually `Neo4jGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `schema` | A `GraphSchema` defining entities, relationships, and patterns. |
| `chunk_size` | Target size in characters for each text chunk. |
| `chunk_overlap` | Overlap in characters between consecutive chunks. |

### Tips

- Run `IndexManager.create_indexes()` before the first build.
- The pipeline can be run multiple times on new text; it appends to the existing graph.
- Pass `metadata` to link extracted chunks and entities back to source documents.

## CommunityPipeline

`CommunityPipeline` detects hierarchical communities using the GDS Leiden algorithm, summarizes each community with an LLM, and embeds the summaries.

```python
from recon_graphrag import CommunityPipeline

community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    relationship_types=["DIRECTED", "ACTED_IN"],  # required
    max_levels=3,               # hierarchy depth (default: 3)
    gamma=1.0,                  # Leiden resolution (default: 1.0)
    theta=0.01,                 # Leiden tolerance (default: 0.01)
    summary_prompt=None,        # custom summary prompt (uses default if None)
)
```

### Build methods

#### Build all levels

```python
result = await community.build()
```

#### Build a specific level

```python
result = await community.build(level=0)
```

### Key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation, usually `Neo4jGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `relationship_types` | Which relationship types form the community graph. **Required.** |
| `max_levels` | Maximum number of community hierarchy levels to detect. |
| `gamma` | Leiden resolution parameter. Higher values produce more communities. |
| `theta` | Leiden tolerance parameter. |
| `summary_prompt` | Optional custom prompt for generating community summaries. |

### Choosing `relationship_types`

`relationship_types` determines which relationships are projected into the community detection graph. Choose relationship types that create meaningful connections between entities.

For example, in a movie domain:

```python
relationship_types=["DIRECTED", "ACTED_IN", "COMPOSED_MUSIC"]
```

would group people and movies connected through creative roles.

### Community levels

Recon-GraphRAG stores communities with `level=0` as the **finest / most local** level and higher numbers as broader parent communities. This is the opposite of some Microsoft GraphRAG descriptions. See [Community Level Numbering](community-level-numbering.md).

## Next steps

- Explore the composable building blocks in [Advanced Workflows](advanced-workflows.md).
- Define your domain model in [Schema](schema.md).
- Create required indexes in [Indexes](indexes.md).
- Search the graph in [Search](search.md).
