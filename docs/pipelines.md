# Pipelines

Recon-GraphRAG splits indexing into two pipelines: one that builds the entity graph, and one that builds hierarchical communities on top of it.

## Architecture overview

| Stage | Pipeline | Steps |
| --------- | --------- | --------- |
| **Graph Building** | `GraphBuilderPipeline` | 1. Extract entities & relationships via LLM |
| | | 2. Entity resolution (merge duplicates) |
| | | 3. Entity embedding |
| **Community Building** | `CommunityPipeline` | 4. Community detection (Neo4j GDS or Memgraph MAGE Leiden) |
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
    extraction_concurrency=5,   # max chunks extracted in parallel (default: 5)
)
```

### Build methods

#### `build_from_text`

Ingest a single string:

```python
result = await pipeline.build_from_text(
    "Christopher Nolan directed Inception...",
    metadata={
        "source": "movie-list",
        "record_id": "movie-row-001",
        "collection": "movies",
    },
)
```

`metadata` can contain any JSON-like keys you need for source tracking. The
same keys are stored on chunks and returned later through
`citation.metadata`.

#### `build_from_pages`

Ingest a list of pages, for example from a paginated document:

```python
pages = [
    {"text": "Page one text...", "metadata": {"page": 1, "file_id": "pdf-7"}},
    {"text": "Page two text...", "metadata": {"page": 2, "file_id": "pdf-7"}},
]
result = await pipeline.build_from_pages(pages)
```

#### `build_from_documents`

Ingest pre-chunked documents:

```python
documents = [
    {"text": "Chunk one...", "metadata": {"record_id": "row-1", "table": "tickets"}},
    {"text": "Chunk two...", "metadata": {"record_id": "row-2", "table": "tickets"}},
]
result = await pipeline.build_from_documents(documents)
```

### Key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `schema` | A `GraphSchema` defining entities, relationships, and patterns. |
| `chunk_size` | Target size in characters for each text chunk. |
| `chunk_overlap` | Overlap in characters between consecutive chunks. |
| `extraction_concurrency` | Maximum number of chunks to extract in parallel. Set to `1` for sequential extraction. |
| `entity_resolution_strategy` | Duplicate entity resolution strategy: `exact`, `normalized`, `fuzzy`, or `hybrid`. |
| `entity_resolution_aliases` | Optional alias hints used by the `hybrid` entity resolution strategy. |
| `entity_resolution_llm_guidance` | Optional guidance included in `hybrid` LLM review prompts. |
| `entity_resolution_context_properties` | Optional list or label-to-list mapping of extra properties to include in LLM review context. Defaults to safe non-internal properties. |
| `entity_resolution_conflict_properties` | Optional list or label-to-list mapping of properties that must not conflict before merge. |
| `entity_resolution_context_mode` | LLM review context mode. Defaults to `"safe_defaults"`; `"config_only"` uses only configured context properties. |
| `allow_ai_auto_merge` | Optional. When `True` with `hybrid`, LLM-approved review candidates can be promoted into actual merge groups. Defaults to `False`. |

When `entity_resolution_strategy="hybrid"`, the pipeline forwards its `embedder`
and `llm` to the store-level resolver so hybrid resolution can use embedding
scores and LLM review candidates. By default, LLM output is audit/review
metadata only. Set `allow_ai_auto_merge=True` to merge candidates where the LLM
returns `same_entity=true`, `merge_allowed=true`, and confidence meets the merge
threshold.

Hybrid LLM review sees compact entity profiles, not just names. The default
profile includes safe fields such as `name`, `title`, `description`,
`canonical_key`, `human_readable_id`, aliases, labels, and non-internal
properties. Internal fields such as embeddings, graph names, raw text, and
timestamps are excluded.

Use conflict properties for domain keys that should prevent unsafe merges:

```python
pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    entity_resolution_strategy="hybrid",
    entity_resolution_context_properties={
        "Movie": ["year", "description"],
        "Person": ["description", "birth_date"],
    },
    entity_resolution_conflict_properties={
        "Movie": ["year"],
        "Person": ["birth_date"],
    },
    entity_resolution_llm_guidance=(
        "Do not merge movies with different release years."
    ),
    allow_ai_auto_merge=True,
)
```

Conflict properties are conservative: missing values do not block a merge, but
two non-empty unequal values do. Blocked candidates are returned in
`review_groups` with `decision="blocked"` and are not sent to the LLM or
auto-merged.

### Tips

- Run `IndexManager.create_indexes()` before the first build.
- The pipeline can be run multiple times on new text; it appends to the existing graph.
- Pass `metadata` to link extracted chunks and entities back to any source
  shape: documents, pages, database rows, tickets, API objects, or list items.

## CommunityPipeline

`CommunityPipeline` detects hierarchical communities using Neo4j GDS or Memgraph MAGE Leiden, summarizes each community with an LLM, and embeds the summaries.

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
    relationship_weight_property="weight",  # numeric edge weight property
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
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `relationship_types` | Which relationship types form the community graph. **Required.** |
| `max_levels` | Maximum number of community hierarchy levels to detect. |
| `gamma` | Leiden resolution parameter. Higher values produce more communities. |
| `theta` | Leiden tolerance parameter. |
| `relationship_weight_property` | Name of the numeric relationship property to use as the Leiden edge weight, for example `"weight"`. Neo4j runs unweighted when this is omitted; Memgraph defaults to `"weight"`. |
| `summary_prompt` | Optional custom prompt for generating community summaries. |

### Choosing `relationship_types`

`relationship_types` determines which relationships are projected into the community detection graph. Choose relationship types that create meaningful connections between entities.

For example, in a movie domain:

```python
relationship_types=["DIRECTED", "ACTED_IN", "COMPOSED_MUSIC"]
```

would group people and movies connected through creative roles.

### Community levels

Recon-GraphRAG stores communities with `level=0` as the **finest / most local** level and higher numbers as broader parent communities. The highest available level is the **coarsest / most global** level.

That convention comes from the community detection path: Leiden returns a community path from fine to coarse, and Recon-GraphRAG writes it by enumerating that path:

```python
for level, community_id in enumerate(path):
    ...
```

This is the opposite of some Microsoft GraphRAG descriptions, where level 0 is often interpreted as the root or coarsest level. For search examples, see [Search](search.md).

## Next steps

- Explore the composable building blocks in [Workflows](workflows.md).
- Define your domain model in [Schema](schema.md).
- Create required indexes in [Indexing](indexing.md).
- Search the graph in [Search](search.md).
