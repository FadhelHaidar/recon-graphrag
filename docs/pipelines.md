# Pipelines

Recon-GraphRAG splits indexing into two pipelines: one that builds the entity graph, and one that builds hierarchical communities on top of it.

## Architecture overview

| Stage | Pipeline | Steps |
| --------- | --------- | --------- |
| **Graph Building** | `GraphBuilderPipeline` | 1. Extract entities & relationships via LLM |
| | | 2. Entity resolution (merge duplicates) |
| | | 3. Entity embedding |
| **Community Building** | `CommunityPipeline` | 4. Community detection (Leiden via the backend store) |
| | | 5. Community summarization (LLM) |

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
    chunk_size=1000,            # text chunking size in characters (default: 1000)
    chunk_overlap=200,          # overlap between chunks in characters (default: 200)
    graph_name="entity-graph",  # graph scope (default: "entity-graph")
    extraction_concurrency=5,   # max chunks extracted in parallel (default: 5)
    max_gleanings=1,            # follow-up extraction loops (default: 0)
    extract_claims=True,        # extract claims about entities (default: False)
)
```

### `GraphBuilderPipeline` build methods

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

Ingest a list of documents, one per source unit. Each document is passed to
`build_from_text()` internally, so it is re-chunked using the pipeline's
`TextChunker`:

```python
documents = [
    {"text": "Document one text...", "metadata": {"record_id": "row-1", "table": "tickets"}},
    {"text": "Document two text...", "metadata": {"record_id": "row-2", "table": "tickets"}},
]
results = await pipeline.build_from_documents(documents)
```

Returns a list of per-document results. To use pre-tokenized or pre-chunked
units, chunk the text yourself with `TextChunker` and pass each chunk through
`build_from_documents()` as its own `"text"`, or assemble `GraphDocument`
artifacts directly (see [Workflows](workflows.md)).

### Token-based chunking

By default, `chunk_size` and `chunk_overlap` are measured in characters. The
pipeline always uses the internal `TextChunker` with character units. For
token-based chunking, pre-chunk your documents and use `build_from_documents()`:

```python
from recon_graphrag.extraction.chunking import TextChunker
from recon_graphrag.utils import ApproximateTokenCounter

chunker = TextChunker(
    chunk_size=500,
    chunk_overlap=50,
    unit="token",
    token_counter=ApproximateTokenCounter(),
)
chunks = chunker.chunk_text(text, document_id="doc:test")
documents = [{"text": c.text, "metadata": c.metadata} for c in chunks]

results = await pipeline.build_from_documents(documents)
```

`ApproximateTokenCounter` uses `ceil(len(text) / 4)` as a fast estimate. For
provider-level accuracy, use `TiktokenTokenCounter` (requires the `tiktoken`
package).

### `GraphBuilderPipeline` key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `schema` | A `GraphSchema` defining entities, relationships, and patterns. |
| `chunk_size` | Target size in characters for each text chunk. Use token-based chunking by pre-chunking externally and calling `build_from_documents()`. |
| `chunk_overlap` | Overlap in characters between consecutive chunks. |
| `graph_name` | Graph scope for all created nodes and relationships. Defaults to `"entity-graph"`. |
| `graph_writer` | Optional `GraphWriter` implementation. When omitted, the pipeline writes directly to `graph_store`. |
| `extraction_concurrency` | Maximum number of chunks to extract in parallel. Set to `1` for sequential extraction. |
| `max_gleanings` | Number of follow-up extraction loops after the initial pass. Each loop asks the LLM whether it missed any entities, then extracts only the missed items. `0` = single-shot extraction (default). Higher values improve recall at the cost of more LLM calls. |
| `extract_claims` | When `True`, runs a second LLM call per chunk to extract claims, assertions, and covariates about extracted entities. Claims are stored as `Claim` nodes linked to their subject entity and source chunk, and are available as evidence in community reports and global search. Defaults to `False`. |
| `perform_entity_resolution` | When `True` (default), resolves duplicate entities after extraction. Set to `False` to skip resolution. |
| `embed_entities` | When `True` (default), embeds entity nodes after extraction and resolution. Set to `False` to skip embedding. |
| `fail_on_resolution_error` | When `True`, raises resolution errors instead of logging and continuing. Defaults to `False`. |
| `fail_on_embedding_error` | When `True`, raises embedding errors instead of logging and continuing. Defaults to `False`. |

### `GraphBuilderPipeline` entity resolution parameters

| Parameter | Description |
| --------- | --------- |
| `entity_resolution_strategy` | Duplicate entity resolution strategy: `exact`, `normalized`, `fuzzy`, or `hybrid`. |
| `entity_resolution_aliases` | Optional alias hints used by the `hybrid` entity resolution strategy. |
| `entity_resolution_llm_guidance` | Optional guidance included in `hybrid` LLM review prompts. |
| `entity_resolution_context_properties` | Optional list or label-to-list mapping of extra properties to include in LLM review context. Defaults to safe non-internal properties. |
| `entity_resolution_conflict_properties` | Optional list or label-to-list mapping of properties that must not conflict before merge. |
| `entity_resolution_context_mode` | Controls which properties appear in LLM review profiles. `"safe_defaults"` (default) includes non-internal extra properties. `"config_only"` includes only explicitly configured `context_properties`. `"all"` includes everything, including embeddings and internal fields. |
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

### Gleaning

By default, the extractor makes a single LLM call per chunk. Set
`max_gleanings` to enable follow-up extraction loops that catch missed entities:

```python
pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    max_gleanings=1,  # one follow-up loop
)
```

Each gleaning loop:

1. **Assess** — asks the LLM whether it missed any entities.
2. **Continue** — if the LLM says yes, extracts only the missed items.
3. **Merge** — adds only genuinely new nodes and relationships (deduped by ID).

The loop stops early when the LLM reports nothing was missed or the continuation
yields no new items. Higher values improve recall at the cost of more LLM calls
per chunk. Most domains benefit from `max_gleanings=1`; use `0` (default) when
extraction cost matters more than completeness.

### Claims extraction

Set `extract_claims=True` to run a second LLM call per chunk that extracts
claims, assertions, and covariates about the entities already found in that
chunk:

```python
pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    extract_claims=True,
)
```

Each claim has a subject entity, a claim type (e.g. `"role"`, `"status"`,
`"opinion"`, `"attribute"`), a description, and an optional status. Claims are
stored as `Claim` nodes in the graph with two edges:

- `(Claim)-[:SUBJECT_OF]->(Entity)` — which entity the claim is about.
- `(Claim)-[:SOURCED_FROM]->(Chunk)` — which text chunk the claim came from.

Claims are automatically available downstream:

- **Community reports** — claims linked to community entities are included in
  the report context and can be cited as evidence in findings.
- **Global search** — map-phase outputs can reference claims, which
  resolve to source chunk citations.

Claims are optional. The pipeline works without them; they simply add another
layer of structured evidence when enabled.

### Tips

- Run `store.create_indexes()` before the first build.
- The pipeline can be run multiple times on new text; it appends to the existing graph.
- Pass `metadata` to link extracted chunks and entities back to any source
  shape: documents, pages, database rows, tickets, API objects, or list items.

## CommunityPipeline

`CommunityPipeline` detects hierarchical communities using the backend store's Leiden implementation and summarizes each community with an LLM.

```python
from recon_graphrag import CommunityPipeline

community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    relationship_types=["DIRECTED", "ACTED_IN"],  # required
    graph_name="entity-graph",  # graph scope (default: "entity-graph")
    max_levels=3,               # hierarchy depth (default: 3)
    gamma=1.0,                  # Leiden resolution (default: 1.0)
    theta=0.01,                 # Leiden theta (default: 0.01)
    tolerance=1e-4,             # Leiden tolerance (default: 1e-4)
    relationship_weight_property="weight",  # numeric edge weight property
    random_seed=42,             # deterministic detection (default: 42)
    summary_prompt=None,        # custom summary prompt (uses default if None)
    use_reports=False,          # generate structured reports (default: False)
    report_rubric=None,         # rating rubric for structured reports
    summarize_concurrency=1,    # concurrent community summaries (default: 1)
    skip_existing=False,        # skip communities that already have a summary
    max_context_tokens=None,    # token budget for community context
    token_counter=None,         # token counter for context packing
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

### `CommunityPipeline` key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `relationship_types` | Which relationship types form the community graph. **Required.** |
| `graph_name` | Graph scope to detect communities within. Defaults to `"entity-graph"`. |
| `max_levels` | Maximum number of community hierarchy levels to detect. |
| `gamma` | Leiden resolution parameter. Higher values produce more communities. |
| `theta` | Leiden theta parameter. |
| `tolerance` | Leiden tolerance parameter. |
| `relationship_weight_property` | Name of the numeric relationship property to use as the Leiden edge weight, for example `"weight"`. Neo4j runs unweighted when this is omitted; Memgraph defaults to `"weight"`. |
| `random_seed` | Random seed for deterministic Neo4j community detection. |
| `summary_prompt` | Optional custom prompt for generating community summaries. |
| `use_reports` | When `True`, generate structured reports instead of plain summaries. |
| `report_rubric` | Optional rating rubric for structured reports. |
| `summarize_concurrency` | Maximum number of community summaries to generate in parallel. Defaults to `1`. Increase for faster community builds when the LLM provider supports high throughput. |
| `skip_existing` | Skip communities that already have a summary. |
| `max_context_tokens` | Maximum tokens for community context passed to the LLM. When set, degree-ranked context is packed to fit this budget. |
| `token_counter` | Token counter for context packing. Defaults to `ApproximateTokenCounter` when `max_context_tokens` is set. |

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
