# Search

Recon-GraphRAG provides three search modes through `GraphRAG.search()`:
**Local**, **Global**, and **DRIFT**. Use this page for both the public API and
the mechanics behind each mode.

## Search Modes

| Mode | Strategy | Best for |
| --- | --- | --- |
| **Local** | Entity subgraph traversal | Specific questions about named entities |
| **Global** | Community report map-reduce | Broad overviews and thematic landscapes |
| **DRIFT** | Entity + community hybrid | Questions needing detail plus surrounding context |

All modes use the same entry point:

```python
from recon_graphrag import GraphRAG

graph_rag = GraphRAG(store, llm, embedder)
result = await graph_rag.search("Your question", mode="local")
```

## Local Search

Local search answers specific questions by retrieving relevant entities, their
neighbors, and related text chunks.

```python
result = await graph_rag.search(
    "Who directed Inception?",
    mode="local",
    top_k=10,
)
```

| Parameter | Description |
| --- | --- |
| `top_k` | Number of top entities to retrieve. |
| `effective_search_ratio` | Over-fetch multiplier before post-filtering. |
| `ranker` | Hybrid ranker: `"naive"` or `"linear"`. |
| `alpha` | Required for the `"linear"` ranker. |

Local search returns citations when retrieved entities have source chunks.
By default, citation metadata is returned after synthesis but is not shown to
the LLM. Opt in when the answer should see source identifiers while being
written:

```python
result = await graph_rag.search(
    "Who directed Inception?",
    mode="local",
    include_citation_metadata=True,
    citation_metadata_keys=["record_id", "collection"],
)
```

## Global Search

Global search answers broad questions from community reports. It supports two
strategies.

### Semantic Strategy

`strategy="semantic"` is the default. It embeds the query, retrieves the top-k
community summaries from the `community-embeddings` vector index, then maps and
reduces those summaries into one answer.

```python
result = await graph_rag.search(
    "What are the main themes in this dataset?",
    mode="global",
    strategy="semantic",
    top_k=5,
    community_level="coarsest",
)
```

| Parameter | Description |
| --- | --- |
| `top_k` | Number of communities to include. |
| `level` / `community_level` | Which community level to search. |

Semantic global preserves the lower-cost legacy behavior and may return an
empty citation list.

### Paper Strategy

`strategy="paper"` reads all eligible successful reports at one resolved
community level, shuffles them with a reproducible seed, packs them into token
batches, scores map outputs for helpfulness, filters/sorts partial answers, and
reduces the highest-scoring partials.

```python
result = await graph_rag.search(
    "What are the main themes in this dataset?",
    mode="global",
    strategy="paper",
    community_level="coarsest",
    random_seed=42,
)
```

Paper strategy ignores vector community top-k retrieval. It requires a selected
level and skips failed or empty community reports.

Paper global citations resolve only explicit map-phase references to entities,
relationships, or claims. It does not cite every source in every used community.

## DRIFT Search

DRIFT search combines local entity retrieval with community context for
questions that need both detail and big-picture framing.

```python
result = await graph_rag.search(
    "Explain the relationship between Christopher Nolan and his frequent collaborators.",
    mode="drift",
    top_k=10,
    community_top_k=3,
    community_level="coarsest",
)
```

| Parameter | Description |
| --- | --- |
| `top_k` | Number of entities to retrieve. |
| `community_top_k` | Number of communities to expand into. |
| `community_level` | Which community level to use. |

DRIFT returns citations for the retrieved local source chunks. Community-summary
citations are a separate extension point.
Like local search, DRIFT accepts `include_citation_metadata=True` and optional
`citation_metadata_keys=[...]` to include compact citation metadata in the
answer synthesis context.

## Citations And Sources

`SearchResult` includes structured citation fields in addition to answer text:

```python
result = await graph_rag.search("Who directed Inception?", mode="local")

for citation in result.citations:
    print(citation.document_id, citation.chunk_id, citation.page_start)
    print(citation.metadata)

for source in result.sources:
    print(source.document_name or source.document_id)
    print([c.chunk_id for c in source.chunk_list])
```

`result.citations` is a flat list of cited chunks. `result.sources` groups the
same citations by document for response envelopes and UI display.

Citation fields:

| Field | Meaning |
| --- | --- |
| `document_id` | Required source document ID. |
| `chunk_id` | Required source chunk ID. |
| `document_name` | Optional display name resolved from metadata such as `title`, `source`, or `filename`. |
| `page_start` / `page_end` | Optional page range when ingestion supplied page provenance. |
| `excerpt` | Optional bounded snippet copied from stored chunk text. |
| `metadata` | Arbitrary source metadata copied from the cited document and chunk. Chunk metadata overrides document metadata on key conflicts. |

`metadata` supports vector-store-style source envelopes. If you ingest list
items, rows, tickets, or other independent records, put the record key in the
input metadata:

```python
await pipeline.build_from_text(
    item["text"],
    metadata={
        "record_id": item["id"],
        "collection": "support-tickets",
        "source": item["id"],
    },
)
```

The same keys are returned on `citation.metadata`, so callers can use
`citation.metadata["record_id"]` as the source identifier even when no document
page metadata exists.

Citation metadata is normally returned in the response envelope after answer
synthesis. For Local and DRIFT search, set `include_citation_metadata=True` to
also include compact citation metadata in the LLM context. Use
`citation_metadata_keys` to keep that prompt context small.

Current citation behavior:

- **Local** resolves retrieved entity evidence from `source_chunk_ids` to
  document/chunk citations.
- **DRIFT** includes the same local evidence citations alongside community
  context.
- **Paper global** resolves validated map references with `target_type` of
  `entity`, `relationship`, or `claim`.
- **Semantic global** may return an empty citation list.

Citation resolution is graph-scoped. A query on one `graph_name` will not resolve
chunks, claims, or entities from another graph.

## Search Diagnostics

Every `SearchResult` includes a `metadata` dict with diagnostics for debugging
and monitoring:

```python
result = await graph_rag.search(
    "What are the main themes?",
    mode="global",
    strategy="paper",
    community_level="coarsest",
)
print(result.metadata)
# {
#   "strategy": "paper",
#   "selected_level": 1,
#   "random_seed": 42,
#   "reports_available": 5,
#   "reports_used": 5,
#   "map_batches": 1,
#   "map_succeeded": 1,
#   "map_failed": 0,
#   "map_filtered_zero": 0,
#   "reduce_partials_used": 1,
#   "elapsed_ms": 10077,
# }
```

Common keys across modes:

| Key | Modes | Meaning |
| --- | --- | --- |
| `strategy` | global | `"semantic"` or `"paper"` |
| `communities_used` | global/semantic | Number of communities retrieved |
| `reports_available` | global/paper | Total reports at the selected level |
| `reports_used` | global/paper | Reports that passed quality filters |
| `map_batches` | global/paper | Number of map-phase batches |
| `map_succeeded` | global/paper | Batches that produced a partial answer |
| `map_failed` | global/paper | Batches that errored |
| `map_filtered_zero` | global/paper | Batches filtered for zero helpfulness |
| `reduce_partials_used` | global/paper | Partial answers included in reduce |
| `elapsed_ms` | global/paper | Total wall-clock time |

Diagnostics are read-only. Use them for logging, dashboards, or tuning search
parameters.

## Community Levels

Recon-GraphRAG stores communities with `level=0` as the **finest / most local**
level. Higher numbers are broader parent communities. The highest available
level is the **coarsest / most global** level.

This is the opposite of some Microsoft GraphRAG descriptions, where level 0 is
often interpreted as the coarsest root level.

Use this mapping when comparing terminology:

```text
Recon level 0       ~= Microsoft finest / deepest community level
Recon highest level ~= Microsoft C0 / root / coarsest community level
```

The search API supports semantic selectors:

```python
community_level="all"       # No level filter
community_level="finest"    # level 0
community_level="coarsest"  # Highest available level
community_level=0           # Exact stored level
community_level=1           # Exact stored level
```

Global search also accepts the existing `level=` argument for backward
compatibility:

```python
await graph_rag.search("What are the major themes?", mode="global", level=0)
await graph_rag.search(
    "What are the major themes?",
    mode="global",
    community_level="coarsest",
)
```

Passing `level=0` does not mean "most global" in this codebase. It means
"finest / most local community summaries." For Microsoft-style global summaries,
prefer `community_level="coarsest"`.

## Customization

Each retriever exposes prompt attributes that you can override for
domain-specific behavior.

### Local Prompt

```python
graph_rag.local.answer_prompt = (
    "You are a film analyst. Answer based on:\n{context}\n\nQuestion: {query}"
)
```

### Semantic Global Prompts

```python
graph_rag.global_.map_prompt = (
    "Based on this report segment about films, answer: {query}\n\nSegment: {summary}"
)
graph_rag.global_.reduce_prompt = (
    "Synthesize these film perspectives:\n{partial_answers}\n\nQuestion: {query}"
)
```

Paper global uses paper-specific map/reduce prompts in
`recon_graphrag.retrieval.global_paper`.

### DRIFT Prompt

```python
graph_rag.drift.answer_prompt = (
    "Given specific findings and broader film context, answer: {query}\n\n"
    "{entity_context}\n{community_context}\n{bridging_context}"
)
```

### Custom Retrieval Query

Advanced users can override the Cypher query used to fetch local or DRIFT
entity context:

```python
graph_rag.local.retrieval_query = (
    "OPTIONAL MATCH (node)-[r]-(neighbor) RETURN node.name AS title, score"
)
```

If you provide a custom query and want citations, return `source_chunk_ids` in
each row.

## How Search Works

Before search can work, the graph should contain:

| Node | What it represents | Key properties |
| --- | --- | --- |
| `__Entity__` | Extracted people, places, concepts, etc. | UUID `id`, `canonical_key`, `human_readable_id`, `name`, `title`, `description`, `embedding`, `graph_name` |
| `Chunk` | A text chunk from the original source unit | `id`, `text`, `embedding`, arbitrary source metadata such as `record_id`, `page`, `table`, `ticket_id` |
| `Document` | The source unit or container | metadata such as `title`, `source`, `filename`, `collection`, `external_id` |
| `Community` | A cluster of related entities | `summary`, `report_text`, `report_json`, `report_status`, `embedding`, `level` |
| `Claim` | A claim/covariate extracted from text | `description`, `claim_type`, `status`, `graph_name` |

Important relationships:

- `(Chunk)-[:FROM_CHUNK]->(__Entity__)` links an entity to evidence text.
- `(Claim)-[:SUBJECT_OF]->(__Entity__)` links a claim to its subject.
- `(Claim)-[:SOURCED_FROM]->(Chunk)` links a claim to evidence text.
- `(__Entity__)-[:IN_COMMUNITY]->(Community)` places an entity in a community.
- `(Community)-[:PARENT_COMMUNITY]->(Community)` builds the hierarchy.

Search is read-only on top of the graph produced by `GraphBuilderPipeline` and
`CommunityPipeline`.

Entity search uses the UUID `id` internally, while report references and
citations use readable keys such as `person:alice`. Citation resolution accepts
either form and resolves it within the active `graph_name`.

### Query Signals

Local and DRIFT use one query string to produce two retrieval signals:

1. **Vector signal**: embed the full query and search `entity-embeddings`.
2. **Keyword signal**: search `entity-names` with the full query text.

The two lists are normalized and fused by `HybridEntityRetriever`.

| Ranker | Score formula |
| --- | --- |
| `naive` | `score = max(vector_score, keyword_score)` |
| `linear` | `score = alpha * vector_score + (1 - alpha) * keyword_score` |

Neo4j passes the raw keyword query to Lucene full-text search. Memgraph rewrites
the text into a Tantivy query, preserving phrases and joining escaped tokens
with `OR`.

### Local Flow

```text
query
  -> vector + keyword entity search
  -> hybrid fusion
  -> top-k entities
  -> one-hop neighbors + source chunks
  -> answer prompt
  -> source_chunk_ids resolved to citations
```

The LLM sees the matched entities, one-hop relationships, and source snippets.
It does not see community summaries.

### Semantic Global Flow

```text
query
  -> query embedding
  -> top-k community summary vector search
  -> map prompt per selected community
  -> reduce prompt
```

This is the lower-cost default strategy. It searches summaries semantically and
does not attempt to process every report at the selected level.

### Paper Global Flow

```text
successful reports at one level
  -> deterministic shuffle
  -> token-batched map prompts
  -> helpfulness-scored partial answers
  -> sorted reduce prompt
  -> validated references resolved to citations
```

Structured report generation stores `report_json`, `report_text`, title,
compatibility `summary`, rating fields, version fields, `report_status`, and
`report_error`. Failed report generations are marked with
`report_status="failed"` and are not embedded or read by paper global search.

Paper map outputs may include stable references:

```json
{"target_id": "person:alice", "target_type": "entity"}
```

Supported target types are `entity`, `relationship`, and `claim`. Entity and
relationship references are normally readable `human_readable_id` /
`canonical_key` values, while the persisted entity `id` remains a UUID.

### DRIFT Flow

```text
query
  -> local-style entity retrieval
  -> extract community keys from matched entities
  -> fetch community summaries
  -> fetch bridging entities in those communities
  -> answer prompt with specific + broader + related context
  -> source_chunk_ids resolved to citations
```

DRIFT is useful when the question needs both evidence around a specific entity
and community-level framing.

## Context Format

The retrievers format graph records into readable text before prompting the LLM.

Local context:

```text
Finding: Nolan (Person)
Connections:
  Person: Nolan -[DIRECTED]-> Movie: Inception
  Person: Nolan -[DIRECTED]-> Movie: Interstellar
Evidence:
  Christopher Nolan directed Inception...
  Nolan's next film was Interstellar...
```

DRIFT adds broader and related context:

```text
=== Broader Context ===
Segment 12 (level 0):
This cluster centers on Christopher Nolan and his science-fiction films...

=== Related Entities ===
Related: [Movie] Inception
    Connected to: DIRECTED -> Nolan
    Connected to: ACTED_IN -> DiCaprio
```

Semantic global context:

```text
Report Segment 12 (level 0):
This cluster centers on Christopher Nolan and his science-fiction films...

Report Segment 15 (level 0):
This cluster covers Hans Zimmer's collaborations with major directors...
```

## Token Utilities

Token budgeting uses shared utilities from `recon_graphrag.utils.tokens`:

- `ApproximateTokenCounter` — fast estimate using `ceil(len(text) / 4)`.
  Always available, no dependencies.
- `TiktokenTokenCounter` — exact count using `tiktoken`. Requires the
  `tiktoken` package. Use when provider-level accuracy matters.
- `create_token_counter("approximate")` or `create_token_counter("tiktoken")`
  — factory function.
- `pack_items(items, max_tokens, counter)` — greedy packing for already ordered
  items. Returns included/excluded lists with token telemetry.

These are used internally by paper global search (map/reduce batching) and
community context packing. You can also use them directly for custom workflows:

```python
from recon_graphrag.utils.tokens import ApproximateTokenCounter, pack_items, PackItem

counter = ApproximateTokenCounter()
items = [PackItem(id="r1", text="report text..."), PackItem(id="r2", text="...")]
result = pack_items(items, max_tokens=4000, counter=counter)
print(f"Packed {len(result.included)} items, {result.used_tokens} tokens")
```

## Required Indexes

Search depends on indexes created by `IndexManager`:

| Index | Type | Indexed property | Used by |
| --- | --- | --- | --- |
| `entity-embeddings` | Vector | `__Entity__.embedding` | Local, DRIFT |
| `community-embeddings` | Vector | `Community.embedding` | Global, DRIFT |
| `chunk-embeddings` | Vector | `Chunk.embedding` | Not used directly by search modes |
| `entity-names` | Full-text / text | `__Entity__.name` | Local, DRIFT |

Neo4j uses `db.index.vector.queryNodes` and `db.index.fulltext.queryNodes`.
Memgraph uses `vector_search.search` and `text_search.search`.

## Community Pipeline Prerequisite

Global and DRIFT search need communities. The community pipeline:

1. Detects communities with Leiden.
2. Writes `Community` nodes and hierarchy edges.
3. Summarizes or generates structured reports.
4. Embeds canonical report text for semantic retrieval.

Only after embeddings exist can semantic global search retrieve communities by
vector similarity.

## Current Limits

- The query is not decomposed into sub-questions.
- Local and DRIFT do not perform entity linking before retrieval.
- Ranking is hybrid vector/keyword fusion, not a re-ranking model.
- Local and DRIFT collect immediate neighbors and source chunks, not arbitrary
  multi-hop paths.
- Global search uses one community level at a time.
- Semantic global map calls are sequential. Paper global uses bounded parallel
  map calls.

## When To Use Each Mode

- **Local** when the question names a specific entity or asks for a concrete
  fact.
- **Global semantic** when the question is broad and you want lower-cost top-k
  community search.
- **Global paper** when you want paper-style all-report processing at a selected
  level and can afford the extra LLM calls.
- **DRIFT** when the question needs specific evidence plus broader context.

## Next Steps

- Try the search examples in [Quick Start](quickstart.md).
- See a full domain example in [Example](example.md).
