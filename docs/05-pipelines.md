# Pipelines

Recon-GraphRAG splits indexing into two pipelines: one that builds the entity graph, and one that builds hierarchical communities on top of it.

## Architecture overview

| Stage | Pipeline | Steps |
| --------- | --------- | --------- |
| **Graph Building** | `GraphBuilderPipeline` | 1. Extract entities & relationships via LLM |
| | | 2. Entity resolution (merge duplicates) |
| | | 3. Entity/relationship description summarization |
| | | 4. Entity embedding |
| **Community Building** | `CommunityPipeline` | 4. Community detection (Leiden via the backend store) |
| | | 5. Community summarization (LLM) |

After both pipelines run, the graph is ready for search using `LocalSearchRetriever`, `GlobalSearchRetriever`, or `DriftSearchRetriever`.

---

## GraphBuilderPipeline

`GraphBuilderPipeline` turns raw text or structured rows into a resolved and embedded entity graph.

The constructor holds infrastructure and finalize configuration (entity
resolution, summarization, embedding), which apply to every build method.
Per-ingest knobs (schema, extraction concurrency, gleanings, chunking) are
parameters of the build methods.

```python
from recon_graphrag import GraphBuilderPipeline

pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    graph_name="entity-graph",  # graph scope (default: "entity-graph")
    summarize_descriptions=True,# summarize raw descriptions before embedding
)
```

### `GraphBuilderPipeline` build methods

#### `build_from_text`

Ingest a single string. `schema` is required — auto-generate one with
`analyze_schema`/`aanalyze_schema` if needed:

```python
result = await pipeline.build_from_text(
    "Christopher Nolan directed Inception...",
    metadata={
        "source": "movie-list",
        "record_id": "movie-row-001",
        "collection": "movies",
    },
    schema=schema,            # required
    extraction_concurrency=5, # max chunks extracted in parallel (default: 5)
    max_gleanings=1,          # follow-up extraction loops (default: 1)
    extract_claims=True,      # extract claims about entities (default: False)
    chunk_size=1200,          # text chunking size (default: 1200)
    chunk_overlap=100,        # overlap between chunks (default: 100)
    chunk_unit="tokens",      # "tokens" or "characters" (default: "tokens")
)
```

`metadata` can contain any JSON-like keys you need for source tracking. The
same keys are stored on chunks and returned later through
`citation.metadata`.

Chunking settings are per-call. By default `chunk_size` and `chunk_overlap`
are measured in tokens. Set `chunk_unit="characters"` for character-based
chunking; see [Token-based chunking](#token-based-chunking) below.

The token default is explicit in the pipeline and supersedes older examples that
used character-based chunking by default.

#### `build_from_documents`

Ingest a batch of document envelopes. Each envelope must contain exactly one
of:

- `"text"`: a raw text string.
- `"pages"`: a list of page strings or page dicts with `"text"`.

Each envelope may also include `"metadata"`. The method always returns a
`list[dict]` with one result per input envelope.

```python
documents = [
    {
        "text": "Document one text...",
        "metadata": {"record_id": "row-1", "table": "tickets"},
    },
    {
        "text": "Document two text...",
        "metadata": {"record_id": "row-2", "table": "tickets"},
    },
]
results = await pipeline.build_from_documents(documents, schema=schema)
```

Paginated documents use sliding page windows instead of text chunking:

```python
pages = [
    {"text": "Page one text...", "metadata": {"page": 1, "file_id": "pdf-7"}},
    {"text": "Page two text...", "metadata": {"page": 2, "file_id": "pdf-7"}},
]
results = await pipeline.build_from_documents(
    [{"pages": pages}],
    schema=schema,
    window_size=2,
    window_overlap=1,
)
result = results[0]
```

Text envelopes are chunked with `chunk_size`, `chunk_overlap`, `chunk_unit`,
`token_counter`, and `token_encoding`. Page envelopes are windowed with
`window_size` and `window_overlap`. You can mix text and page envelopes in the
same call.

#### Build method parameters

| Parameter | Applies to | Description |
| --------- | ---------- | ----------- |
| `schema` | `build_from_text`, `build_from_documents` | **Required.** A `GraphSchema` defining entities, relationships, and patterns. Auto-generate one with `analyze_schema`/`aanalyze_schema`. |
| `extraction_concurrency` | all build methods | Maximum number of chunks/rows extracted in parallel. Defaults to `5`. |
| `max_gleanings` | all build methods | Follow-up extraction loops after the initial pass. Defaults to `1`; use `0` for single-shot extraction. |
| `extract_claims` | `build_from_text`, `build_from_documents` | When `True`, runs a second LLM call per chunk to extract claims about entities. Defaults to `False`. |
| `document_concurrency` | `build_from_documents` | Maximum number of envelopes processed in parallel. Defaults to `1`. |
| `chunk_size` | `build_from_text`, `build_from_documents` text envelopes | Target chunk size. In tokens by default, or characters when `chunk_unit="characters"`. Defaults to `1200`. |
| `chunk_overlap` | `build_from_text`, `build_from_documents` text envelopes | Overlap between consecutive chunks. Defaults to `100`. |
| `chunk_unit` | `build_from_text`, `build_from_documents` text envelopes | `"tokens"` (default) or `"characters"`. |
| `token_counter` | `build_from_text`, `build_from_documents` text envelopes | Optional `TokenCounter`. Overrides the default tiktoken counter in token mode. |
| `token_encoding` | `build_from_text`, `build_from_documents` text envelopes | Tiktoken encoding name when `chunk_unit="tokens"` and no custom counter is provided. Defaults to `"cl100k_base"`. |
| `window_size` | `build_from_documents` page envelopes | Number of pages per window. Defaults to `2`. |
| `window_overlap` | `build_from_documents` page envelopes | Number of pages shared between consecutive windows. Defaults to `1`. |

#### What concurrency bounds, and what it does not

`extraction_concurrency` and `document_concurrency` bound how much work runs
**at once**. They do not bound the work a build does in total, and they do not
bound memory.

Internally a build uses a fixed pool of workers that pull from a shared queue,
so a 10,000-document batch with `document_concurrency=2` keeps at most two
documents in flight rather than creating 10,000 pending tasks. Still input-sized
and unavoidable for a single call:

- the `documents` / `rows` argument you pass in;
- the returned result list, one entry per document;
- one document's assembled extraction state, which is written once at the end.

For bounded memory on very large corpora, call the build method repeatedly with
smaller batches rather than passing one enormous list.

### Token-based chunking

By default, `chunk_size` and `chunk_overlap` are measured in tokens using
`TiktokenTokenCounter` with the `cl100k_base` encoding. Set
`chunk_unit="characters"` for character-based chunking instead:

```python
result = await pipeline.build_from_text(
    text,
    chunk_size=500,
    chunk_overlap=50,
    chunk_unit="characters",
)
```

Use a custom `TokenCounter` for non-OpenAI tokenizers:

```python
from recon_graphrag.utils import ApproximateTokenCounter

result = await pipeline.build_from_text(
    text,
    chunk_size=500,
    chunk_overlap=50,
    token_counter=ApproximateTokenCounter(),
)
```

`ApproximateTokenCounter` uses `ceil(len(text) / 4)` as a fast estimate. For
provider-level accuracy with OpenAI models, use the default tiktoken-backed
counter or pass `token_encoding`.

You can still pre-chunk text externally with `TextChunker` and pass each chunk
as its own `"text"` envelope if you need full control over chunk boundaries
(see [Workflows](09-workflows.md)).

### Structured data: `build_from_rows` / `build_from_csv` / `build_from_excel` / `build_from_sql`

Structured rows (CSV, Excel, SQL query results) are ingested via a `RowMapping`
that declares how columns become entities, properties, and relationships —
deterministically, with no LLM call per row. Columns holding free text can
additionally get an LLM extraction pass.

```python
from recon_graphrag import RowMapping, ColumnEntity, RowRelationship, TextColumn

mapping = RowMapping(
    entities=[
        ColumnEntity(label="Company", identity_column="supplier_name",
                     properties={"country": "country"}),
        ColumnEntity(label="Product", identity_column="product_name",
                     properties={"price": "unit_price", "sku": "sku"},
                     types={"unit_price": "FLOAT"},
                     description_template="{product_name} supplied by {supplier_name}"),
    ],
    relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
    text_columns=[TextColumn("notes", anchor_label="Product")],  # LLM pass per row with notes
    record_id_column="order_id",          # → chunk metadata["record_id"] (citations)
    metadata_columns=["region"],          # copied into chunk metadata
)

await pipeline.build_from_csv("orders.csv", mapping)
await pipeline.build_from_excel("orders.xlsx", mapping, sheet="Q1")  # pip install recon-graphrag[excel]
await pipeline.build_from_sql(conn, "SELECT * FROM orders", mapping)  # any DB-API connection
await pipeline.build_from_rows(list_of_dicts, mapping)                # bring your own rows
```

How it works:

- Each row becomes one `Chunk` (readable `"column: value"` text) linked
  `FROM_CHUNK` to the row's entities, so provenance and citations carry
  `record_id` just like text ingestion.
- Direct entities use label-scoped canonical keys (`"Product:Widget"`), so the
  same name under two labels stays two nodes. Each gets a deterministic
  description (from `description_template` or a compact property summary).
- Rows with a non-empty `TextColumn` get an LLM extraction pass over the row
  text, using `extraction_schema` (defaults to `mapping.to_schema()`).
  Extracted entities are linked from the anchor entity via `MENTIONS` (or the
  relationship you configure) and merge with same-label/same-name direct
  entities during finalize entity resolution.
- `mapping.resolution_context()` returns per-label typed columns to pass as
  `entity_resolution_context_properties` so hybrid LLM review sees structured
  fields as disambiguation context.
- A table with no `text_columns`, `entity_resolution_strategy="normalized"`
  (default), and `summarize_descriptions=False` ingests with **zero LLM calls**.

### `GraphBuilderPipeline` key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `embedder` | An embedder instance from `create_embedder()`. |
| `graph_name` | Graph scope for all created nodes and relationships. Defaults to `"entity-graph"`. |
| `graph_writer` | Optional `GraphWriter` implementation. When omitted, the pipeline uses the shared `CypherGraphWriter` for Neo4j/Memgraph replacement semantics. |
| `perform_entity_resolution` | When `True` (default), resolves duplicate entities after extraction. Set to `False` to skip resolution. |
| `summarize_descriptions` | When `True` (default), runs a mandatory LLM pass after entity resolution to collapse raw entity and relationship `descriptions` arrays into scalar `description` summaries before embedding. |
| `summarization_concurrency` | Maximum concurrent description summarization calls. Defaults to `5`. |
| `summarization_limit` | Batch size for fetching unsummarized entities or relationships. Defaults to `500`. |
| `embed_entities` | When `True` (default), embeds entity nodes after extraction and resolution. Set to `False` to skip embedding. |
| `embed_chunks` | When `True`, also embeds source chunks after the build so search can rank source passages by similarity to the query. Defaults to `False`: it costs one embedding call per chunk and only affects which passages survive the retrieval-side source limit. See [Ranking source chunks](06-search.md#ranking-source-chunks). |
| `fail_on_resolution_error` | When `True`, raises resolution errors instead of logging and continuing. Defaults to `False`. |
| `fail_on_embedding_error` | When `True`, the build fails instead of logging and continuing when the embedding stage cannot complete: after the embedding pass the pipeline re-checks the graph and raises if any entity in the configured graph has a missing or stale embedding (for example, because input/model changed or the provider/vector write failed). Defaults to `False`. |

Description summarization retries parser-invalid output (including blank or empty fenced
summaries) up to **3 total attempts**, with 1- and 2-second delays and the same
prompt. Provider and prompt errors are not retried by the summarizer. Each batch
persists successful summaries and failed statuses together before raising
`DescriptionSummarizationError` if any item failed; embedding and validation do
not run after that failure. Failed items retain their previous scalar description.
Rerunning skips already-successful items and retries failed items. A database
write failure stops the stage without a second write; batch outcomes are not
confirmed saved in that case.

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

The default `normalized` strategy also applies configured conflict properties. Matching names without a known conflict remain a heuristic, not proof of identity. Extracted IDs such as `p1` are local to a chunk; absent a trusted canonical/domain key, observations get document-and-chunk-scoped persistent IDs and are left for resolution to merge.

Conflict properties are conservative: missing values do not block a merge, but
two non-empty unequal values do. Blocked candidates are returned in
`review_groups` with `decision="blocked"` and are not sent to the LLM or
auto-merged.

### Custom prompts

Each LLM prompt the pipeline builds can be customized with a plain string.
The string becomes the user-facing instruction; the backend appends the
standard structural sections (allowed schema, rules, JSON format, source
text) so parsing keeps working.

```python
pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    extraction_prompt="You are a legal analyst. Extract parties and obligations from the text.",
    assessment_prompt="Review whether you missed any important legal entities.",
    continuation_prompt="Extract only the missing legal entities.",
    claim_prompt="Extract claims about the known entities.",
    entity_summary_prompt="Summarize this legal entity in one sentence.",
    relationship_summary_prompt="Summarize this legal relationship in one sentence.",
)
```

Every parameter defaults to `None`, which keeps the existing default prompt
unchanged. Each string may include optional placeholders to control where
the backend inserts content; a placeholder that is present is substituted
in place, and one that is absent is appended in the standard location.

| Parameter | Used for | Optional placeholders |
| --- | --- | --- |
| `extraction_prompt` | Initial graph extraction per chunk | `{text}` |
| `assessment_prompt` | Gleaning: "did you miss anything?" check | `{text}`, `{existing}` |
| `continuation_prompt` | Gleaning: extract only missed items | `{text}`, `{existing}` |
| `claim_prompt` | Claim extraction (`extract_claims=True`) | `{text}`, `{entity_ids}` |
| `entity_summary_prompt` | Description summarization of entities | `{descriptions}`, `{entity_name}`, `{entity_type}` |
| `relationship_summary_prompt` | Description summarization of relationships | `{descriptions}`, `{source}`, `{target}`, `{rel_type}` |

To include a literal brace, double it (`{{` / `}}`). For full control over
prompt structure, subclass `SchemaPromptBuilder` and pass it to
`LLMGraphExtractor` or `DescriptionSummarizer` directly. See
[custom_prompts.py](../examples/custom_prompts.py) for a runnable example
that prints the generated prompts.

### Gleaning

By default, the extractor makes a single LLM call per chunk. Set
`max_gleanings` on the build call to enable follow-up extraction loops that
catch missed entities:

```python
result = await pipeline.build_from_text(
    text,
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
per chunk. Most domains use the default `max_gleanings=1`; use `0` when
extraction cost matters more than completeness.

### Claims extraction

Set `extract_claims=True` on the build call to run a second LLM call per chunk
that extracts claims, assertions, and covariates about the entities already
found in that chunk:

```python
result = await pipeline.build_from_text(
    text,
    schema=schema,
    extract_claims=True,
)
```

Each claim has a subject entity, a claim type (e.g. `"role"`, `"status"`,
`"opinion"`, `"attribute"`), a description, optional status, optional object
entity, source text excerpt, and text unit ID. Claims are
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

### Document replacement and legacy graphs

The default writer replaces one document's contribution within its `graph_name`, including when called through `store.write_graph_document()`. Reusing the same document ID removes obsolete chunks, evidence and claims, replaces descriptions, and recomputes relationship chunk counts, weights and maximum strength from surviving document contributions. Shared evidence/descriptions survive; only unreferenced entities and relationships without surviving evidence are deleted.

Use stable `metadata["document_id"]` to identify updates explicitly. Otherwise the ID hashes the exact source/title, or the text when neither is supplied. Changing text without stable source/ID metadata creates a separate document; it cannot be recognized as a replacement automatically. Replacing the same ID in another graph is independent.

Description ownership (`description_sources`) and optional relationship strength ownership (`strength_sources`) are internal flat lists of JSON-encoded contribution records, compatible with both databases. Do not supply these fields. If legacy descriptions or strength have unknown ownership, the writer raises a migration diagnostic before writing. Identical text is not ownership evidence. Rebuild the affected documents/graph from their sources through the current writer; index creation alone cannot reconstruct ownership. Already mismerged identities also require rebuilding.

Changed description inputs invalidate summaries and vectors. Removing an observation rebuilds raw text from the surviving contributions rather than retaining an obsolete summary. The embedding pass checks both input and provider/model fingerprints; changing a model refreshes vectors even when text is unchanged. Stale in-flight embedding results cannot certify newer input. Low-level vector writes without fingerprints remain supported but clear freshness certification and need a normal embedding pass before strict completion.

Replacement of one document runs in **one database transaction**: `session.begin_transaction()` for Neo4j and Memgraph, committed after the last write and rolled back if any statement inside fails. A failure therefore leaves the document's previous state intact instead of a half-written one. The transaction covers only the document write: chunking, extraction, summarization and embedding stay outside it, and a failure later in the build still leaves the written document in place. Concurrent writes of different documents are not ordered; two writers can abort each other. Both a deadlock/serialization abort and a unique-constraint failure on the shared entity are retried inside the writer up to 3 times with a short backoff, reusing the same extracted document: the rollback leaves nothing to duplicate, and a constraint loser's retry matches the node its winner committed. An abort that survives all retries surfaces to the caller with the graph untouched. Nothing else is retried automatically. Stores that do not expose the transaction hook (custom store implementations and test doubles) keep the previous auto-commit behavior and its partial-write risk, and are not retried. `write_stats` reports what the database confirmed writing for each record kind, not the submitted input counts. A reference that cannot resolve — a relationship endpoint, evidence entity, or claim entity/chunk that exists neither in the document nor in the graph — raises `ValueError` before the first write, because those rows are only written when both endpoints exist. Custom writers must define their own guarantees.

Blocking database calls in pipeline writes/backfill/validation, description summarization, entity embedding, and resolver preflight/load/merge run through `asyncio.to_thread`. Cancellation does not terminate an in-flight database call or undo its writes.

### Tips

- Run `store.create_indexes()` before the first build and before starting concurrent writers; see [identity constraints](04-indexing.md#identity-constraints-and-existing-graphs).
- New document IDs extend the graph; an existing document ID replaces that document's contribution.
- Pass `metadata` to link extracted chunks and entities back to any source
  shape: documents, pages, database rows, tickets, API objects, or list items.

#### Extraction failures

Extraction (including gleaning and optional claims) stops the build with a
`RuntimeError`, for example:
`Extraction failed (document=<fingerprint> chunk=21): exception=APIStatusError status=429; ...`.
The document fingerprint and one-based chunk ordinal locate the failure without
exposing source metadata or text. Diagnostics include a fixed, recognized exception
type label (unknown custom types become `Exception`) and an HTTP error status only
when stored as an integer (400–599) on the exception. Provider messages, response bodies,
URLs and chained tracebacks are deliberately omitted; do not log source text or
credentials to investigate a failure.

Follow the fixed hint: check credentials/model access for authentication failures,
quota and `extraction_concurrency` for rate limits, connectivity/timeouts for
connection failures, and provider availability for server errors. Request errors
suggest checking model/configuration/limits. `OpenAICompatibleProviderError` means
the provider returned an error payload, even if the HTTP request succeeded; payload
contents are not exposed. JSON decoding failures suggest checking schema, prompt
templates and structured-output support. Other value/type/key errors can originate
in the provider adapter or extraction code, so check the provider response format
as well as schema/prompt compatibility. Hints are diagnostic categories, not proof
of a root cause. This change only surfaces errors; it does not repair a failed
extraction. Extraction errors are not retried
automatically. Fix the cause before rerunning; a document's extraction must finish
before it is written, but other documents in a multi-document build may already
have been written.

#### Embedding failures and recovery

Entity embedding is best-effort by default: provider or vector-write errors are
logged and the build continues. With `fail_on_embedding_error=True` (or when you
ask for strict completion directly), the build raises
`RuntimeError("Entity embedding incomplete ...")` if any entity in the configured
graph has a missing or stale embedding after the embedding pass. The check is
graph-wide and point-in-time: each entity must have a vector with current input
and provider/model fingerprints in `graph_name`. It does not prove global
coverage, atomicity under concurrent writers, or that every intended document
was ingested.

A failed strict build keeps earlier work: extraction results, graph writes,
entity resolution, and any successfully stored vectors are retained; only
missing/stale entities remain pending. It does not roll back partial writes,
and extraction is not replayed for you.

To recover, call `complete_embeddings()` on a pipeline pointing at the same
graph store, embedder, and `graph_name`. It embeds entities with missing or stale
vectors, verifies freshness/completion, and raises if anything remains:

```python
pipeline = GraphBuilderPipeline(
    graph_store=store, llm=llm, embedder=embedder,
    fail_on_embedding_error=True,
)
try:
    await pipeline.build_from_text(text, schema=schema)
except RuntimeError:
    # Later, once the provider is healthy: refresh pending vectors, no extraction
    await pipeline.complete_embeddings()
```

`complete_embeddings()` (and a strict build) does not resume interrupted
extraction or resolve/summarize anything new; and re-running an entire
`build_from_text(...)` after a failure re-extracts (and re-bills) documents, so
prefer targeted recovery when only embeddings were incomplete.

---

## CommunityPipeline

`CommunityPipeline` detects hierarchical communities using the backend store's Leiden implementation and generates a structured report for each community.

```python
from recon_graphrag import CommunityPipeline

community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,          # report vectors for semantic DRIFT primer search
    relationship_types=["DIRECTED", "ACTED_IN"],  # required
    graph_name="entity-graph",  # graph scope (default: "entity-graph")
    max_levels=3,               # hierarchy depth (default: 3)
    gamma=1.0,                  # Leiden resolution (default: 1.0)
    theta=0.01,                 # Leiden theta (default: 0.01)
    tolerance=1e-4,             # Leiden tolerance (default: 1e-4)
    relationship_weight_property="weight",  # numeric edge weight property
    random_seed=42,             # deterministic detection (default: 42)
    report_rubric=None,         # rating rubric for structured reports
    summarize_concurrency=1,    # concurrent report generations (default: 1)
    skip_existing=False,        # skip unchanged community reports
    max_context_tokens=8000,    # token budget for community context
    token_counter=None,         # token counter for context packing
    max_report_words=2000,      # soft word limit in the report prompt
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

### Report failures and diagnostics

If a report response has malformed JSON or fails report validation, generation
attempts one repair with one additional LLM call. If the repaired response still
fails, the report is recorded as failed; repair does not guarantee success.
Provider/authentication/network errors do not trigger this repair.

Report failures do not discard successful reports or stop other communities.
`build()` returns `status="degraded"` and a per-level entry in
`maintenance_warnings` when any report fails:

```python
{
    "code": "community_report_generation_failed",
    "level": 0,
    "failed": 1,
    "errors": [{
        "community_id": "community:1",
        "error_type": "ValueError",
        "message": "Community report generation or storage failed.",
    }],
}
```

By default, messages are fixed diagnostics and unrecognized exception types are
reported as `"Exception"`. Resume/preparation failures use the message
`"Community report preparation failed."`. Healthy builds keep an empty
`maintenance_warnings` list.

For **trusted/private debugging only**, opt in to original exception types and
messages:

```python
community = CommunityPipeline(store, llm, include_error_details=True)
result = await community.build()
```

These original messages are **not sanitized**: they may contain credentials,
provider details, or source content. Keep detailed results private; do not expose
them to untrusted users. `CommunitySummarizer` accepts the same option, and
`generate_all()` returns these error records in `BuildStats.errors`.

### `CommunityPipeline` key parameters

| Parameter | Description |
| --------- | --------- |
| `graph_store` | A `GraphStore` implementation such as `Neo4jGraphStore` or `MemgraphGraphStore`. |
| `llm` | An LLM instance from `create_llm()`. |
| `relationship_types` | Which relationship types form the community graph. Defaults to `None` (all types). |
| `graph_name` | Graph scope to detect communities within. Defaults to `"entity-graph"`. |
| `max_levels` | Maximum number of community hierarchy levels to detect. |
| `gamma` | Leiden resolution parameter. Higher values produce more communities. |
| `theta` | Leiden theta parameter. |
| `tolerance` | Leiden tolerance parameter. |
| `relationship_weight_property` | Name of the numeric relationship property to use as the Leiden edge weight, for example `"weight"`. Defaults to `"weight"`. Set to `None` to run Neo4j unweighted; Memgraph always uses `"weight"`. |
| `random_seed` | Random seed for deterministic Neo4j community detection. |
| `report_rubric` | Optional rating rubric for structured reports. |
| `report_prompt` | Optional custom report prompt string. A simple instruction (e.g. `"You are a security analyst. Identify risks in this community."`) gets the standard structured body (community context, reference allowlist, rubric, JSON format) appended. A full template must include all placeholders in `recon_graphrag.communities.reports.REQUIRED_REPORT_PLACEHOLDERS`; a partial template raises `ValueError` at construction. |
| `summarize_concurrency` | Maximum number of community reports to generate in parallel. Bounds concurrent reports, not total reports: every selected community is still processed and every report returned. Defaults to `1`. Increase for faster community builds when the LLM provider supports high throughput. |
| `skip_existing` | Skip community reports whose input fingerprint has not changed. |
| `max_context_tokens` | Maximum tokens for community context passed to the LLM. Degree-ranked context is greedily packed to fit this budget. Defaults to `8000`. Set to `None` to include all context without budgeting. |
| `token_counter` | Token counter for context packing. Defaults to `ApproximateTokenCounter` when `max_context_tokens` is set. |
| `max_report_words` | Soft word limit stated in the report prompt. Defaults to `2000`. Set to `None` to omit the instruction and let only the LLM's `max_tokens` bound report length. |
| `include_error_details` | Include original exception types/messages in failure diagnostics. Defaults to `False`; enable only for trusted/private debugging. |
| `embedder` | Embedder used to create report vectors after generation. |
| `embed_community_reports` | Generate report vectors during `build()`; defaults to `True`. |

The `embedder` parameter is required for DRIFT search, which uses community
report vectors for the primer phase. When `embed_community_reports=True`
(default), the pipeline embeds reports after generation. For existing graphs
without report embeddings, use `CommunityReportEmbedder` to backfill:

```python
from recon_graphrag import CommunityReportEmbedder

count = await CommunityReportEmbedder(
    store, embedder, "entity-graph"
).embed_reports()
```

### Choosing `relationship_types`

`relationship_types` determines which relationships are projected into the community detection graph. Choose relationship types that create meaningful connections between entities.

For example, in a movie domain:

```python
relationship_types=["DIRECTED", "ACTED_IN", "COMPOSED_MUSIC"]
```

would group people and movies connected through creative roles.

### Community levels

Recon-GraphRAG stores communities with `level=0` as the **coarsest / most
global** level. Higher numbers are progressively finer communities. This
matches Microsoft GraphRAG level semantics.

---

## Token usage

Token usage is tracked **by default**. Both `GraphBuilderPipeline` and
`CommunityPipeline` (and the [search classes](06-search.md)) wrap their `llm`
internally, open a run scope per call, and surface a `token_usage` summary in
their result — no setup required. One call in, one usage summary out, scoped to
exactly that call:

```python
from recon_graphrag import GraphBuilderPipeline, render_usage_table

pipeline = GraphBuilderPipeline(graph_store=store, llm=llm, embedder=embedder)
result = await pipeline.build_from_text(text=text, schema=schema)

print(render_usage_table(result["token_usage"]))
```

```text
token usage (run 9f3c...):
  stage                          calls    input   output     total  est
  construction.describe_entities     4    3,100      800     3,900    0
  construction.extract              30   21,400    5,200    26,600    0
  TOTAL                             34   24,500    6,000    30,500    0
```

Pass `track_token_usage=False` to any pipeline or search constructor to opt out
entirely.

Stages follow an `area.operation` taxonomy: `construction.*` (extract,
gleaning, descriptions, entity resolution, schema analysis), `community.*`
(report, report_repair), plus `retrieval.*` and `drift.*` from the
[search classes](06-search.md). When a provider omits usage, tokens are
estimated with tiktoken and counted in the `est` column.

The library does not retain usage history — the default ledger forgets a run
the moment its result is returned. To persist usage, custom-scope it per web
request, attach a `JsonlUsageSink`, or track standalone building blocks, see
[Token usage observability](09-workflows.md#token-usage-observability) in the
workflows guide.

---

## Next steps

- Explore the composable building blocks in [Workflows](09-workflows.md).
- Define your domain model in [Schema](03-schema.md).
- Create required indexes in [Indexing](04-indexing.md).
- Search the graph in [Search](06-search.md).

### Assembly identity and conflict reports

Before writing, custom non-name schema `NodeType.identity_property` values identify
entities; different trusted values remain separate in memory, even if explicit
canonical keys match. Explicit canonical keys otherwise retain their historical
trusted meaning. A canonical key shared across labels is rejected with `ValueError`
before that document is written if any occurrence is trusted; trusted keys are
never renamed to resolve these collisions. Collisions involving only untrusted
keys remain label-disambiguated. Without either, LLM local IDs remain chunk-local
observations.
Confirmed identities union only schema `LIST` properties; conflicting populated
scalars remain first-wins with source-bearing conflict observations in
`GraphDocument.assembly_review_groups` and
`result["extraction"]["assembly_review_groups"]`. These reports are not DB properties.
For name-only structured candidates, provisional splitting and its limitations are
documented in [structured ingestion](11-structured-data.md#in-memory-conflicts-before-writing).
Default normalized database resolution can undo in-memory separation: configure
conflict properties, disable resolution, or inspect with `finalize=False` on
row/document builds. No database resolver policy changes are implied.
