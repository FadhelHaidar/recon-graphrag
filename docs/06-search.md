# Search

Recon-GraphRAG provides three search modes through dedicated search classes:
**Local**, **Global**, and **DRIFT**. Use this page for both the public API and
the mechanics behind each mode.

## Search Modes

| Mode | Strategy | Best for |
| --- | --- | --- |
| **Local** | Entity subgraph traversal | Specific questions about named entities |
| **Global** | Community report map-reduce | Broad overviews and thematic landscapes |
| **DRIFT** | Entity + community hybrid | Questions needing detail plus surrounding context |

Each mode has its own search class:

```python
from recon_graphrag import (
    LocalSearchRetriever,
    GlobalSearchRetriever,
    DriftSearchRetriever,
    DriftSearchConfig,
)

# Local search — entity-centric subgraph traversal
local_search = LocalSearchRetriever(
    store, llm, embedder,
    graph_name="entity-graph",        # graph scope (default: "entity-graph")
    use_mixed_context=True,           # enable mixed-context (default: True)
    top_k_relationships=10,           # max relationships per entity (default: 10)
    top_k_sources=3,                  # max source chunks per entity (default: 3)
)
result = await local_search.search("Your question", top_k=10)

# Global search — community report map-reduce
global_search = GlobalSearchRetriever(
    store, llm,
    graph_name="entity-graph",
    token_counter=None,               # optional token counter for batching
    map_budget_tokens=12000,          # map budget (default: 12000)
    reduce_budget_tokens=12000,       # reduce budget (default: 12000)
)
result = await global_search.search("Your question", community_level="coarsest")

# DRIFT search — entity + community hybrid
drift_search = DriftSearchRetriever(
    store, llm, embedder,
    graph_name="entity-graph",
    config=DriftSearchConfig(primer_top_k=3),
)
result = await drift_search.search("Your question", top_k=10)
```

Each class owns its own constructor parameters, prompts, and search method.

---

## Local Search

Local search answers specific questions by retrieving relevant entities, their
neighbors, and related text chunks.

```python
result = await local_search.search(
    "Who directed Inception?",
    top_k=10,
)
```

| Parameter | Description |
| --- | --- |
| `top_k` | Number of top entities to retrieve. |
| `effective_search_ratio` | Over-fetch multiplier before post-filtering. |
| `ranker` | Hybrid ranker: `"naive"` or `"linear"`. |
| `alpha` | Required for the `"linear"` ranker. |
| `query_params` | Optional dict forwarded to the underlying hybrid entity retriever. |
| `community_level` | Community level for mixed-context reports. Only used when `use_mixed_context=True`. |
| `token_budget` | Total token budget for mixed context. Only used when `use_mixed_context=True`. Defaults to `12000`. |

With `use_mixed_context=True`, `LocalSearchRetriever` uses `MixedContextBuilder`.
It collects seed entities, relationships, text units (chunks), community reports, and claims.
It ranks these candidates and packs them into one context string within the token budget. This provides richer context
than the default entity-subgraph-only mode.

Within the community-report share of that budget, reports rank by how many of
the retrieved seed entities belong to the community, then by stored report
rating, then by community ID. Overlap with the seed entities is a heuristic for
query fit, not a choice of hierarchy level: local search still reads reports at
one requested level (default `"coarsest"`), so this ranking changes which of the
already-eligible reports gets the limited report budget first.

Local search returns citations when retrieved entities have source chunks.
By default, citation metadata is returned after synthesis but is not shown to
the LLM. Opt in when the answer should see source identifiers while being
written:

```python
result = await local_search.search(
    "Who directed Inception?",
    synthesize_citation_metadata=True,
    synthesis_metadata_keys=["record_id", "collection"],
)
```

---

## Global Search

Global search answers broad questions from community reports. Omit
`community_level` to select reports dynamically across the hierarchy; pass a
level to map every eligible report at that level:

```text
select reports (dynamic when community_level is omitted, static per level otherwise)
  -> read all successful reports at that level (or up to max_reports)
  -> shuffle reports with random_seed
  -> pack reports into map batches using map_budget_tokens,
     stopping at max_map_calls batches or max_map_tokens input tokens
  -> run one LLM map call per batch, concurrently
  -> filter map answers with helpfulness == 0
  -> sort remaining partial answers by helpfulness DESC
  -> pack partial answers using reduce_budget_tokens
  -> run one LLM reduce call for the final answer
  -> resolve citations from map references and used community report references
```

```python
result = await global_search.search(
    "What are the main themes in this dataset?",
    community_level="coarsest",
    random_seed=42,
)
```

**Example walkthrough:**

Given two community reports at level 0:

```text
0: "Christopher Nolan directed Oppenheimer starring Cillian Murphy..."
1: "Hans Zimmer composed scores for Interstellar, Inception, and Dune..."

Query: "What are the main themes in this dataset?"
  ↓ resolve level (coarsest = 0)
  ↓ read all reports at level 0
  ↓ shuffle with seed=42
  ↓ pack into batches (map_budget_tokens=12000)
Batch 0: [0, 1]
  ↓ LLM map call
Map output: { answer: "Themes include human ambition, time, space...", helpfulness: 75, report_ids: [...] }
  ↓ filter (helpfulness > 0) and sort
  ↓ LLM reduce call
Final answer: "The main themes are human ambition, exploration of time and space..."
  ↓ resolve citations from report_json references
Citations: [chunk:0, chunk:1, chunk:2, chunk:3, chunk:4]
```

Global search does not touch entity nodes or embeddings at query time. It reads
stored community reports and synthesizes a corpus-level answer.

| Parameter | Description |
| --- | --- |
| `community_level` | Which community level to search. Use `"coarsest"` when you do not know the numeric levels. |
| `random_seed` | Seed for reproducible report shuffling. Defaults to `42`. |
| `map_budget_tokens` | Constructor option on `GlobalSearchRetriever`; maximum report text packed into one map prompt. |
| `reduce_budget_tokens` | Constructor option on `GlobalSearchRetriever`; maximum partial-answer text packed into the final reduce prompt. |
| `map_concurrency` | Constructor option on `GlobalSearchRetriever`; max concurrent map calls. Defaults to `5`. |
| `max_map_calls` | Constructor option on `GlobalSearchRetriever`; max total map batches, and therefore max map calls. Defaults to `None` (unlimited). Batches beyond the cap are never built. |
| `max_map_tokens` | Constructor option on `GlobalSearchRetriever`; sum of complete formatted map-prompt token counts, including query/template overhead and separators, using `token_counter`. Stop before the first batch that does not fit (possibly sending no map calls). Defaults to `None` (unlimited). This is not a billed-token or output-token cap. |
| `max_reports` | Constructor option on `GlobalSearchRetriever`; max reports read from the graph for the query. Defaults to `None` (every report at the level). |

Global search reads stored `Community.report_text` / `Community.summary`
values directly from the graph. It does not embed the user query, use vector
similarity, or traverse raw entity nodes at query time.

After the reduce phase, global search also reads `Community.report_json` for the
community reports that the map phase used. It extracts validated entity, relationship, and claim references from `findings[*].references`.
If `report_json` has no usable references, it parses `[refs: ...]` blocks in `report_text` instead.
It resolves these references to chunk-level citations. Map-returned
`report_ids` are validated against the batch's actual report IDs; invalid IDs are
ignored. If a map response omits `report_ids`, the batch's report IDs are used as
a fallback. Citation resolution is non-fatal: if report reading or citation
resolution fails, the answer is still returned with partial or empty citations
and the failure is recorded in `metadata["source_resolution_errors"]`.

One map batch means one map LLM call, followed by one reduce LLM call:

```text
3 reports, each ~2k tokens
map_budget_tokens=12000

batch 0 = reports [c2, c1, c3]

LLM calls:
  map(batch 0)
  reduce(partial answers)
```

Multiple map batches means multiple map LLM calls, usually concurrent, followed
by one reduce LLM call:

```text
10 reports, each ~2k tokens
map_budget_tokens=6000

batch 0 = reports [c8, c1, c4]
batch 1 = reports [c9, c2, c6]
batch 2 = reports [c5, c3, c7]
batch 3 = reports [c10]

LLM calls:
  map(batch 0), map(batch 1), map(batch 2), map(batch 3)
  reduce(sorted partial answers)
```

`community_level="coarsest"` resolves to level `0`. `"finest"` resolves to
the highest level currently stored in the graph. Any explicit level keeps the
static, single-level behavior described above.

Global search skips failed or empty community reports.

### Dynamic community selection (default)

Omitting `community_level`, or passing `None`, selects reports per query
instead of mapping one whole level:

```text
read eligible level-0 (coarsest) reports
  -> rate each report 0-5 for relevance to the query (LLM)
  -> rating 1-5 keeps the report and queues its children
  -> a kept child replaces its parent
  -> rating 0 stops descent into that branch
  -> if nothing was selected, rate the whole next level (up to level 2)
  -> map/reduce the selected reports
```

Each visited report costs one additional LLM rating call, billed under the
`retrieval.global_selection` stage and counted separately from map/reduce.
`max_reports`, `max_map_calls`, and `max_map_tokens` still bound the map stage
only, not the rating calls, so a broad question can rate much of the hierarchy.
To avoid rating calls, pass an explicit level. Map limits do not cap selection cost.

Selected reports can come from different levels; citations and graph evidence
use each report's real level. The metadata carries `selection_mode`,
`levels_visited`, `selected_report_keys`, `reports_rated`,
`fallback_levels_attempted`, and `rating_errors` alongside the existing map and
citation counters.
Because selection depends on model ratings, a false negative can prune a useful
branch; no cost or quality improvement is guaranteed.

#### How a report is rated

Rating is a single LLM judgment, not a computed score. The fixed prompt asks four
evidence questions before the model answers:

- **Direct support:** does the report state facts answering any requested aspect?
- **Indirect support:** if not, does it state a concrete relationship, event,
  mechanism, or background fact useful for answering the question?
- **Broad synthesis:** for a question about themes or patterns, does the report
  contribute a relevant aspect even if it does not answer the whole question?
- **Grounding:** is that connection supported by the report itself, rather than by
  shared names or words, outside knowledge, or content assumed to exist in child
  communities?

The model then reports the highest anchor the evidence supports: `0` no supported
connection (unrelated content, or superficial word/name overlap only), `1` limited
but explicit indirect connection, `2` concrete relevant background, `3` direct
evidence answering part of the question, `4` covering most requested aspects, `5`
sufficient to address all requested aspects. When two positive anchors both seem to
fit, the lower one is used.

Only the `0` versus positive split changes behavior: `1`-`5` all keep the report and
queue its children, and the rating does not weight map helpfulness, the reduce
ranking, or the token budget. The rating reason is validated as a string and is not
exposed in metadata or logs.

Prompt construction is deterministic for a given question and report text; model
output is not. The SDK does not cache ratings, retry with a rewritten prompt, or pin
temperature, so the same question can still rate the same report differently between
runs. Fix the model and generation settings in the application, and use temperature
`0` where the provider supports it, to reduce that variation without eliminating it.
Response caching and invalidation belong in the application, not the SDK.
A provider error or an unparsable rating keeps the report with rating `1` and
increments `rating_errors`, so a failure is never read as a deliberate `0`.

### Scored point context (`map_strategy="scored_points"`)

By default global search joins whole map answers and runs a reduce call. Pass `map_strategy="scored_points"` to return ranked evidence points for an agent instead.
Each map call emits a strict list of `{description, score, report_ids}` points.
The retriever packs the best points into a bounded context string without a reduce call.

```python
retriever = GlobalSearchRetriever(
    graph_store=store,
    llm=llm,  # must support structured output
    map_strategy="scored_points",
)
result = await retriever.search(
    "What are the main themes in this dataset?",
    community_level="coarsest",
    synthesize_response=False,  # required in point mode
)
```

Point mode is context-only. `synthesize_response=True` and an LLM adapter whose
`supports_structured_output` is `False` are both rejected before any graph read
or LLM call. Adapters without that attribute are allowed through.

* Points are sorted by score DESC, then batch ID, then position in the batch.
  Cached map responses use an integer `score`; points scoring `0` or less are
dropped, and a batch whose points are all dropped contributes no evidence.
* The context is one JSON envelope (`query`, `mode`, `answer`, `context`,
  `citations`) charged against `reduce_budget_tokens`, so a fixed budget holds
  across models. Points that do not fit are skipped whole (never sliced). If
  even the bare envelope plus the omission notice cannot fit, the search raises
  `ValueError`. When only the query field alone overflows, the envelope drops
  `query` and keeps the points.
* Skipped points add a fixed notice to the context, so a truncated context is
  visible to the caller.
* `report_ids` returned by a map batch are validated against that batch and
  out-of-batch IDs are discarded; citations still resolve from the community
  reports that contributed evidence.

Metadata adds `map_strategy`, `context_budget_tokens`, `context_tokens`,
`context_parts_available`, `context_parts_kept`, `context_parts_dropped`,
`context_warnings`, and `context_failed`. A failed map batch in point mode is
recorded in `map_failures` with a fixed reason (`invalid_schema`, `truncated`,
`refused`, `provider_error`) and logged once at WARNING with no provider text.
If every batch fails, `context_failed` is `True` and the result carries empty
`answer` and `context` rather than a synthesized guess.

Global citations resolve explicit map-phase references to entities,
relationships, or claims, plus validated references from the used community
reports' structured `report_json` or rendered `[refs: ...]` text.

---

## DRIFT Search

DRIFT search combines local entity retrieval with community context for
questions that need both detail and big-picture framing. It uses an iterative
traversal that begins with semantic community-report retrieval, generates
follow-up questions, performs local subgraph searches for each follow-up,
and reduces all gathered evidence into a final answer.

```python
result = await drift_search.search(
    "Explain the relationship between Christopher Nolan and his frequent collaborators.",
    top_k=10,
    community_level="coarsest",
)
```

**Example walkthrough:**

```text
Query: "How does Hans Zimmer connect Inception to Dune?"
  ↓ Phase 1: Primer
  ↓ embed query → vector search community reports (primer_top_k=3)
Reports: [1 ("Hans Zimmer composed scores for...")]
  ↓ LLM generates initial answer + follow-ups
Primer: { answer: "Zimmer composed for both...", score: 65,
          follow_ups: ["What films did Zimmer score for Nolan?",
                       "How did Zimmer's style evolve between Inception and Dune?"] }
  ↓ Phase 2: Traversal (2 rounds, up to 2 selected questions each)
  ↓ round 1 selects action:1 and action:2 from the shuffled pending pool
  ↓ action:1 — "What films did Zimmer score for Nolan?"
    → local entity retrieval (top_k=10): [Hans_Zimmer, Christopher_Nolan, ...]
    → LLM scores 80, no new questions
  ↓ action:2 — "How did Zimmer's style evolve between Inception and Dune?"
    → local entity retrieval: [Hans_Zimmer, Inception, Dune, ...]
    → LLM scores 45, 1 follow-up queued for the next round
  ↓ round 2 selects action:3 (the queued follow-up) from the pool
  ↓ action:3 — follow-up from action:2
    → local entity retrieval → LLM scores 30
  ↓ Phase 3: Reduction
  ↓ sort completed actions by score: [action:1 (80), action:2 (45), action:3 (30)]
  ↓ pack into reduce_budget_tokens=12000
  ↓ LLM synthesizes final answer
  ↓ resolve citations from local chunks + primer report references
```

Total LLM calls: 1 (primer) + 3 (actions) + 1 (reduce) = 5. Two rounds with
two selected questions each allow at most four actions; generation stopped
after three because no further questions remained.

| Parameter | Description |
| --- | --- |
| `top_k` | Number of entities to retrieve. |
| `community_level` | Which community level to use. |
| `query_params` | Optional dict forwarded to the underlying hybrid entity retriever. |
| `conversation_history` | Optional conversation history string injected into prompts. |

### DriftSearchConfig

Control iterative DRIFT behavior with `DriftSearchConfig`:

```python
from recon_graphrag import DriftSearchConfig

config = DriftSearchConfig(
    primer_top_k=3,           # top community reports for primer phase
    actions_per_round=3,      # questions selected from the pool per round
    max_rounds=3,             # traversal rounds
    max_llm_calls=20,         # hard cap on total LLM calls per search
    action_concurrency=3,     # max concurrent action evaluations
    community_level="coarsest",  # default community level
    reduce_budget_tokens=12000,  # token budget for final reduction
)
```

Traversal keeps one shared pool of pending follow-up questions. Each round shuffles the pool with a seed derived from the query.
It selects up to `actions_per_round` questions and evaluates them with `action_concurrency` parallelism.
It adds their follow-ups to the pool for a later round. Unselected
questions stay available, and a question generated in a round is never run in
that same round. `DriftAction.depth` remains the parent-chain depth of a
question, not the round number, so later rounds can select older shallow
questions.

| Field | Default | Description |
| --- | --- | --- |
| `primer_top_k` | `3` | Number of community reports retrieved by vector similarity in the primer phase. |
| `actions_per_round` | `3` | Maximum questions selected from the pending pool per round, independent of `action_concurrency`. `0` disables traversal. |
| `max_rounds` | `3` | Maximum traversal rounds. `0` disables traversal. Work still pending after the last round stops with `max_rounds_reached`. |
| `max_llm_calls` | `20` | Hard cap on total LLM calls across all phases (primer, actions, reduction). Prevents runaway costs. |
| `action_concurrency` | `3` | Maximum number of actions evaluated concurrently during traversal. |
| `community_level` | `"coarsest"` | Default community level for primer report search. Overridden by the `community_level` search parameter if provided. |
| `reduce_budget_tokens` | `12000` | Token budget for packing scored action answers into the final reduction prompt. |
| `use_hyde` | `True` | Enable HyDE (Hypothetical Document Embedding) for primer. Generates a hypothetical answer from a random report template, re-embeds, and re-searches for better report retrieval. |
| `primer_folds` | `1` | Number of folds to split primer reports into for parallel LLM decomposition. When > 1, reports are split into N folds, processed in parallel, and follow-ups are merged. |
| `action_use_mixed_context` | `True` | When `True`, each DRIFT action builds mixed context (entities + community reports + claims) instead of entity-only context. |
| `action_mixed_context_tokens` | `12000` | Token budget for mixed context in each DRIFT action. Only used when `action_use_mixed_context=True`. |

#### Migrating from `max_depth` / `max_followups` / `min_expand_score`

The traversal config is a breaking change, not a rename:

```python
# Old: per-parent branching, depth limit, and score pruning
DriftSearchConfig(max_depth=3, max_followups=3, min_expand_score=20)

# New: up to three selected questions per round, for three rounds
DriftSearchConfig(max_rounds=3, actions_per_round=3)
```

`max_depth`, `max_followups`, and `min_expand_score` are rejected as unknown
keywords. These settings are not cost- or answer-equivalent: equal numbers mean
different work, and low-scoring actions now contribute follow-up questions.
`min_expand_score` is removed rather than renamed. Positional construction is
also affected because the fourth field is gone, so prefer keyword arguments and
re-check any call site that passed three or more arguments positionally.

DRIFT returns citations for retrieved local source chunks and for references in
the selected primer reports.
Like local search, DRIFT accepts `synthesize_citation_metadata=True` and optional
`synthesis_metadata_keys=[...]` to include compact citation metadata in the
answer synthesis context.

---

## Agent Context Mode (`synthesize_response`)

All search modes accept `synthesize_response=False` to skip final LLM answer
synthesis and return the raw retrieved context and citations instead. This is
useful when an outer agent or orchestration layer wants to consume the context
directly.

```python
result = await local_search.search(
    "What evidence is relevant?",
    synthesize_response=False,
)
agent_context = result.context
sources = result.citations
```

When `synthesize_response=False`:

- `result.answer` is `""` (empty string, not a synthesized answer).
- `result.context` contains the full retrieved context that would normally be
  sent to the LLM.
- `result.citations` are resolved as usual.
- `result.metadata` includes `"synthesize_response": False` and
  `"response_synthesis_skipped": True`.

Mode-specific behavior:

| Mode | What still runs | What is skipped |
| --- | --- | --- |
| **Local** | Entity retrieval, context formatting, citation resolution | LLM answer generation |
| **DRIFT** | Entity retrieval, community expansion, context formatting, citation resolution | LLM answer generation |
| **Global** | Level resolution, report reading, batching, map phase, citation resolution | Final reduce synthesis |

> **Note:** For global search, `synthesize_response=False` skips the final
> reduce LLM call but still runs map-phase LLM calls for relevance scoring and
> reference extraction. This is by design — map calls provide the scored
> partial answers that form the context.

> **Note:** Do not confuse `synthesize_response` with
> `synthesize_citation_metadata`. The former controls whether the LLM generates
> a final answer. The latter controls whether citation metadata is included in
> the LLM prompt context during answer synthesis.

---

## Citations And Sources

`SearchResult` includes structured citation fields in addition to answer text:

```python
result = await local_search.search("Who directed Inception?")

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
synthesis. For Local and DRIFT search, set `synthesize_citation_metadata=True` to
also include compact citation metadata in the LLM context. Use
`synthesis_metadata_keys` to keep that prompt context small.

Current citation behavior:

- **Local** resolves retrieved entity evidence from `source_chunk_ids` to
  document/chunk citations.
- **DRIFT** includes the same local evidence citations alongside community
  context.
- **Global** resolves validated map references and references from used
  community reports (`report_json` first, then `[refs: ...]` fallback) with
  `target_type` of `entity`, `relationship`, or `claim`.

Citation resolution is graph-scoped. A query on one `graph_name` will not resolve
chunks, claims, or entities from another graph.

---

## Structured Graph Evidence

Every retriever accepts `include_graph_evidence=True`, which fills
`result.graph` with the entity nodes and directed relationships that were
*included in synthesis* alongside the answer, context, and citations:

```python
result = await local_search.search("Who leads Acme?", include_graph_evidence=True)

for node in result.graph["nodes"]:
    print(node["id"], node["name"], node["labels"])
for edge in result.graph["edges"]:
    print(edge["source"], f"-[{edge['type']}]->", edge["target"], edge["weight"])
```

| Key | Meaning |
| --- | --- |
| `nodes` | Deduplicated vertices: `id`, `name`, `labels`, `description`, `origin`. Sorted by `id`. |
| `edges` | Deduplicated directed edges: `source`, `target`, `type`, `description`, `weight`, `key`, `origin`. Both endpoints are always present in `nodes`. |
| `provenance` | `graph_name`, `mode`, the contributing community reports (`id`, `level`, plus mode-specific fields), and any `warnings`. |
| `truncation` | `applied`, `dropped_nodes`, `dropped_edges`, `limits`, and the `reasons` behind every cut. |

The field defaults to `{}`; with the flag off, no extra reads happen. `origin`
is `retrieved` (direct graph fact), `community_member` (expanded report
member), or `edge_endpoint` (node only present because an edge points at it).

What gets included, per mode:

- **Local** — the entities and relationships that survived mixed-context token
  packing, or for `use_mixed_context=False` the entity plus the
  `top_k_relationships` slice the context renders. Community reports included in
  the context are expanded into their member entities and edges.
- **DRIFT** — each action keeps its own evidence on `DriftAction.graph`; the
  result merges only actions packed into the final reduction
  (`provenance.retained_actions`, with `explored_actions` listed but excluded).
  The primer's selected reports are expanded only when the primer reached
  reduction. The local-style fallback path uses the local selection.
- **Global** — only partial answers that fit the actual reduce-stage token
  packing are considered; the community reports behind them are expanded at the
  searched level. Reports that were merely read or mapped are not expanded.

`Community` nodes are never returned as vertices: a contributing report shows up
as its member entities and relationships, with the report id and level kept in
`provenance.communities`. Community expansion costs one bounded query per report
(at most `max_reports`), so it only runs when you opt in.

Reported totals are only those actually known. `member_limit` (25), `edge_limit`
(50) per report, `max_reports` (5), `max_nodes` (200) and `max_edges` (500) per
result are all deterministic; when a cap bites, `truncation.reasons` records
what was kept and dropped. Selection is degree-ranked and stable across runs.

This payload describes evidence included in synthesis. It does not verify
sentence-level grounding, prove answer correctness, or establish capacity.

---

## How Sources Work

Each mode builds `result.citations` and `result.sources` through a different
path. The rest of this section walks through the same example corpus in three
phases so you can see exactly where source metadata comes from and where it
gets lost.

---

### Phase 0: What the Graph Contains

After running `GraphBuilderPipeline` and `CommunityPipeline` on a small movie
corpus, the graph contains nodes and edges like these. IDs and excerpts below
are illustrative; actual IDs depend on extraction and backend results:

**Documents and chunks**

One `Document` node, "smoke-source", owns five `Chunk` nodes:

```text
Document "smoke-source"
  Chunk "chunk:0" (page 1): Christopher Nolan directed Interstellar...
  Chunk "chunk:1" (page 2): Leonardo DiCaprio starred in Inception...
  Chunk "chunk:2" (page 3): The Dark Knight Rises was directed by Christopher Nolan...
  Chunk "chunk:3" (page 4): Denis Villeneuve directed Dune...
  Chunk "chunk:4" (page 5): Oppenheimer was directed by Christopher Nolan...
```

Every chunk connects to its document with `(Chunk)-[:PART_OF]->(Document)`.

**Entities and chunk links**

Each extracted `__Entity__` node has `(Chunk)-[:FROM_CHUNK]->(__Entity__)` edges
back to the chunks it was found in:

```text
Christopher_Nolan  <- FROM_CHUNK <- chunk:0, chunk:1, chunk:2, chunk:4
Hans_Zimmer        <- FROM_CHUNK <- chunk:0, chunk:1, chunk:3
Cillian_Murphy     <- FROM_CHUNK <- chunk:2, chunk:4
Inception          <- FROM_CHUNK <- chunk:1
Interstellar       <- FROM_CHUNK <- chunk:0
Dune               <- FROM_CHUNK <- chunk:3
Oppenheimer        <- FROM_CHUNK <- chunk:4
```

**Relationships**

Relationships also carry chunk provenance:

```text
Christopher_Nolan -[DIRECTED]-> Inception          source_chunk_ids: [chunk:1]
Christopher_Nolan -[DIRECTED]-> Oppenheimer        source_chunk_ids: [chunk:4]
Hans_Zimmer       -[COMPOSED_FOR]-> Dune           source_chunk_ids: [chunk:3]
Cillian_Murphy    -[STARRED_IN]-> Oppenheimer      source_chunk_ids: [chunk:4]
```

**Communities and reports**

Community detection groups entities into clusters. Each reported `Community`
node has `report_text` plus `report_json` containing findings with validated
entity references:

```text
Community "0"
  <- IN_COMMUNITY <- Christopher_Nolan, Cillian_Murphy, Oppenheimer
  report_text: "Christopher Nolan directed Oppenheimer starring Cillian Murphy..."
  report_json: {
    "findings": [
      {
        "description": "Christopher Nolan directed Oppenheimer starring Cillian Murphy",
        "references": [
          {"target_id": "Christopher_Nolan", "target_type": "entity"},
          {"target_id": "Cillian_Murphy", "target_type": "entity"}
        ]
      }
    ]
  }

Community "1"
  <- IN_COMMUNITY <- Hans_Zimmer, Inception, Dune, Interstellar
  report_text: "Hans Zimmer composed scores for science-fiction films..."
  report_json: {
    "findings": [
      {
        "description": "Hans Zimmer composed scores for Interstellar, Inception, and Dune",
        "references": [
          {"target_id": "Hans_Zimmer", "target_type": "entity"},
          {"target_id": "Christopher_Nolan", "target_type": "entity"},
          {"target_id": "Denis_Villeneuve", "target_type": "entity"}
        ]
      }
    ]
  }
```

The `report_json` references are validated against the real entity allowlist
when the report is generated. Global search reads `report_text` for synthesis
and `report_json` for evidence references.

---

### Phase 1: Local Search

**Question:** `"Which movies were directed by Christopher Nolan and feature Cillian Murphy?"`

**Step 1 — Find entities.** The query is embedded and searched against the
entity vector and full-text indexes. Top matches are `Christopher_Nolan` and
`Cillian_Murphy`.

**Step 2 — Traverse the graph.** Starting from those entities, the retriever
follows one-hop relationships:

```text
Christopher_Nolan -[DIRECTED]-> Inception, Interstellar, The_Dark_Knight_Rises, Oppenheimer
Cillian_Murphy    -[STARRED_IN]-> Oppenheimer, The_Dark_Knight_Rises
```

**Step 3 — Collect chunk IDs.** From every traversed entity and relationship,
the retriever collects `source_chunk_ids`:

```text
[chunk:0, chunk:1, chunk:2, chunk:4]
```

These IDs come from stored edges, not from the LLM.

**Step 4 — Resolve citations.** `resolve_chunk_citations` looks up each chunk
ID, finds its document, page, and excerpt, and returns:

```text
Citation(document_id="smoke-source", chunk_id="chunk:0", page_start=1, page_end=1, excerpt="Christopher Nolan directed Interstellar...")
Citation(document_id="smoke-source", chunk_id="chunk:1", page_start=2, page_end=2, excerpt="Leonardo DiCaprio starred in Inception...")
Citation(document_id="smoke-source", chunk_id="chunk:2", page_start=3, page_end=3, excerpt="The Dark Knight Rises was directed by...")
Citation(document_id="smoke-source", chunk_id="chunk:4", page_start=5, page_end=5, excerpt="Oppenheimer was directed by Christopher Nolan...")
```

**Step 5 — Generate answer.** The LLM receives the entity descriptions,
relationships, and chunk excerpts, then writes the final answer.

**Result:** `citations` has four items, `sources` groups them under the single
document "smoke-source". Each chunk ID came from a stored graph edge.
These links identify evidence; they do not prove that every answer claim is correct.

---

### Phase 2: DRIFT Search

**Question:** `"How does Hans Zimmer connect Inception to Dune?"`

**Step 1 — Select primer reports.** The query embedding searches community
report vectors at the requested hierarchy level.

**Step 2 — Plan focused actions.** The primer LLM proposes follow-up questions
grounded in the selected report text.

**Step 3 — Retrieve local evidence.** Each action runs hybrid entity retrieval,
collects typed relationship context, and resolves source chunk IDs.

**Step 4 — Score and expand.** Useful actions may propose bounded follow-ups;
the configured round and LLM-call limits cap the traversal.

**Step 5 — Resolve citations and reduce.** The highest-scoring action answers
are packed into the reduction budget. Citations combine local source chunks
with validated references from the selected primer reports.

**Result:** `citations` and `sources` include the evidence actually retained for
the final reduction.

---

### Phase 3: Global Search

**Question:** `"What are the most common themes across science-fiction films?"`

Global search reads community reports for map-reduce synthesis. Citation
resolution can also read entities, relationships, claims, and source chunks.

#### Step 1 — Read reports at the selected level

The retriever runs a query that returns, for each community:

```text
id: "0"
level: 0
summary: "Christopher Nolan directed Oppenheimer starring Cillian Murphy..."

id: "1"
level: 0
summary: "Hans Zimmer composed scores for science-fiction films including Interstellar, Inception, and Dune..."
```

This initial read does not return `report_json`. Citation resolution later
reads structured references from the reports used by the map phase.

#### Step 2 — Shuffle once

The reports are shuffled with a reproducible random seed. If there were six
reports instead of two, the shuffle might reorder them like this:

```text
Before shuffle (database order): 0, 1, 2, 3, 4, 5
After shuffle (seed=42):         3, 5, 1, 4, 0, 2
```

The shuffle happens exactly once per search call. Its purpose is to remove
positional bias so the LLM does not always overweight the first report.

#### Step 3 — Pack into batches

The retriever counts tokens and packs reports into batches that fit within
`map_budget_tokens`. With two short reports and a 12000-token budget, everything
fits in one batch. With six reports of ~2000 tokens each and a tight 8000-token
budget, the packing might produce:

```text
Batch 0: 3, 5, 1, 4
Batch 1: 0, 2
```

If one report exceeds the available budget, the retriever truncates its formatted text to fit. If `max_map_calls` is set and there are more
batches than the limit, later batches are dropped and never processed.

#### Step 4 — Map phase: one LLM call per batch

Each batch is sent to the LLM. The LLM is asked to:

- Write a partial answer using only the reports in the batch.
- Rate how helpful those reports are for the question (0-100).
- List report IDs it used.
- List entity/claim/relationship references it found in the text.

**Input to the map LLM (Batch 0):**

```text
Question: What are the most common themes across science-fiction films?

Community Reports:

Report 1:
Hans Zimmer composed scores for science-fiction films including Interstellar,
Inception, and Dune. These films share themes of time, space, and human
ambition. Denis Villeneuve directed Dune while Nolan directed Interstellar and
Inception.

Report 0:
Christopher Nolan directed Oppenheimer (2023) starring Cillian Murphy. The film
won Best Picture. Nolan is known for complex narratives and practical effects.
```

**Output from the map LLM:**

```json
{
  "answer": "Science-fiction films in this corpus share themes of human ambition, exploration of time and space, and emotionally intense musical scores. Directors such as Nolan and Villeneuve, along with composer Hans Zimmer, recur across these works.",
  "helpfulness": 75,
  "report_ids": ["1"],
  "references": [
    {"target_id": "Hans Zimmer", "target_type": "entity"},
    {"target_id": "Christopher Nolan", "target_type": "entity"}
  ]
}
```

The `answer` is **new text** written by the LLM. It is not a copy of any report.
The `references` are the LLM's guesses about which strings are valid entity IDs.

#### Step 5 — Filter and sort partial answers

Partial answers with `helpfulness == 0` are discarded. The rest are sorted from
highest to lowest helpfulness. In this example, the single batch scored 75, so
it survives.

#### Step 6 — Reduce phase: one final LLM call

The surviving partial answers are packed into the reduce prompt (subject to
`reduce_budget_tokens`) and sent to the LLM. The reduce prompt asks the LLM to
combine the partial answers into one coherent final answer.

**Input to the reduce LLM:**

```text
Question: What are the most common themes across science-fiction films?

Partial Answers (sorted by helpfulness):

[Score 75] Science-fiction films in this corpus share themes of human ambition,
exploration of time and space, and emotionally intense musical scores...
```

**Output from the reduce LLM:**

```text
The most common themes across science-fiction films in this corpus include
humanity's ambition, the manipulation of time and space, and the unifying role
of Hans Zimmer's musical scores. Directors Christopher Nolan and Denis
Villeneuve both appear in this space, favoring practical effects and
philosophical depth.
```

Again, this is new text, not a copy of any stored summary.

#### Step 7 — Resolve citations from map and report references

The system combines two reference sources:

1. **Map output references** — the `references` array returned by each map call.
2. **Used community report references** — for each report ID the map phase
   actually used, the retriever reads `Community.report_json` and extracts
   `findings[*].references`. If a report has no structured references, it falls
   back to parsing `[refs: ...]` blocks from `report_text`.

The retriever filters both lists to supported `target_type` values: `entity`, `relationship`, and `claim`.
It removes duplicate `(target_type, target_id)` pairs.
It resolves chunk citations with the same graph-scoped resolver as Local and DRIFT.

Using the example above, the map output references do not match stored entity
IDs. However, the structured `report_json` for `1` contains validated
references:

```json
{
  "findings": [
    {
      "description": "Hans Zimmer composed scores for science-fiction films",
      "references": [
        {"target_id": "Hans_Zimmer", "target_type": "entity"},
        {"target_id": "Christopher_Nolan", "target_type": "entity"},
        {"target_id": "Denis_Villeneuve", "target_type": "entity"}
      ]
    }
  ]
}
```

Because the map response included `1` and that report ID was
present in the batch, the retriever reads the report's JSON and resolves:

```text
Hans_Zimmer       -> FROM_CHUNK -> chunk:0, chunk:1, chunk:3 -> Document "smoke-source"
Christopher_Nolan -> FROM_CHUNK -> chunk:0, chunk:1, chunk:2, chunk:4 -> Document "smoke-source"
Denis_Villeneuve  -> FROM_CHUNK -> chunk:3 -> Document "smoke-source"
```

If the same report only had rendered text, a finding line such as
`Hans Zimmer composed scores... [refs: entity:Hans_Zimmer, entity:Christopher_Nolan]`
would produce the same references after parsing.

**Result:**

```text
SearchResult(
  mode="global",
  answer="The most common themes across science-fiction films...",
  citations=[
    Citation(document_id="smoke-source", chunk_id="chunk:0", page_start=1, page_end=1, excerpt="Christopher Nolan directed Interstellar..."),
    Citation(document_id="smoke-source", chunk_id="chunk:1", page_start=2, page_end=2, excerpt="Leonardo DiCaprio starred in Inception..."),
    Citation(document_id="smoke-source", chunk_id="chunk:2", page_start=3, page_end=3, excerpt="The Dark Knight Rises was directed by..."),
    Citation(document_id="smoke-source", chunk_id="chunk:3", page_start=4, page_end=4, excerpt="Denis Villeneuve directed Dune..."),
    Citation(document_id="smoke-source", chunk_id="chunk:4", page_start=5, page_end=5, excerpt="Oppenheimer was directed by Christopher Nolan..."),
  ],
  sources=[
    DocumentSource(document_id="smoke-source", document_name=None, chunk_list=[...])
  ],
  metadata={
    "selected_level": 0,
    "reports_available": 2,
    "reports_used": 2,
    "map_batches": 1,
    "map_succeeded": 1,
    "map_failed": 0,
    "map_filtered_zero": 0,
    "reduce_partials_used": 1,
    "source_report_ids_used": 1,
    "source_references_extracted": 3,
    "source_citations_resolved": 5,
    "source_resolution_errors": [],
    "elapsed_ms": ...
  }
)
```

The answer now carries source metadata because the structured report references
pointed to real graph entities with stored chunk links.

---

### How Global Uses the Validated References

When communities are generated with structured reports, every community report
has a `report_json` property containing findings with validated references. For
`1`, that JSON includes:

```json
{
  "findings": [
    {
      "description": "Hans Zimmer composed scores for science-fiction films",
      "references": [
        {"target_id": "Hans_Zimmer", "target_type": "entity"},
        {"target_id": "Christopher_Nolan", "target_type": "entity"},
        {"target_id": "Denis_Villeneuve", "target_type": "entity"}
      ]
    }
  ]
}
```

These `target_id` values are validated against the real entity allowlist when
the report is created. Global search now reads `report_json` for the reports the
map phase used and passes those references to the citation resolver, following:

```text
Hans_Zimmer       -> FROM_CHUNK -> chunk:0, chunk:1, chunk:3 -> Document "smoke-source"
Christopher_Nolan -> FROM_CHUNK -> chunk:0, chunk:1, chunk:2, chunk:4 -> Document "smoke-source"
Denis_Villeneuve  -> FROM_CHUNK -> chunk:3 -> Document "smoke-source"
```

If `report_json` is missing or malformed, global search falls back to parsing
`[refs: ...]` blocks from `report_text`. If neither source yields references,
or if citation resolution fails, the search still returns the answer with
partial or empty citations and records the failure in
`metadata["source_resolution_errors"]`.

---

### When to Expect Sources

| Mode | Source reliability | Reason |
| --- | --- | --- |
| Local | High | Chunk IDs come from stored `(Chunk)-[:FROM_CHUNK]->(__Entity__)` edges. No LLM guessing. |
| DRIFT | High | Local actions use edge-based chunk IDs; selected report references are also resolved to citations. |
| Global | Medium/High | Chunk IDs come from validated `report_json` references (or rendered `[refs: ...]` fallback) for reports the map phase actually used. |

No search mode guarantees source metadata for every claim or sentence-level
grounding. Inspect the cited passages when answer correctness matters.

---

## Search Diagnostics

Every `SearchResult` includes a `metadata` dict with diagnostics for debugging
and monitoring:

```python
result = await global_search.search(
    "What are the main themes?",
    community_level="coarsest",
)
print(result.metadata)
# {
#   "selected_level": 0,
#   "random_seed": 42,
#   "reports_available": 5,
#   "reports_used": 5,
#   "map_batches": 1,
#   "map_succeeded": 1,
#   "map_failed": 0,
#   "map_filtered_zero": 0,
#   "reduce_partials_used": 1,
#   "source_report_ids_used": 1,
#   "source_references_extracted": 3,
#   "source_citations_resolved": 5,
#   "source_resolution_errors": [],
#   "elapsed_ms": 10077,
# }
```

Common keys across modes:

| Key | Modes | Meaning |
| --- | --- | --- |
| `reports_available` | global | Total reports at the selected level |
| `reports_used` | global | Reports that passed quality filters |
| `map_batches` | global | Number of map-phase batches |
| `map_succeeded` | global | Batches that produced a partial answer |
| `map_failed` | global | Batches that errored |
| `map_filtered_zero` | global | Batches filtered for zero helpfulness |
| `reduce_partials_used` | global | Partial answers included in reduce |
| `source_report_ids_used` | global | Number of distinct community report IDs used for source resolution |
| `source_references_extracted` | global | Total distinct references fed to citation resolution |
| `source_citations_resolved` | global | Number of citations returned |
| `source_resolution_errors` | global | Non-fatal error messages from report reading/reference extraction/citation resolution |
| `elapsed_ms` | all | Total wall-clock time |
| `mixed_context` | local | `True` when `use_mixed_context` was enabled for this search |
| `used_tokens` | local | Tokens used by `MixedContextBuilder` (only when `use_mixed_context=True`) |
| `max_tokens` | local | Token budget for mixed context (only when `use_mixed_context=True`) |
| `token_usage` | all | LLM token usage for this search, by stage (see below) |

Token usage is tracked **by default**: every search surfaces a `token_usage`
summary under `result.metadata["token_usage"]` (`retrieval.*` and `drift.*`
stages), the same shape pipelines return. Render it with `render_usage_table`,
and pass `track_token_usage=False` to the retriever constructor to opt out. See
[Token usage](05-pipelines.md#token-usage) for the full explanation and the
[workflows guide](09-workflows.md#token-usage-observability) for sinks,
per-request scoping, and cross-call ledgers.

Diagnostics are read-only. Use them for logging, dashboards, or tuning search
parameters.

---

## Community Levels

Recon-GraphRAG stores communities with `level=0` as the **coarsest / most
global** level. Higher numbers are progressively finer communities.

Use this mapping when comparing terminology:

```text
Recon level 0       = Microsoft C0 / root / coarsest community level
Recon highest level = Microsoft finest / deepest community level
```

The search API supports semantic selectors:

```python
community_level="all"       # No level filter
community_level="finest"    # Highest available level
community_level="coarsest"  # level 0
community_level=0           # Exact stored level
community_level=1           # Exact stored level
```

Passing `community_level=0` selects the coarsest, most global community reports.

`community_level="all"` means "no level filter" for APIs that support it.
Built-in global search does not run across every level at once; it requires one
resolved level.

---

## Token Budgets

There are three separate token budgets. They happen at different times:

| Setting | Where | When | What it controls |
| --- | --- | --- | --- |
| `max_context_tokens` | `CommunityPipeline` / `CommunitySummarizer` | Indexing time | How much entity, relationship, and claim context fits into one community report-generation prompt. |
| `map_budget_tokens` | `GlobalSearchRetriever` | Query time | How many stored community reports fit into one map prompt. |
| `reduce_budget_tokens` | `GlobalSearchRetriever` | Query time | How many scored partial answers fit into the final reduce prompt. |

`max_context_tokens` shapes the report that gets stored on a `Community`.
`map_budget_tokens` and `reduce_budget_tokens` shape how those already-stored
reports are read during global search. They do not overlap.

---

## Customization

Each retriever exposes prompt attributes that you can override for
domain-specific behavior.

### Local Prompt

```python
local_search.answer_prompt = (
    "You are a film analyst. Answer based on:\n{context}\n\nQuestion: {query}"
)
```

### Global Map/Reduce Prompts

```python
global_search.map_prompt = (
    "Answer from this batch of community reports.\n\n"
    "Question: {query}\n\nReports:\n{batch_text}"
)
global_search.reduce_prompt = (
    "Synthesize these partial answers.\n\n"
    "Question: {query}\n\nPartials:\n{partial_text}"
)
```

Global search uses scored map/reduce prompts defined in
`recon_graphrag.retrieval.search_global`.

### DRIFT Prompt

```python
drift_search.reduce_prompt = (
    "Answer {query} from these scored DRIFT actions:\n\n"
    "{action_context}\n\n{conversation_history}"
)
```

DRIFT exposes internal prompts as module-level constants in
`recon_graphrag.retrieval.search_drift`: `PRIMER_PROMPT`, `ACTION_PROMPT`,
`REDUCE_PROMPT`, and `REPAIR_PROMPT`. Only `reduce_prompt` is stored as an
instance attribute and can be overridden on the retriever. To customize the
primer or action prompts, import and patch the module constants before calling
search, or subclass `DriftSearchRetriever`.

### Custom Retrieval Query

Advanced users can override the Cypher query used to fetch local or DRIFT
entity context. A custom query must enforce its own limits.
The retriever still passes `rel_limit` and `source_limit` in the query parameters.
A query that ignores them returns all matching records.

```python
local_search.retrieval_query = (
    "OPTIONAL MATCH (node)-[r]-(neighbor) RETURN node.name AS title, score"
)
```

If you provide a custom query and want citations, return `source_chunk_ids` in
each row.

---

## How Search Works

Before search can work, the graph should contain:

| Node | What it represents | Key properties |
| --- | --- | --- |
| `__Entity__` | Extracted people, places, concepts, etc. | UUID `id`, `canonical_key`, `human_readable_id`, `name`, `title`, `description`, `embedding`, `graph_name` |
| `Chunk` | A text chunk from the original source unit | `id`, `text`, arbitrary source metadata such as `record_id`, `page`, `table`, `ticket_id` |
| `Document` | The source unit or container | metadata such as `title`, `source`, `filename`, `collection`, `external_id` |
| `Community` | A cluster of related entities | `report_text`, `report_json`, `report_status`, `level` |
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

---

## Traversal Limits

Each search mode controls how many nodes, reports, and LLM calls are involved.
The table below summarizes the hard and soft limits.

### Local Search

| Control | Default | What it limits |
| --- | --- | --- |
| `top_k` | `10` | Number of seed entities retrieved by hybrid vector + keyword search. |
| `effective_search_ratio` | `1` | Over-fetch multiplier before post-filtering (`candidate_k = top_k * effective_search_ratio`). |
| Cypher traversal depth | **1 hop** | The retrieval query traverses one hop out from each seed entity: outgoing and incoming relationships to non-Chunk, non-Document, non-Community neighbors. |
| `top_k_relationships` | `10` | Relationships kept **per direction per seed entity**: up to 10 outgoing and 10 incoming. Applied in Cypher (`LIMIT $rel_limit` after ranking), bounding relationship data returned to Python, not relationships inspected by the database. |
| `top_k_sources` | `3` | `FROM_CHUNK` source chunks kept per seed entity. Applied in Cypher (`LIMIT $source_limit`). Ordered by similarity to the query when chunk vectors exist, otherwise by chunk id. Fewer sources means fewer candidate citations. |
| `token_budget` (mixed context) | `12000` | When `use_mixed_context=True`, total tokens packed into the LLM context. Split: 50% text units, 10% community reports, 40% graph facts. |

**Example:** With `top_k=10`, local search retrieves 10 entities, then for each
entity fetches up to `top_k_relationships=10` relationships in each direction and up to
`top_k_sources=3` source chunks. Suppose entity `Christopher_Nolan` has 5 relationships and 4 source chunks.
All 5 relationships are returned, but only 3 chunks.
With chunk vectors, these are the 3 closest to the question. Otherwise, they have the 3 lowest chunk IDs. The rest never leave the database.

These limits bound returned neighborhood/source data, not database traversal or
sorting work, nor LLM cost. `top_k`, `top_k_relationships`, and `top_k_sources`
scale together. The query returns separate incoming/outgoing rows, each carrying
the same bounded source list. Sources are limited in a correlated per-seed
subquery; seeds without sources are retained. Truncation reduces citation
candidates. Raising `top_k_sources` admits more evidence but does not guarantee
that every supporting passage is cited.

These are retrieval-candidate limits. The legacy non-mixed formatter still shows
at most three source excerpts for local search or two for DRIFT; mixed context
uses its token-packing limits. Raising a retrieval cap does not guarantee that
all fetched candidates enter the prompt or are cited.

Callers can override either limit per query via `query_params`, for example
`await local_search.search(question, query_params={"source_limit": 8})`.

### Ranking source chunks

`top_k_sources` has to choose which chunks to keep. Keeping the lowest chunk ids
is deterministic but arbitrary: the kept passages have nothing to do with the
question.

Ingestion can store one vector per chunk with `GraphBuilderPipeline(embed_chunks=True)`.
It is off by default because it costs one embedding call per chunk. When chunk vectors exist and the query supplies a vector, sources are ordered by cosine similarity to the query.
Ties use chunk-ID order, so ordering stays deterministic. The retriever already has a query vector for
hybrid entity search, so ranking adds no provider call at query time.

Scoring happens in the database as a dot product over unit-length vectors, so
Neo4j and Memgraph return the same order and no vector index is needed. Chunks without a vector score `0.0` rather than failing. Vectors with a different dimension than the query also score `0.0`.
Thus, a graph built without `embed_chunks=True` retains chunk-ID ordering.
Callers that pass their own `retrieval_query=` keep full control of source
selection and get no automatic ranking.

### Global Search

| Control | Default | What it limits |
| --- | --- | --- |
| `community_level` | required | Which community level to read. All reports at that level are loaded. |
| `max_reports` | `None` (unbounded) | Reports read from the graph for this query. `None` reads every non-failed, non-empty report at the selected level. |
| `map_budget_tokens` | `12000` | Tokens per map batch. Reports are greedily packed until this budget is filled. Controls how many reports fit in one map LLM call. |
| `map_concurrency` | `5` | Max concurrent map LLM calls. |
| `max_map_calls` | `None` (unlimited) | Hard cap on total map batches, applied while batching: reports beyond the cap are never read into a batch and no map call is made for them. |
| `max_map_tokens` | `None` (unlimited) | Sum of complete formatted map-prompt counts under `token_counter`, including repeated query/template overhead and separators. Stops before the first batch that would exceed the remaining allowance; sends no partial final batch. A cap below the first prompt sends no map calls. Approximate counters provide estimated, not provider-billed, token bounds. |
| `reduce_budget_tokens` | `12000` | Tokens for packing scored partial answers into the final reduce prompt. |
| Map calls (total) | One per retained batch | Depends on report sizes, prompt overhead, packing and configured caps; not an exact `ceil(total_report_tokens / map_budget_tokens)` calculation. |
| Reduce calls | At most `1` | Skipped when synthesis is disabled or no usable map answers remain. |

**Example:** Suppose 20 reports total about 40k tokens and packing produces four
batches with `map_budget_tokens=12000`:
- Batches: 4 (actual packing depends on individual report sizes and overhead)
- Map LLM calls: 4 (concurrent, up to `map_concurrency=5`)
- Reduce LLM calls: 1
- Total LLM calls: 5

`map_concurrency` bounds only how many map calls run at once. `max_reports`, `max_map_calls`, and `max_map_tokens` bound total map work. All three default to unlimited.
Without these limits, this four-batch example uses 5 LLM calls.
With `max_map_calls=2`, it uses 3 calls (2 map and 1 reduce) and drops the remaining reports.
`max_reports` chooses the ID-ordered report prefix in the database **before**
shuffling. The existing `random_seed` shuffles that subset; `max_map_calls` and
`max_map_tokens` retain a prefix of its uncapped batches. All reports in the
selected subset are still loaded unless `max_reports` is set. No cap here bounds
provider output tokens, billed cost, or database scan/sort work. Token accounting
uses the configured counter, which is approximate by default.

### DRIFT Search

| Control | Default | What it limits |
| --- | --- | --- |
| `primer_top_k` | `3` | Community reports retrieved by vector similarity in the primer phase. |
| `top_k` | `10` | Entities retrieved per follow-up action (same as local search). |
| `actions_per_round` | `3` | Questions selected from the shared pending pool per round, independent of `action_concurrency`. |
| `max_rounds` | `3` | Maximum traversal rounds. Depth 0 = primer; a question's `depth` stays its parent-chain depth, not the round number. |
| `max_llm_calls` | `20` | Hard cap on total LLM calls across all phases (primer + actions + reduce). |
| `action_concurrency` | `3` | Max concurrent action evaluations during traversal. |
| `top_k_relationships` | `10` | Relationships kept per direction per seed entity, applied in Cypher (same query as local search). |
| `top_k_sources` | `2` | Source chunks kept per seed entity, applied in Cypher. Fewer sources means fewer candidate citations. |
| `reduce_budget_tokens` | `12000` | Token budget for packing scored action answers into the final reduction prompt. |
| Cypher traversal depth | **1 hop** | Each action's local entity retrieval uses the same 1-hop query as local search. |

**Example:** With `max_rounds=3` and `actions_per_round=3`, normal traversal
is bounded by 3 × 3 = 9 action answer calls:
- 1 primer LLM call
- Up to 3 selected questions per round, for at most 3 rounds (9 action calls)
- 1 reduce LLM call
- **Theoretical max without `max_llm_calls`:** 1 + 9 + 1 = 11
- **With default `max_llm_calls=20`:** the cap still applies, but rounds are the
  normal bound. Follow-ups from any completed action are queued regardless of
  its score, so low-confidence branches are explored too.
- HyDE, extra primer folds, and JSON repairs consume the same call budget, so
  they can reduce how many rounds complete.

### Comparison

| | Local | Global | DRIFT |
| --- | --- | --- | --- |
| **Graph nodes touched** | `top_k` entities + 1-hop neighbors + source chunks | All community reports at one level | `primer_top_k` reports + `top_k` entities per action × actions |
| **Cypher traversal depth** | 1 hop | N/A (reads reports directly) | 1 hop per action |
| **LLM calls** | 1 | map batches + 1 reduce | 1 primer + N actions + 1 reduce (capped by `max_llm_calls`) |
| **What scales cost** | `top_k` and neighbor count | Number of reports and `map_budget_tokens` | `max_llm_calls`, `max_rounds`, `actions_per_round` |

---

## Context Format

The retrievers format graph records into readable text before prompting the LLM.

Local context (`use_mixed_context=False`):

```text
Finding: Nolan (Person)
Connections:
  Person: Nolan -[DIRECTED]-> Movie: Inception
  Person: Nolan -[DIRECTED]-> Movie: Interstellar
Evidence:
  Christopher Nolan directed Inception...
  Nolan's next film was Interstellar...
```

Local context (default mixed-context mode, `use_mixed_context=True`):

```text
=== Graph Facts ===
Entity: Christopher_Nolan (Person)
  Description: Film director
Relationship: Christopher_Nolan -[DIRECTED]-> Inception (weight: 1.0)
Claim: Christopher_Nolan won Academy Award (evidence: 2 chunks)

=== Source Text ===
Christopher Nolan directed Inception. Hans Zimmer composed the score...

=== Community Reports ===
Report 0 (level 0, rating: 8.5):
  Christopher Nolan directed Oppenheimer starring Cillian Murphy...
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

Global context:

```text
Report Segment 12 (level 0):
This cluster centers on Christopher Nolan and his science-fiction films...

Report Segment 15 (level 0):
This cluster covers Hans Zimmer's collaborations with major directors...
```

---

## Token Utilities

Token budgeting uses shared utilities from `recon_graphrag.utils`:

- `ApproximateTokenCounter` — fast estimate using `ceil(len(text) / 4)`.
  Always available, no dependencies.
- `TiktokenTokenCounter` — exact count for the selected `tiktoken` encoding.
  The package is a core dependency. Counts need not match provider billing or chat-message overhead.
- `create_token_counter("approximate")` or `create_token_counter("tiktoken")`
  — factory function.
- `pack_items(items, max_tokens, counter)` — greedy packing for already ordered
  items. Returns included/excluded lists with token telemetry.

These are used internally by global search (map/reduce batching) and
community context packing. You can also use them directly for custom workflows:

```python
from recon_graphrag.utils import ApproximateTokenCounter, pack_items, PackItem

counter = ApproximateTokenCounter()
items = [PackItem(id="r1", text="report text..."), PackItem(id="r2", text="...")]
result = pack_items(items, max_tokens=4000, counter=counter)
print(f"Packed {len(result.included)} items, {result.used_tokens} tokens")
```

---

## Required Indexes

Search depends on indexes created by `store.create_indexes()`:

| Index | Type | Indexed property | Used by |
| --- | --- | --- | --- |
| `entity-embeddings` | Vector | `__Entity__.embedding` | Local, DRIFT |
| `entity-names` | Full-text / text | `__Entity__.name` | Local, DRIFT |
| `community-report-embeddings` | Vector | `Community.report_embedding` | DRIFT (primer phase) |

Neo4j uses `db.index.vector.queryNodes` and `db.index.fulltext.queryNodes`.
Memgraph uses `vector_search.search` and `text_search.search`.

---

## Community Pipeline Prerequisite

Global and DRIFT search need communities. The community pipeline:

1. Detects communities with Leiden.
2. Writes `Community` nodes and hierarchy edges.
3. Summarizes or generates structured reports.
4. Generates report embeddings for semantic DRIFT primer search (when `embedder` is provided).

Built-in global search uses the stored reports from step 3. It does not require
community embeddings.

DRIFT search uses community report embeddings (step 4) for the primer phase.
If embeddings are missing, DRIFT falls back to local-style entity search. Use
`CommunityReportEmbedder` to backfill embeddings for existing graphs:

```python
from recon_graphrag import CommunityReportEmbedder

count = await CommunityReportEmbedder(store, embedder, "entity-graph").embed_reports()
```

---

## Current Limits

- Local and global search do not decompose the query. DRIFT generates follow-up questions.
- Local and DRIFT do not perform entity linking before retrieval.
- Ranking is hybrid vector/keyword fusion, not a re-ranking model.
- Local and DRIFT collect immediate neighbors and source chunks, not arbitrary
  multi-hop paths.
- Global search uses one community level at a time.
- Global search uses bounded parallel map calls.

---

## When To Use Each Mode

- **Local** when the question names a specific entity or asks for a concrete
  fact.
- **Global** when you want all-report processing at a selected community level
  for broad themes, risks, or corpus-level questions.
- **DRIFT** when the question needs specific evidence plus broader context.

---

## Microsoft GraphRAG Alignment

Recon-GraphRAG aligns with Microsoft GraphRAG's search implementation. The table
below shows alignment status for each feature.

| Feature | Status | Default | Notes |
| --- | --- | --- | --- |
| Local entity subgraph traversal | Aligned | — | Same 1-hop neighbor + source chunk retrieval. |
| Hybrid vector + keyword entity search | Aligned | — | Same normalization and ranking logic. |
| Mixed context (entities + reports + claims) | Aligned | `use_mixed_context=True` | Microsoft always includes community reports; recon defaults to `True`. |
| Relationship cap per entity | Aligned | `top_k_relationships=10` | Caps displayed relationships per entity in context. |
| Global map-reduce over community reports | Aligned | — | Same shuffle, batch, map, filter, reduce pipeline. |
| General knowledge fallback | Opt-in | `allow_general_knowledge=False` | When all map scores are 0, optionally appends general knowledge instruction to reduce prompt. Off by default to avoid hallucination. |
| DRIFT primer with community reports | Aligned | — | Same vector search over community report embeddings. |
| HyDE primer query expansion | Aligned | `use_hyde=True` | Generates hypothetical answer from random report template, re-embeds for better retrieval. |
| Primer folds (parallel decomposition) | Opt-in | `primer_folds=1` | Splits reports into N folds for parallel primer LLM calls. Default 1 preserves single-call behavior. |
| DRIFT action mixed context | Aligned | `action_use_mixed_context=True` | Each action builds full mixed context, not just entity-only context. |
| DRIFT iterative traversal | Aligned | `max_rounds=3`, `actions_per_round=3` | Shared pending pool, seeded shuffle per round, bounded rounds and selected questions. |
| Community level resolution | Aligned | — | Same `"coarsest"` / `"finest"` / numeric level selection. |
| Citation resolution from report references | Aligned | — | Same `report_json` → `findings[*].references` → chunk citation pipeline. |
| `synthesize_response=False` agent mode | Aligned | — | All modes support skipping LLM synthesis. |

Key differences from Microsoft's implementation:

- **Default off for `allow_general_knowledge`**: Microsoft allows it as an option; recon
  keeps it off by default to avoid hallucination risk.
- **`primer_folds` defaults to 1**: Microsoft uses parallel folds; recon defaults to
  single-call primer for simplicity. Set `primer_folds > 1` to match.
- **Token budgeting**: Uses `ApproximateTokenCounter` by default (`ceil(len/4)`); Microsoft
  uses tiktoken. Install tiktoken and use `TiktokenTokenCounter` for exact parity.

---

## Next Steps

- Try the search examples in [Quick Start](02-quickstart.md).
- See a full domain example in [Example](07-example.md).
