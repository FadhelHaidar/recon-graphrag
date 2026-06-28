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

graph_rag = GraphRAG(
    store,
    llm,
    embedder,
    graph_name="entity-graph",     # graph scope (default: "entity-graph")
    token_counter=None,            # optional token counter for global search
    map_budget_tokens=12000,       # global map budget (default: 12000)
    reduce_budget_tokens=12000,    # global reduce budget (default: 12000)
)
result = await graph_rag.search("Your question", mode="local")
```

---

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
| `query_params` | Optional dict forwarded to the underlying hybrid entity retriever. |

Local search returns citations when retrieved entities have source chunks.
By default, citation metadata is returned after synthesis but is not shown to
the LLM. Opt in when the answer should see source identifiers while being
written:

```python
result = await graph_rag.search(
    "Who directed Inception?",
    mode="local",
    synthesize_citation_metadata=True,
    synthesis_metadata_keys=["record_id", "collection"],
)
```

---

## Global Search

Global search answers broad questions from community reports. It is an
all-report map/reduce flow over one resolved community level:

```text
resolve level
  -> read all successful reports at that level
  -> shuffle reports with random_seed
  -> pack reports into map batches using map_budget_tokens
  -> run one LLM map call per batch, concurrently
  -> filter map answers with helpfulness == 0
  -> sort remaining partial answers by helpfulness DESC
  -> pack partial answers using reduce_budget_tokens
  -> run one LLM reduce call for the final answer
  -> resolve citations from map references and used community report references
```

```python
result = await graph_rag.search(
    "What are the main themes in this dataset?",
    mode="global",
    community_level="coarsest",
    random_seed=42,
)
```

| Parameter | Description |
| --- | --- |
| `level` / `community_level` | Which community level to search. Use `"coarsest"` when you do not know the numeric levels. |
| `random_seed` | Seed for reproducible report shuffling. Defaults to `42`. |
| `map_budget_tokens` | Constructor option on `GlobalSearchRetriever`; maximum report text packed into one map prompt. |
| `reduce_budget_tokens` | Constructor option on `GlobalSearchRetriever`; maximum partial-answer text packed into the final reduce prompt. |
| `map_concurrency` | Constructor option on `GlobalSearchRetriever`; max concurrent map calls. Defaults to `1`. |
| `max_map_calls` | Constructor option on `GlobalSearchRetriever`; max total map calls. Defaults to `None` (unlimited). |

Global search reads stored `Community.report_text` / `Community.summary`
values directly from the graph. It does not embed the user query, use vector
similarity, or traverse raw entity nodes at query time.

After the reduce phase, global search also reads `Community.report_json` for the
community reports that the map phase used. It extracts validated entity,
relationship, and claim references from `findings[*].references`, falls back to
parsing `[refs: ...]` blocks in `report_text` if `report_json` has no usable
references, and resolves those references to chunk-level citations. Map-returned
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

`community_level="coarsest"` resolves to the highest level currently stored in
the graph, so callers do not need to know whether the database has levels
`0..1`, `0..2`, or something else.

Global search skips failed or empty community reports.

Global citations resolve explicit map-phase references to entities,
relationships, or claims, plus validated references from the used community
reports' structured `report_json` or rendered `[refs: ...]` text.

---

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
| `query_params` | Optional dict forwarded to the underlying hybrid entity retriever. |

DRIFT returns citations for the retrieved local source chunks. Community-summary
citations are a separate extension point.
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
result = await graph_rag.search(
    "What evidence is relevant?",
    mode="local",
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

## How Sources Work

Each mode builds `result.citations` and `result.sources` through a different
path. The rest of this section walks through the same example corpus in three
phases so you can see exactly where source metadata comes from and where it
gets lost.

---

### Phase 0: What the Graph Contains

After running `GraphBuilderPipeline` and `CommunityPipeline` on a small movie
corpus, the graph contains nodes and edges like these:

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

Community detection groups entities into clusters. Each `Community` node has a
`summary` (plain text) and, when structured reports are enabled, a `report_json`
property containing findings with validated entity references:

```text
Community "comm:0:0"
  <- IN_COMMUNITY <- Christopher_Nolan, Cillian_Murphy, Oppenheimer
  summary: "Christopher Nolan directed Oppenheimer starring Cillian Murphy..."
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

Community "comm:1:0"
  <- IN_COMMUNITY <- Hans_Zimmer, Inception, Dune, Interstellar
  summary: "Hans Zimmer composed scores for science-fiction films..."
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
when the report is generated. The global search retriever, however, does not
read `report_json`. It only reads `summary`. That distinction is the root of the
global source problem.

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
document "smoke-source". Sources are reliable because every chunk ID came from a
stored graph edge.

---

### Phase 2: DRIFT Search

**Question:** `"How does Hans Zimmer connect Inception to Dune?"`

**Step 1 — Find entities.** Vector/keyword search finds `Hans_Zimmer`,
`Inception`, and `Dune`.

**Step 2 — Entity traversal (same as Local).** The retriever follows
relationships and collects source chunk IDs:

```text
[chunk:0, chunk:1, chunk:3]
```

**Step 3 — Expand to communities.** From the matched entities, the retriever
follows `IN_COMMUNITY` edges to discover which communities they belong to. All
three entities belong to `comm:1:0`.

**Step 4 — Fetch community summaries.** It reads the `summary` text of
`comm:1:0`: "Hans Zimmer composed scores for science-fiction films including
Interstellar, Inception, and Dune..."

**Step 5 — Fetch bridging entities.** It looks for other entities in
`comm:1:0` that were not in the original top-K results, such as `Interstellar`
and the entity representing the broader science-fiction theme. These are
"bridging" entities that connect the user's explicit question to wider context.

**Step 6 — Resolve citations and generate answer.** Citations come from the
same chunk IDs as Local search (`chunk:0`, `chunk:1`, `chunk:3`). The LLM answer
sees three sections: specific entity findings, broader community context, and
related bridging entities.

**Result:** `citations` and `sources` are populated the same way as Local. The
community summary and bridging entities improve the answer text but do not add
citations; the citations still come from the entity traversal.

---

### Phase 3: Global Search

**Question:** `"What are the most common themes across science-fiction films?"`

Global search does not touch entity nodes. It reads community reports and runs
a map-reduce pipeline.

#### Step 1 — Read reports at the selected level

The retriever runs a query that returns, for each community:

```text
id: "comm:0:0"
level: 0
summary: "Christopher Nolan directed Oppenheimer starring Cillian Murphy..."

id: "comm:1:0"
level: 0
summary: "Hans Zimmer composed scores for science-fiction films including Interstellar, Inception, and Dune..."
```

It does **not** return `report_json`. The validated entity references inside
`report_json` are left on the graph node and ignored.

#### Step 2 — Shuffle once

The reports are shuffled with a reproducible random seed. If there were six
reports instead of two, the shuffle might reorder them like this:

```text
Before shuffle (database order): comm:0:0, comm:1:0, comm:2:0, comm:3:0, comm:4:0, comm:5:0
After shuffle (seed=42):         comm:3:0, comm:5:0, comm:1:0, comm:4:0, comm:0:0, comm:2:0
```

The shuffle happens exactly once per search call. Its purpose is to remove
positional bias so the LLM does not always overweight the first report.

#### Step 3 — Pack into batches

The retriever counts tokens and packs reports into batches that fit within
`map_budget_tokens`. With two short reports and a 12000-token budget, everything
fits in one batch. With six reports of ~2000 tokens each and a tight 8000-token
budget, the packing might produce:

```text
Batch 0: comm:3:0, comm:5:0, comm:1:0, comm:4:0
Batch 1: comm:0:0, comm:2:0
```

If one report is larger than the budget, it goes into a batch by itself and
slightly exceeds the budget. If `max_map_calls` is set and there are more
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

Report comm:1:0:
Hans Zimmer composed scores for science-fiction films including Interstellar,
Inception, and Dune. These films share themes of time, space, and human
ambition. Denis Villeneuve directed Dune while Nolan directed Interstellar and
Inception.

Report comm:0:0:
Christopher Nolan directed Oppenheimer (2023) starring Cillian Murphy. The film
won Best Picture. Nolan is known for complex narratives and practical effects.
```

**Output from the map LLM:**

```json
{
  "answer": "Science-fiction films in this corpus share themes of human ambition, exploration of time and space, and emotionally intense musical scores. Directors such as Nolan and Villeneuve, along with composer Hans Zimmer, recur across these works.",
  "helpfulness": 75,
  "report_ids": ["report:comm:1:0:0"],
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

Both reference lists are filtered to supported `target_type` values (`entity`,
`relationship`, `claim`), deduplicated by `(target_type, target_id)`, and then
resolved to chunk citations through the same graph-scoped resolver used by Local
and DRIFT.

Using the example above, the map output references do not match stored entity
IDs. However, the structured `report_json` for `comm:1:0` contains validated
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

Because the map response included `report:comm:1:0:0` and that report ID was
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

### Why Global Cannot Use the Validated References

When communities are generated with structured reports, every community report
has a `report_json` property containing findings with validated references. For
`comm:1:0`, that JSON includes:

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
| DRIFT | High | Same edge-based chunk IDs as Local; community summaries add context but not citations. |
| Global | Medium/High | Chunk IDs come from validated `report_json` references (or rendered `[refs: ...]` fallback) for reports the map phase actually used. |

If you need traceable source metadata for every claim, prefer Local or DRIFT.
Global is now suitable for broad thematic answers where source metadata is
valuable but not required to be sentence-exact.

---

## Search Diagnostics

Every `SearchResult` includes a `metadata` dict with diagnostics for debugging
and monitoring:

```python
result = await graph_rag.search(
    "What are the main themes?",
    mode="global",
    community_level="coarsest",
)
print(result.metadata)
# {
#   "selected_level": 1,
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
| `elapsed_ms` | global | Total wall-clock time |

Diagnostics are read-only. Use them for logging, dashboards, or tuning search
parameters.

---

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
graph_rag.local.answer_prompt = (
    "You are a film analyst. Answer based on:\n{context}\n\nQuestion: {query}"
)
```

### Global Map/Reduce Prompts

```python
graph_rag.global_.map_prompt = (
    "Answer from this batch of community reports.\n\n"
    "Question: {query}\n\nReports:\n{batch_text}"
)
graph_rag.global_.reduce_prompt = (
    "Synthesize these partial answers.\n\n"
    "Question: {query}\n\nPartials:\n{partial_text}"
)
```

Global search uses scored map/reduce prompts defined in
`recon_graphrag.retrieval.global_search`.

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

---

## How Search Works

Before search can work, the graph should contain:

| Node | What it represents | Key properties |
| --- | --- | --- |
| `__Entity__` | Extracted people, places, concepts, etc. | UUID `id`, `canonical_key`, `human_readable_id`, `name`, `title`, `description`, `embedding`, `graph_name` |
| `Chunk` | A text chunk from the original source unit | `id`, `text`, `embedding`, arbitrary source metadata such as `record_id`, `page`, `table`, `ticket_id` |
| `Document` | The source unit or container | metadata such as `title`, `source`, `filename`, `collection`, `external_id` |
| `Community` | A cluster of related entities | `summary`, `report_text`, `report_json`, `report_status`, `level` |
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

### Global Flow

```text
query + selected community_level
  -> resolve numeric level ("coarsest" = highest stored level)
  -> graph query reads all successful non-empty reports at that level
  -> deterministic shuffle
  -> pack reports into map batches
  -> concurrent map LLM calls, one per batch
  -> helpfulness filter and sort
  -> pack partial answers
  -> one reduce LLM call
  -> validate used report IDs
  -> read report_json / report_text for used reports
  -> extract references from findings or [refs: ...] blocks
  -> dedupe and resolve references to citations
```

Structured report generation stores `report_json`, `report_text`, title,
compatibility `summary`, rating fields, version fields, `report_status`, and
`report_error`. Failed report generations are marked with
`report_status="failed"` and are not read by global search.

The map phase reads summaries/reports, not raw nodes. The reduce phase reads
map partial answers, not the original reports. Neither phase performs vector
similarity search.

Global map outputs may include stable references:

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

---

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
- `TiktokenTokenCounter` — exact count using `tiktoken`. Requires the
  `tiktoken` package. Use when provider-level accuracy matters.
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
| `chunk-embeddings` | Vector | `Chunk.embedding` | Not used directly by search modes |
| `entity-names` | Full-text / text | `__Entity__.name` | Local, DRIFT |

Neo4j uses `db.index.vector.queryNodes` and `db.index.fulltext.queryNodes`.
Memgraph uses `vector_search.search` and `text_search.search`.

---

## Community Pipeline Prerequisite

Global and DRIFT search need communities. The community pipeline:

1. Detects communities with Leiden.
2. Writes `Community` nodes and hierarchy edges.
3. Summarizes or generates structured reports.

Built-in global search uses the stored reports from step 3. It does not require
community embeddings.

---

## Current Limits

- The query is not decomposed into sub-questions.
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

## Next Steps

- Try the search examples in [Quick Start](02-quickstart.md).
- See a full domain example in [Example](07-example.md).
