# recon-graphrag resolution: maintainer follow-up

Status: deferred to the user, who maintains recon-graphrag. No upstream issue filed and no
library patch applied. Findings concern installed version **0.10.1**; source inspection and
offline fakes supplement the previous live run, not a new reproduction of its OOM.

## Observed workload and failure

- On dome usecase with large data, structured build produced 136,038 entities and 378,005 chunks.
- Entity vectors: 768 dimensions, stored on Neo4j nodes.
- Neo4j `dbms.memory.transaction.total.max`: **716.8 MiB**.
- Resolution run: 45 minutes, zero merges, client RSS 4.8 GB, no LLM calls.
- Recorded error: `Neo.TransientError.General.MemoryPoolOutOfMemoryError`:
  "allocation of an extra 32.0 MiB would use more than the limit 716.8 MiB …
  dbms.memory.transaction.total.max threshold reached".

## Library findings

### 1. Unbounded load, including unnecessary vectors

`.venv/lib/python3.12/site-packages/recon_graphrag/graphdb/neo4j/entity_resolution.py:32-50`,
`_Neo4jEntityResolver._load_entities`, executes:

```cypher
MATCH (e:__Entity__)
WHERE e.graph_name = $graph_name AND e.name IS NOT NULL
RETURN elementId(e) AS node_id, e.id AS entity_id, e.graph_name AS graph_name,
       e.name AS resolve_value, labels(e) AS labels, properties(e) AS properties
```

No paging, label restriction or property projection. The result is materialized into a list.
All four strategies (exact, normalized, fuzzy, hybrid), including dry runs, share this loader.
The public resolver surface inspected offers no paging/label-scope control.

136,038 × 768 = 104,477,184 vector values before accounting for other properties or runtime
overhead. Ingestion batching does not bound this later query. Moving vectors elsewhere would
reduce payload, but would not make this loader bounded.

### 2. Candidate generation is also a scaling problem

`.venv/lib/python3.12/site-packages/recon_graphrag/graphdb/entity_resolution.py:417-507`:
`_build_fuzzy_groups` calls `_blocked_candidates` per remaining entity. The latter scans the
singleton pool before truncating candidates:

```python
for other in pool:
    # filtering/scoring
    ...
return candidates[:max_candidates]
```

An offline counting fake observed 10,000 visits for 100 lookups with `max_candidates=1`.
Hybrid uses this path too. With mostly unmatched entities, the scan work is quadratic;
pagination alone does not fix it. The entire observed stall cannot be attributed only to OOM.

### 3. Hybrid does not require persisted entity vectors

`.venv/lib/python3.12/site-packages/recon_graphrag/graphdb/entity_resolution.py:772`,
`_score_with_embeddings`, embeds candidate **names** afresh. An offline embedder fake confirmed
calls for those names. Neo4j merge logic invalidates the surviving entity vector with
`node.embedding = null`.

### 4. Failure propagation is opt-in

`GraphBuilderPipeline._resolve_entities` catches/logs exceptions by default;
`fail_on_resolution_error=True` propagates them. An offline failing-store fake confirmed both
paths. This is an API behavior to document, not evidence that every caller wants fail-fast.

## Application corrections needed separately

- `graph/finalize.py` embeds before calling the private pipeline finalizer, incorrectly assuming
  hybrid consumes stored vectors. Resolve before the explicit embedding pass instead.
- `graph/settings.py` disables pipeline embedding. Current ordering therefore leaves merged
  survivors without vectors, with no replacement pass.
- Enable resolution failure propagation if retaining this pipeline: a caller's retry wrapper
  cannot retry an exception swallowed inside the library.
- Reordering helps a fresh graph only: existing embeddings return on incremental runs.
- Preserve all current thinking-off, batching, constraints, distinct-text and provenance rules.

## Suggested upstream acceptance criteria (not implemented)

1. Scoped resolution with bounded, stable reads; project only matching/context/conflict fields,
   excluding embeddings. Do not collect the whole result merely after paging the database.
2. Generate candidates using indexed blocks rather than rescanning the full singleton pool;
   verify scaling on mostly unique event IDs as well as repeated names.
3. Preserve existing label/context conflict checks, aliases, descriptions, relationship
   properties and source provenance. A short custom merge query is not an equivalent fix.
4. Test fresh and already-embedded graphs, all strategies and dry-run behavior under a small
   transaction-memory budget. Merges must not cause paging to skip candidates.
5. Make failures observable to callers; test embedding invalidation and subsequent re-embedding.

Application policy owns which labels may merge and how identities/conflicts are defined.
Library mechanics should own bounded I/O, candidate generation and safe merging.

Changing database or vector backend alone is **not** a resolution fix. No whole-graph library
resolution/summarization pass should be retried against the current 716.8 MiB pool.
