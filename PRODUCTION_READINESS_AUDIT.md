# Production-readiness audit

**Verdict: not production-ready as a general-purpose, reusable GraphRAG SDK.**

The original audit found identity corruption, partial writes and misleading success reporting. Subsequent corrections on `prod-hardening` address graph-scoped embeddings, source identity, reserved fields and the F4/F7/F8 defects below, with deterministic and bounded live AI/database evidence. Partial-write behavior, destructive maintenance, scaling limits and other open findings still prevent a general production-readiness claim.

It is suitable for **controlled evaluation and supervised batch workflows**, subject to the constraints below. A supported document count, graph size, or query throughput has not been established.

**Original audited revision:** `46db746`, package version `0.10.1`. Original locations and reproductions below are historical unless identified as corrective-pass evidence.

**Corrective scope:** follow-up patch to `6160f86` on `prod-hardening`, including the F4/F7/F8 review corrections, regression tests and documentation. Current verification includes 1,182 deterministic passes, 34 real-backend regression/smoke passes, and 4 live AI/database passes; skips and remaining limits are detailed at the end of this report.

## Workloads and deployment constraints

A defensible evaluation deployment would have:

- One graph/security domain per database—not merely separate `graph_name` values.
- A single ingestion/maintenance writer, with community and index rebuilds outside query-serving periods.
- Curated, stable document and entity identifiers.
- Sequential document updates using stable source identities and known contribution ownership; no assumption of atomic or safe concurrent replacement. Legacy records with unknown ownership require migration/rebuild.
- Entity-resolution rules reviewed for the domain; normalized-name auto-merging disabled where homonyms are possible.
- Explicit post-ingestion checks for vector coverage, dangling evidence, and expected document counts.
- Caller-controlled provider deadlines, spending limits, backups, and database privileges.
- A corpus small enough to fit the measured Python and database working sets.

These constraints do **not** establish a production capacity. Unattended mutable-corpus ingestion, shared-database tenant isolation, and latency-sensitive concurrent serving are not approved by this audit.

## Prioritized findings

**Evidence labels**

- **Confirmed:** reproduced locally or directly established by the implementation.
- **Risk:** the mechanism is present, but its production frequency or performance impact needs live testing.
- Database effects inferred from generated Cypher are distinguished from live database reproductions below.

| ID | Severity | Original finding and production impact | Current status and evidence |
|---|---|---|---|
| F1 | High | Entity embedding ignores graph scope; one graph's run can embed another graph's data using the wrong provider/model. | Scope defect fixed; cross-graph regression passes. Global candidate-window recall risk remains. |
| F2 | High | Source-derived document IDs collide, allowing unrelated documents and chunks to overwrite each other. | Fixed with collision-resistant source identity; regressions pass. |
| F3 | High | Metadata/properties can overwrite reserved identity and scope fields during persistence. | Fixed with pre-write reserved-field validation, including UID; regressions pass. |
| F4 | High | Changed-document ingestion does not prune obsolete evidence; entity vectors are not invalidated when their input changes. | Corrected; replacement/vector regressions and live AI E2E pass on both backends. Atomicity/concurrent replacement remain excluded. |
| F5 | High | Community and index maintenance delete existing resources before replacement succeeds. | Fixed for the documented detection/index boundary: detection now computes all rows and publishes them in one transaction after preflight, so a failure keeps the previous communities, and `create_indexes()` reuses compatible indexes instead of dropping them. Verified on both backends against live disposable containers (Neo4j 5.26.0, Memgraph MAGE), plus deterministic failure-injection and rollback regressions. Mid-transaction rollback and post-commit report/embedding staging remain unproven. |
| F6 | High | Embedding failures can be reported as success; graph writes are partial and reported counts are input counts. | Fixed. Strict embedding completion, one transaction per document, confirmed write counts, fail-fast reference validation and bounded retry all landed and were verified on live Neo4j and Memgraph. Residual: work after the write (resolution, summarization, embedding) is not in the transaction. |
| F7 | High | Default entity resolution merges homonyms; extraction identity mistakes can collapse entities before resolution. | Chunk scope/conflict guards corrected; live AI resolution and E2E pass on both backends. Semantic ambiguity and old bad merges remain. |
| F8 | High for concurrent use | Synchronous database operations block async callers; core node identities lack uniqueness constraints. | Offload/UID mechanisms corrected; async, bounded database-race and live E2E checks pass. Load envelope unmeasured. |
| F9 | High as data grows | Full materialization, quadratic candidate generation, and unbounded neighborhood expansion create scaling limits. | Open; confirmed mechanisms, thresholds unmeasured. |
| F10 | Medium | Citation failures can disappear silently; one retrieval mode ignores citation-metadata options. | Fixed: local (mixed and non-mixed) and DRIFT resolution failures are reported in `SearchResult.metadata["citation_warnings"]` (`code="citation_resolution_failed"`, `stage="chunk"|"reference"`) with sanitized notes, DRIFT also records them in `drift_trace.failures`, and mixed mode now forwards `synthesize_citation_metadata`/`synthesis_metadata_keys`. Successful-result shapes are unchanged. Failure-injection regressions pass. Whether cited passages actually support generated statements remains unevaluated. |
| F11 | Medium | Invalid concurrency can hang ingestion; malformed externally written reports can cause a non-progress loop. | Concurrency half fixed: zero/negative values are rejected at every entry point. Blank externally written report loop remains open. |
| F12 | Medium | Lockfile drift, limited mandatory integration coverage, and inconsistent diagnostic sanitization weaken operational confidence. | Partially addressed: lockfile fixed, integration evidence expanded, embedding diagnostics now redact entity names. CI gating and version-matrix coverage remain open. |

## Findings, evidence, and remedies

### F1 — `graph_name` is not an embedding isolation boundary

**Locations:**

- `recon_graphrag/embeddings/entities.py:37-99`
- `recon_graphrag/graphdb/neo4j/store.py:417-426`
- `recon_graphrag/graphdb/memgraph/store.py:458-467`
- `recon_graphrag/pipelines/graphrag_pipeline.py:958-962`

`EntityEmbedder` accepts `graph_name` but fetches:

```python
self.graph_store.get_unembedded_entities(limit=batch_size)
```

Both backends select unembedded entities across the database. A fake-store probe confirmed that an embedder configured for graph A processes a graph-B row.

**Trigger/impact:** Multiple graphs share a database. This can send another graph's entity text to the configured embedding provider, incur unrelated costs, and write vectors from an incompatible model or dimension.

There is also a separate **recall risk**: Neo4j vector/full-text searches obtain global candidates and then filter by graph. Other graphs can consume the candidate window, producing too few in-scope results. See `recon_graphrag/graphdb/neo4j/store.py:250-312,350-382`. This is not evidence that returned search results cross the graph filter.

**Remediation:** Thread `graph_name` through the embedding contract, fetches, and writes. Define whether indexes are database-wide or graph-specific; document candidate post-filtering if retained.

**Fix status:** Fixed (commit `aee4ba8 fix(graphdb): graph-scoped embeddings, hashed doc IDs, neo4j overfetch parity`, on `prod-hardening`). `EntityEmbedder` now threads `graph_name` through fetches and vector writes so an embedder only processes entities of the configured graph; both backends' `get_unembedded_entities`/`upsert_vectors` remain graph-scoped. The separate candidate-window recall risk (global vector/keyword candidates post-filtered by graph) is unchanged by that fix and remains open.

### F2 — Document identity is collision-prone

**Locations:**

- `recon_graphrag/pipelines/graphrag_pipeline.py:982-988`
- `recon_graphrag/extraction/chunking.py:151-164`

When source/title metadata exists, document identity uses a lossy slug rather than a collision-resistant encoding. Reproduced:

- `source="A/B"` and `source="A-B"` both produce `doc:a-b`.
- Distinct non-ASCII-only sources both produce `doc:`.

Chunks derive their IDs from the document ID and ordinal, so the collision propagates.

**Trigger/impact:** Punctuation differences, repeated titles, non-ASCII source names, or a constant source identifier across records. Unrelated content can become one document and overwrite overlapping chunks.

**Remediation:** Provide an explicit stable document-ID input; otherwise hash the exact source identity rather than its display slug. Keep content version separate from document identity.

**Validation:** Test Unicode, punctuation, repeated titles, and same-source updates; assert distinct source identities remain distinct while intentional updates retain identity.

**Fix status:** Fixed (commit `aee4ba8`, on `prod-hardening`). Source-derived document identity is now content-hashed so colliding or unrelated inputs can no longer overwrite each other via the source-based ID path; identity remains based on supplied document attributes otherwise.

### F3 — Caller metadata can mutate persistence identity

**Locations:**

- `recon_graphrag/pipelines/writer.py:44-73,96-102,258-295`

The writer merges nodes using trusted identity fields, then executes:

```cypher
SET d += row.metadata
```

Chunk metadata and entity properties have equivalent unrestricted assignments.

**Trigger/impact:** Metadata containing `id` or `graph_name`, or schema-declared/public-artifact entity properties containing reserved fields. These can change the node's identity or scope after `MERGE`; subsequent evidence and relationship matches may find nothing.

The probe confirmed that reserved metadata reaches the generated query unchanged. This corruption was not executed against a live database.

**Remediation:** Reject reserved keys at public write boundaries. Keep application metadata separate from internal graph fields, or explicitly allowlist assignable fields.

**Validation:** Attempt writes containing `id`, `graph_name`, `embedding`, and internal summary fields; require a clear validation error and no persisted mutation.

**Fix status:** Fixed (commit `49d85c3 fix(pipelines): reject reserved keys before any write in CypherGraphWriter`, on `prod-hardening`). `CypherGraphWriter.write_graph_document` rejects reserved SDK-owned keys in every free-form record map (document/chunk metadata, entity/relationship properties) before any query runs, raising a `ValueError` that names the record type and offending keys; a rejected artifact leaves the database untouched. Document identity continues via metadata `document_id`.

This is a **data-integrity problem, not a demonstrated Cypher-injection vulnerability**.

### F4 — Reingestion is additive, not safe replacement

**Locations:**

- `recon_graphrag/pipelines/writer.py:60-74,104-138,139-216`
- `recon_graphrag/graphdb/neo4j/store.py:417-426,464-482`
- `recon_graphrag/embeddings/entities.py:110-121`

Overlapping chunk IDs have their text replaced, but the writer does not remove:

- Trailing chunks when the replacement document is shorter.
- `FROM_CHUNK` links to entities no longer mentioned.
- Previously contributed relationships and descriptions.

Entity embeddings are selected only when `embedding IS NULL`; changing names/descriptions or persisting new summaries does not invalidate existing vectors.

**Trigger/impact:** A source changes after ingestion. Retrieval can combine new text with old evidence and stale semantic vectors. Identical-input replay has useful `MERGE`/deduplication behavior, but that is not replacement semantics.

**Remediation:** Define an explicit append-versus-replace contract. For replacement, reconcile document-owned evidence and chunks. Fingerprint embedding input plus model identity and invalidate/recompute only changed vectors.

**Validation:** Ingest a document, remove facts and shorten it, then reingest. Removed evidence must disappear, shared entities must survive, and changed entities must receive refreshed vectors.

**Corrective status:** The earlier “fixed” conclusion was premature: nested provenance properties failed on Neo4j, legacy ownership was guessed, and removal could retain old summaries/strength. The shared writer now stores `description_sources` as a flat list of JSON-encoded `[document_id, text]` records and `strength_sources` as JSON-encoded per-document strength records. It replaces document-owned chunks/evidence/claims, rebuilds surviving descriptions and relationship counts/weights/maximum strength, invalidates stale summaries/vectors, and preserves other documents' contributions. Unknown legacy ownership raises a migration error before mutation; identical text is not treated as ownership evidence. Both resolver implementations preserve these records and the canonical scalar UID.

Vectors are now refreshed rather than only filled in: the writer records `embedding_input_fingerprint` for the text it expects to be embedded, `EntityEmbedder` records `embedding_content_fingerprint` and `embedding_model_fingerprint` when it writes a vector, `get_unembedded_entities` selects `embedding IS NULL OR content <> input OR model <> current model` (both backends), and `persist_entity_summaries` nulls the expected fingerprint so a new summary description marks its vector stale. `fail_on_embedding_error` also rejects stale vectors, not just missing ones. Legacy graphs without fingerprints are re-embedded on the next pass.

String-valued OpenAI/Ollama model IDs now produce distinct fingerprints. The Memgraph vector query receives its model parameter. Vector writes compare expected input (or a validated NULL-input text snapshot) rather than overwriting newer invalidation; Neo4j acquires the node lock before testing freshness. Low-level writes without fingerprints clear certification. Resolution invalidates merged input fingerprints, including without a subsequent summary pass.

Evidence includes focused pure-function/query/parameter regressions plus `tests/integration/test_writer_invariants.py` on both disposable backends: shorter replacement, both resolve→replace directions, identical shared text, canonical UID/constraint enforcement, lowered/removed strength, legacy rejection, stale-vector interleavings and resolve→embed. See the corrective-pass table below for executed checks. This does not provide whole-document rollback or serialize competing document replacements.

### F5 — Maintenance is destructive before successful replacement

**Locations:**

- `recon_graphrag/communities/neo4j/detection.py:56-95`
- `recon_graphrag/communities/memgraph/detection.py:67-96`
- `recon_graphrag/graphdb/neo4j/index_manager.py:44-90`

Community detection deletes existing `Community` nodes before projection/precondition checks finish. Those nodes also hold reports and report embeddings. A probe confirmed that the deletion query is submitted before the “no entities” error.

Index creation similarly starts by dropping existing retrieval indexes.

**Trigger/impact:** Missing backend procedures, projection failure, invalid configuration, or other maintenance failure can remove previously usable retrieval artifacts. Concurrent readers can observe missing communities or indexes.

**Remediation:** Preflight before destructive operations; make index setup non-destructive by default. For community replacement, compute the replacement successfully before publishing/deleting the old result. Preflight alone does not protect against failures later in the rebuild.

**Validation:** Inject failures at projection, detection, persistence, and index creation. Previously working retrieval should remain usable, or an explicitly documented maintenance-only operation should fail without pretending recovery was automatic.

**Status and implemented boundary:** Community replacement is atomic at detection: both detectors finish projection and all precondition checks, build every community/membership/parent row in memory, and write them through one `_transaction()` executor, so a failure before commit leaves the previous `Community` rows and reports intact. Index setup is non-destructive: `create_indexes()` inspects existing indexes, reuses a compatible one, and raises an actionable error when a name exists with an incompatible definition.

**Executed evidence (deterministic):** Detector/index/community-pipeline/graph-store/compatibility unit suites pass with `-m "not integration"` (full suite: 1252 passed, 1 skipped), including failure-injection tests for no entities, missing relationships, incompatible index definitions, stores without transactional writes, and seeded prior communities with rolled-back writes.

**Executed evidence (live containers):** A probe ran against disposable `neo4j:5.26.0` and `memgraph/memgraph-mage` containers.

- Neo4j: detection published a hierarchy, `SHOW INDEXES YIELD *` exposed `options.indexConfig` (9 rows), repeated setup was a no-op success, a 99-dimension index was rejected (`dimensions=99, expected 1536`), precondition failures (no entities, no relationship types) and a raw `gds.leiden.stream` procedure error left the seeded community rows and their `report_text` intact, a rerun left no stale communities, and a second graph was never touched. At `gamma=0.05` an 18-node clustered graph produced level 0-1 communities with 3 `PARENT_COMMUNITY` edges.
- Memgraph: the same failure injections kept prior communities and reports intact, reruns left no stale communities, and parent edges were written whenever MAGE reported multiple levels. This probe found and fixed a real gap: `SHOW INDEX INFO` reports `label+property_vector` / `label_text (name: ...)`, prefixes labels with `:`, and omits vector index names and dimensions, so the compatibility check never matched an existing vector index and silently reused an unverified one. `_index_signature()` now normalises those shapes and an existing vector index fails closed (`Cannot verify dimensions and similarity…`), with nothing dropped. MAGE never returned more than a single community level in any tested configuration, so Memgraph hierarchies are flat in practice.

**Residual risk:** Staging is per-detection-call, not per pipeline: community reports are summarized and embedded after commit, so a report failure leaves the new communities without reports (surfaced through the documented `degraded` result), and a failure *after* commit cannot restore the previous generation. Mid-transaction rollback was verified with a stateful in-memory store, not by killing a live driver mid-commit. Memgraph's partition is non-deterministic and MAGE exposes no seed, so community ids and counts can differ between identical runs of the same graph on that backend.

### F6 — Success reporting does not reliably mean a complete graph

**Locations:**

- `recon_graphrag/embeddings/entities.py:49-99`
- `recon_graphrag/pipelines/graphrag_pipeline.py:958-974`
- `recon_graphrag/pipelines/writer.py:33-40,139-163,247-255`
- `recon_graphrag/graphdb/store_base.py:23-50`

Three related problems:

1. Per-entity embedding exceptions are logged and swallowed. The no-progress guard exits normally. **A probe confirmed that `fail_on_embedding_error=True` still returns normally when every provider call fails.**
2. Document, chunk, entity, evidence, relationship, and claim writes use separate queries. Injecting an entity-write failure leaves earlier write calls completed; there is no document-wide rollback.
3. Write statistics return input-list lengths, not actual matched/persisted counts. Missing relationship endpoints can therefore silently drop rows while reported counts remain positive.

Graph “validation” supplies aggregate counts, not proof of embedding coverage or complete document persistence.

**Remediation:** Return explicit stage outcomes and failed IDs; make strict flags effective. Return actual matched/written counts and reject unexpected mismatches. Define partial-write semantics, then either use a bounded document transaction or provide a clear retry/reconciliation path.

**Validation:** Fail every provider request, drop a relationship endpoint, and fail each write stage in turn. Assert accurate failure status, counts, and documented replay behavior.

The SDK does not promise transactions in its documentation; the concern is incomplete persistence and misleading success, not violation of a documented atomicity guarantee.

**Fix status:** Sub-problem 1 is fixed (commit `774457f fix(pipelines): fail_on_embedding_error verifies embedding completion`, on `prod-hardening`). A strict build now verifies that no entity in the configured graph still lacks an embedding after the embedding pass and raises `RuntimeError("Entity embedding incomplete ...")` when any remain; failures during the pass are logged and re-raised in strict mode (`recon_graphrag/pipelines/graphrag_pipeline.py:958-981`). A public recovery method `complete_embeddings()` fills only missing vectors without re-running extraction, resolution, or summarization. Behavior is pinned by 10 deterministic regressions in `tests/pipelines/test_graph_builder.py` (all-fail, partial success with retained vectors, no-progress upsert, graph-scoped `limit=1` check, `embed_entities=False` skip, strict/non-strict vector-write errors, final-lookup failure), reflected in `tests/compatibility/fixtures/public_api.json`, and documented in `docs/05-pipelines.md` ("Embedding failures and recovery"). Probe-confirmed live on a disposable graph against Neo4j and Memgraph, including provider failure → retained pending state → successful recovery. Sub-problems 2 (document-wide rollback) and 3 (write-statistic accuracy) are fixed:

- **Rollback:** `write_graph_document` runs its reads, writes and reconcile of one document inside a single transaction on both backends (commit `14afa92`). A failure inside the write leaves the document's previous state intact; stores without the private `_transaction` hook keep the previous auto-commit path.
- **Truthful counts:** every write query now returns `count(*)` and `write_stats` reports confirmed rows, falling back to submitted rows only when a store reports no count (commit `5c90cbe`). The same commit rejects a document whose relationship endpoint, evidence entity, or claim entity/chunk resolves in neither the document nor the graph, before the first write, with a `ValueError` naming the document and the missing ids.
- **Retry:** a transaction aborted by a deadlock, serialization conflict, or unique-constraint race is retried up to 3 times with a short backoff, reusing the already-extracted document (commits `b8d78a6`, `b2e8b8e`). Nothing else is retried.

**Live evidence (disposable Neo4j 5.26 and Memgraph, this session):** 23 writer-invariant regressions pass on both backends, and a stress run of 15 rounds of two concurrent document writes over shared entities went from 5/15 and 2/15 failed rounds on the pre-patch code, and 7/15 and 15/15 with the transaction alone, to **0/15 on both** with the unique-constraint retry. The failing mode was a `MERGE` race creating the same `:__Entity__` twice (`Neo.TransientError.Transaction.DeadlockDetected` on Neo4j; `ConstraintValidationFailed` / Memgraph's contract violation on `:__Entity__(uid)`). After recovery each shared entity existed exactly once with evidence from both documents.

**Remaining:** extraction, summarization and embedding are outside the transaction, so a failure after the write still leaves that document written; there is no atomicity across the documents of one build; retry is bounded at 3, and a persistent abort still fails the whole batch.

### F7 — Identity resolution can merge distinct real-world entities

**Locations:**

- `recon_graphrag/graphdb/entity_resolution.py:171-191,397-415`
- `recon_graphrag/extraction/assembler.py:58-75,110-114`
- `recon_graphrag/extraction/validator.py:59-64,105-117`

Normalized resolution groups by graph, label, and normalized name. It does not apply hybrid conflict checks. A probe grouped two “Alice Smith” records despite conflicting email properties.

Separately, the assembler treats extracted node IDs as canonical identities. Reusing chunk-local `p1` for different people collapses them before resolution; validation does not enforce semantic identity across chunks.

**Trigger/impact:** Homonyms, ambiguous organization names, or LLM-generated IDs that are unique only within a response. Incorrect merges contaminate relationships, descriptions, and citations.

**Remediation:** Prefer domain keys; make ambiguous normalized matches reviewable rather than automatically destructive. Define and test the extraction ID contract, and detect conflicting identity attributes.

**Validation:** Test same-name/different-person records, conflicting identifiers, and repeated chunk-local IDs.

**Narrow additional defect:** A node's `canonical_key` override is not applied consistently to relationship endpoints. This was reproduced, but default schema filtering removes that property unless explicitly declared; it is an extension-path defect, not a default-path failure.

**Corrective status (F7):** The earlier document-only fallback still collapsed `p1` reused within one document. The fallback is now `document_id:chunk_id:label:local_id`; trusted explicit canonical keys and non-name identity properties retain their existing label-scoped behavior. The same per-chunk mapping feeds entities, relationship endpoints, claims, observations and evidence. Unit regressions assert Alice/Bob separation within a document and correct citations/endpoints. Live tests on both backends persist four reused-`p1` observations and retain same-name people with conflicting configured birth dates through normalized resolution.

Normalized resolution applies configured conflict checks and leaves conflicting groups for review; missing values do not conflict. Matching names without known conflicts remain heuristic, not proof of identity. Both backends retain canonical scalar UIDs after merges and reject duplicate UIDs with constraints installed. Already incorrectly merged legacy identities are not reconstructed automatically and require a source rebuild.

### F8 — Async concurrency and database identity guarantees are insufficient

**Locations:**

- `recon_graphrag/graphdb/neo4j/store.py:39-47`
- `recon_graphrag/pipelines/graphrag_pipeline.py:886-913`
- `recon_graphrag/graphdb/neo4j/index_manager.py:79-90`
- `recon_graphrag/graphdb/memgraph/index_manager.py:91-106`

Async orchestration calls synchronous database methods directly. `session.run` and result materialization block the event-loop thread.

Index setup creates a community uniqueness constraint, but not corresponding identity constraints for documents, chunks, or entities.

**Trigger/impact:** Slow database operations stall unrelated async work; cancellation cannot promptly interrupt blocking calls. Concurrent ingestion can race on `MERGE` without database-enforced identity uniqueness. The latter needs live race testing.

**Remediation:** Offload blocking database work at a defined boundary, or offer an async backend implementation. Add supported identity constraints after checking existing data for duplicates.

**Validation:** Measure event-loop heartbeat during delayed queries; run simultaneous ingestion of identical IDs and verify one logical node per identity. Use `EXPLAIN`/`PROFILE` to verify lookup plans—no query-plan performance claim is established here.

**Corrective status:** Merely making backfill `async def` did not offload its body; it now awaits `asyncio.to_thread`, as do all four resolver preflights. `tests/pipelines/test_async_offload.py` exercises actual pipeline, summarizer, embedder and resolver methods with delayed synchronous local-store operations, asserting thread identity and event-loop heartbeat across reads/writes/preflight/load/merge. This is deterministic boundary coverage, not a production-latency benchmark. Offloaded calls are not immediately cancellable database operations.

Identity is now database-enforced: the writer records a length-prefixed graph-scoped `uid` (`len(graph_name):graph_name:id`) on Document, Chunk, and `__Entity__` (reserved key, so callers cannot spoof it), and both index managers preflight existing duplicates (raising a `ValueError` that names label/graph/id/count and advises re-ingestion), backfill missing `uid` values with the same encoding in Cypher, and create a uniqueness constraint per label (Neo4j `REQUIRE n.uid IS UNIQUE`, Memgraph `ASSERT n.uid IS UNIQUE`). Community constraint failures now fail loudly instead of being swallowed. The existing `assert_uid_contract` scenarios were executed on both disposable backends. The new suite also runs two simultaneous identical-ID ingestions with constraints already installed, requires one committed Document/Chunk/entity per identity after success, allows an explicitly surfaced competing backend conflict, and checks the same IDs in another graph remain independent. A held-writer Neo4j test waits for a blocked vector transaction before commit and confirms the stale result cannot erase newer invalidation. These bounded races are not evidence of a safe concurrent document-replacement protocol or a supported load envelope.

### F9 — Bounded concurrency does not bound memory or total work

| Operation | Evidence and growth mechanism |
|---|---|
| Ingestion | `recon_graphrag/pipelines/graphrag_pipeline.py:339-342,438-454,833-886`: materializes inputs and creates tasks for all documents/chunks; semaphores bound active work, not queued tasks/results. |
| Fuzzy resolution | `recon_graphrag/graphdb/entity_resolution.py:417-505`: repeatedly scans the singleton pool to select candidates; quadratic candidate-inspection work in the worst case. |
| Local/DRIFT context | `recon_graphrag/retrieval/_queries.py:21-150`: gathers incident neighbors/evidence before downstream context limits. A small entity `top_k` does not bound hub expansion. |
| Global search | `recon_graphrag/retrieval/search_global.py:586-681`: builds report batches and runs a map call per batch. Token limits bound individual prompts, not total per-query tokens or calls. |
| Embedding/reporting | `recon_graphrag/embeddings/entities.py:65-89` and `recon_graphrag/communities/summarization.py:99-177`: per-item provider work and materialized task collections; batch fetching is not equivalent to provider-side batch embedding. |

**Impact:** Python memory growth, database response expansion, long event-loop stalls, provider throttling, and unexpectedly expensive global queries. Community projection also requires database-side capacity measurement.

**Remediation:** Start with bounded producer/worker queues, query-side neighborhood limits, and indexed/block-based resolution candidates. Add explicit total-work/cost limits for retrieval. Use provider batching where available; do not rewrite the architecture first.

**Validation:** Measure each stage independently over document count, graph degree, report count, and concurrency. Confirm both active work and resident queued work stay bounded.

### F10 — Citation degradation can be invisible

**Locations:**

- `recon_graphrag/retrieval/search_local.py:128-135,196-203,261-270`
- `recon_graphrag/retrieval/mixed_context.py:293-303`

Citation lookup exceptions can become an empty citation list without a diagnostic in the result. Mixed-context local search also does not forward the public citation-metadata synthesis options.

**Trigger/impact:** Citation-store failure or mixed-context callers relying on those options. An answer can appear successful without distinguishable provenance degradation.

**Remediation:** Return a citation-resolution warning/status, optionally fail strictly, and propagate supported options consistently.

**Validation:** Inject citation lookup failure and assert an observable degraded result. Test option behavior in both context modes. Separately evaluate whether generated statements are actually supported by cited passages; structural citation tests cannot establish that.

**Status and executed evidence:** Citation-resolution failures are reported in `SearchResult.metadata["citation_warnings"]` (absent on success, so the successful result shape is unchanged) with entries carrying `code="citation_resolution_failed"`, `stage` (`"chunk"` or `"reference"`), and one sanitized note — exception type and counts only, never provider text. Local search reports them in both context modes, DRIFT appends them to `drift_trace.failures`, `MixedContextResult` carries them, and the mixed-context builder now honours `synthesize_citation_metadata`/`synthesis_metadata_keys` in local and DRIFT actions while `SearchResult.context` stays exactly the text sent to the LLM. Failure-injection regressions (`tests/retrieval/test_hybrid.py`, `test_mixed_context.py`, `test_drift.py`) assert the warning text, the absence of raw exception payloads, that a degraded citation list does not abort the answer, and option behaviour with the flags on and off.

**Residual risk:** These tests establish observability and option plumbing only. They do not show that generated statements are supported by the cited passages, and no live provider citation-failure was exercised.

### F11 — A few invalid inputs can prevent progress

**Locations:**

- `recon_graphrag/pipelines/graphrag_pipeline.py:561,833`
- `recon_graphrag/embeddings/community_reports.py:35-113`
- `recon_graphrag/graphdb/store_base.py:247-265`

- `extraction_concurrency=0` creates a semaphore that never admits work. Reproduced and cancelled after a short test timeout.
- A whitespace-only externally stored report can be fetched repeatedly without embedding, mutation, or a no-progress guard.

**Qualification:** Normal SDK report rendering emits a heading, so the whitespace-report loop is a defensive interoperability issue, not an ordinary SDK-generated report path.

**Remediation/validation:** Reject non-positive concurrency at entry; filter blank report text and add a progress guard. Test zero/negative values and malformed externally written rows.

**Fix status:** the concurrency half is fixed by commit `5f9edb6`. `_require_concurrency()` validates `extraction_concurrency` and `document_concurrency` in `build_from_text`, `build_from_documents` and `build_from_rows` before any work starts (CSV/Excel/SQL callers reach it through `build_from_rows`), raising `ValueError("extraction_concurrency must be at least 1")` or `ValueError("document_concurrency must be at least 1")`. Pinned by a parametrized regression over 0 and -1 that asserts no LLM call, no write and no query happens first. The whitespace-only externally stored report loop remains open (defensive interoperability case only).

### F12 — Operational evidence and diagnostics need tightening

**Locations:**

- `pyproject.toml:1-65`
- `uv.lock:1245-1249`
- `.github/workflows/ci.yml:14-36`
- `recon_graphrag/embeddings/entities.py:69-99`
- `recon_graphrag/pipelines/graphrag_pipeline.py:612-630`

- **Lock drift:** ~~package metadata is `0.10.1`, while the committed editable-package lock entry is `0.10.0`. `uv lock --check --offline` failed.~~ Fixed by commit `aee4ba8` (lockfile updated to match `0.10.1`); not re-run since.
- **Compatibility evidence:** mandatory CI runs non-integration tests on Python 3.11. This audit passed on 3.14.7, but that does not establish backend/plugin or minimum dependency compatibility.
- **Sensitive diagnostics:** extraction errors deliberately sanitize document/provider details, but embedding failures log entity names and raw exception text. The outage probe demonstrated the entity name in logs.

**Remediation:** Refresh/check the lockfile in CI; run a small real-backend contract matrix for supported versions and failure paths; apply consistent diagnostic redaction.

**Validation:** Require a clean lock check, backend contract tests, and a log-capture test with sensitive entity names/provider errors.

**Fix status:** the diagnostics half is fixed by commit `83cc613`. An entity embedding failure now logs `entity id` (a `uuid5` URN) and the exception class at `WARNING`; the entity name and provider text moved to `DEBUG`, so the failure stays traceable without leaking graph content. Pinned by a `caplog` regression asserting neither the name nor the payload appears at `WARNING` or above. The lockfile item was fixed earlier (`aee4ba8`). Still open: CI does not gate `uv lock --check`, and there is no supported-version backend matrix.

## Positive evidence and important qualifications

- **1,042 non-integration tests pass.** The suite covers significantly more than basic happy paths.
- Shared graph contracts, neutral artifacts, and backend-specific adapters provide a workable extension structure.
- The writer parameterizes values and escapes dynamic identifiers. No demonstrated injection route was found in these paths; caller-supplied query extensions remain trusted code.
- Extraction has sibling-task cancellation/cleanup and sanitizes contextual errors.
- Community reports include status/fingerprint handling; report-vector invalidation already offers a pattern for fixing entity vectors.
- Token budgets, concurrency controls, usage tracking, and provider configuration exist. It would be inaccurate to claim “no retries, timeouts, or observability anywhere.” The gap is consistent end-to-end failure, deadline, and recovery semantics.
- Database sessions are scoped per operation; the finding is blocking behavior, not a demonstrated session leak.

The maintainability issues are concentrated at shared boundaries—identity, writer validation, stage outcomes, and sync/async storage—not evidence that the SDK needs a wholesale rewrite.

## Remediation roadmap

### Production blockers

1. Fix graph-scoped embedding and reserve internal persistence fields.
2. Replace lossy document identity generation.
3. Make strict failures and persistence counts truthful. **Done:** strict embedding completion, confirmed write counts, fail-fast unresolved references.
4. Replacement/replay corrective regressions pass on disposable backends. **Done:** one document write is one transaction on both backends; deadlock/serialization/unique-constraint aborts are retried three times before surfacing; verified live on Neo4j 5.26 and Memgraph, including a concurrency stress run. Residual: extraction, summarization and embedding stay outside the transaction, and there is no atomicity across documents in one build.
5. Protect existing communities/indexes during failed maintenance.
6. Chunk-scoped fallback IDs, configured conflict checks and graph-scoped UID constraints now have regression evidence on both backends. Name matching remains heuristic; do not interpret absence of a configured conflict as proof that a merge is safe.

Concurrent serving additionally requires measuring race behavior under load; blocking database calls are now offloaded from the event loop, and per-document write races are retried rather than lost.

### Near-term improvements

- Bound ingestion queues and database neighborhood expansion.
- Make citation degradation observable.
- Validate concurrency and add progress guards.
- Align dependency lock/CI checks and introduce live-backend failure tests.
- Standardize sanitized diagnostics and stage-level completion metrics.

### Optional, measurement-driven optimizations

- Provider-side embedding batches.
- Better fuzzy candidate indexing.
- Selective community recomputation and report reuse.
- Caching repeated global-search work.

These should follow correctness work and benchmarks, not precede them.

## Benchmark plan

The following are **test points, not claimed supported capacities**:

| Dimension | Suggested coverage |
|---|---|
| Documents | 100, 1,000, 10,000, then 100,000 only if resource limits permit; vary document token length independently. |
| Graph size/density | Approximately 1K–1M entities; sparse, medium-density, and hub-heavy graphs. Include skewed degree distributions. |
| Updates | Identical replay, 1%/10% changed documents, shorter replacements, removed facts, and interrupted ingestion. |
| Concurrency | 1, 4, 16, 64 callers; separate ingestion, local search, global search, DRIFT, and mixed workloads. |
| Failure cases | Provider timeouts/429s, database disconnects, unavailable procedures, cancellation, citation-store failure, and maintenance overlap. |

Record:

- Per-stage throughput; p50/p95/p99 latency and error/degraded-result rates.
- Python peak RSS and event-loop lag; database heap/projection memory, query plans, and connection waits.
- Provider requests, retries, input/output tokens, embedding volume, and actual priced cost.
- Duplicate identities, missing evidence, stale vectors, citation resolution, and retrieval/answer quality against labeled examples.

Use deterministic providers first to isolate SDK/database behavior, then a budget-capped real-provider sample. Establish acceptance thresholds from the intended deployment SLO before declaring a supported envelope.

## Checks performed and evidence gaps

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -m 'not integration' -q` | **1,042 passed, 22 deselected**; latest audit run 2.24 seconds. |
| `uv run pytest -m "not integration" -q` (after F6 sub-problem 1 fix) | **1,082 passed**; plus `RUN_WORKFLOW_INTEGRATION_TESTS=1` deterministic workflow suite Neo4j + Memgraph, and live strict-embedding-failure/recovery probe on a disposable graph for both backends. |
| `uv run pytest -q -m "not integration"` (after F4/F7/F8 fixes) | **1,122 passed, 22 deselected** on branch `prod-hardening`. |
| Targeted deterministic probes | **11 behavior checks confirmed**, using real SDK code with fake stores/providers. |
| `uv lock --check --offline` | Failed: committed lockfile needs updating. **Resolved** by commit `aee4ba8` (lockfile updated to match `0.10.1`); not re-run since. |
| Independent source review | Findings challenged using `deepseek-v4.1-flash`; reachability qualifications above incorporated. |
| Final tracked-file check during audit | Clean; no implementation changes retained. This report was saved afterward. |

Probe artifact: `/tmp/recon_production_audit_checks.py`

Probe output: `/tmp/recon-production-audit-checks.log`

These are temporary local artifacts, not committed test fixtures. They may not survive cleanup or be available on another machine.

Reproduce the probes, while the artifact is available, with:

```bash
PYTHONPATH=. .venv/bin/python /tmp/recon_production_audit_checks.py
```

Test environment included Python **3.14.7**, Neo4j driver **6.2.0**, and OpenAI SDK **2.36.0**.

### Corrective-pass checks (supersede the original F4/F7/F8 completion claims)

| Check | Result |
|---|---|
| `uv run pytest -q -m 'not integration'` | **1,182 passed, 1 skipped, 48 deselected**. Skip: Memgraph parameter of Neo4j-specific lock-query check. |
| `uv build` | Passed; source distribution and wheel built. |
| `RUN_DATABASE_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_writer_invariants.py -m database` | **23 passed, 1 skipped** on disposable Neo4j 5.26 + APOC core and Memgraph 3.13.2. Skip: Neo4j-only held-lock race on Memgraph. Local mock embedding provider only; persistence uses real databases. |
| Existing backend smoke/resolution checks | **11 passed, 2 skipped** across both store-smoke and entity-resolution integration modules; includes constraints and legacy UID backfill. The two skipped tests require live AI providers. |
| Fresh independent review | Found resolve→embed fingerprint invalidation gap; reproduced failing on both backends, fixed in both resolver merge paths and regression-tested without summarization. |
| Live AI + disposable database follow-up | **4 passed, 0 skipped in 552.03 seconds**: both synthetic E2E tests in `tests/integration/test_smoke_full.py`, plus `test_neo4j_hybrid_uses_real_embedder_and_llm_for_review` and `test_memgraph_hybrid_uses_real_embedder_and_llm_for_review`. The two previously skipped AI-resolution cases are now exercised. |

The new backend suite covers provenance/property compatibility, both rewiring directions, resolve→replace and resolve→embed, shorter documents/claims, strength recomputation, legacy rejection, stale snapshots/held locks, conflict-aware chunk identity, concurrent identical-ID ingestion and graph isolation. The compatibility fixture change is limited to declaring the deliberate shared Alice key; no public API snapshot rewrite was made. The SQLite mini-Cypher vector emulator was replaced by focused query/parameter tests plus real persistence regressions.

The live follow-up used the configured OpenAI-compatible endpoints with LLM `qwen3.8-27b` (`LLM_ENABLE_THINKING=false`) and embedder `jina-omni-nano-retrieval`, against fresh Neo4j 5.26.0 (APOC 5.26.0, GDS 2.13.2) and Memgraph 3.13.2/MAGE. Both E2E runs completed extraction/claims, conflict-aware hybrid resolution, entity/relationship summarization, vector persistence, community detection/report embedding, and local/global/DRIFT retrieval. Citation assertions passed; DRIFT had primer reports and no fallback. Neo4j produced 11 entities, 20 entity relationships, 14 written claims and 7 embedded reports; Memgraph produced 10 entities, 18 entity relationships, 12 written claims and 5 embedded reports. These are observed stochastic outputs, not required equal backend results or quality scores.

The run emitted 422 existing Neo4j notification-setting deprecation warnings and 4 embedding-dimension warnings: configured `EMBEDDING_DIM=1536` exceeded the embedder's native 768 dimensions, so the SDK kept 768. E2E index dimensions were obtained from an actual embedding probe. Sanitized local output is `/tmp/recon-ai-live-safe.log` (temporary, not committed). No source or `.env` changes were needed. Only fresh disposable databases were written, and their containers were removed after testing; existing database containers were left intact.

**Not checked in the corrective pass:** production data, representative load, large legacy migrations, answer-quality benchmarks, query-plan/performance claims, alternate AI models/providers, or dependency vulnerability scans. Successful synthetic live runs do not establish a production operating envelope.

**Residual risk:** multiple-query writes can remain partial, write stats remain input counts, concurrent document replacements are not serialized, and F5 is fixed only for the documented detection/index boundary (post-commit report/embedding staging remains). Passing these unit and bounded backend regressions does not establish production capacity or a complete production-safety guarantee.
