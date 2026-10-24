# Production-readiness audit

**Verdict: not production-ready as a general-purpose, reusable GraphRAG SDK.**

The implementation has a substantial test suite and useful separation between extraction, graph storage, and retrieval. However, several confirmed behaviors can silently corrupt graph identity, leave partially updated data, or report success without usable embeddings. Sharing a database between graphs is particularly unsafe.

It is suitable for **controlled evaluation and supervised batch workflows**, subject to the constraints below. A supported document count, graph size, or query throughput has not been established.

**Audited revision:** `46db746`, package version `0.10.1`.

**Implementation changes:** none during the audit; the tracked working tree was clean when audit checks concluded. This report was subsequently saved at the user's request.

## Workloads and deployment constraints

A defensible evaluation deployment would have:

- One graph/security domain per database—not merely separate `graph_name` values.
- A single ingestion/maintenance writer, with community and index rebuilds outside query-serving periods.
- Curated, stable document and entity identifiers.
- Immutable or append-oriented inputs; no assumption that reingesting changed documents replaces their previous evidence.
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

| ID | Severity | Finding and production impact | Evidence |
|---|---|---|---|
| F1 | High | Entity embedding ignores graph scope; one graph's run can embed another graph's data using the wrong provider/model. | Confirmed |
| F2 | High | Source-derived document IDs collide, allowing unrelated documents and chunks to overwrite each other. | Confirmed |
| F3 | High | Metadata/properties can overwrite reserved identity and scope fields during persistence. | Confirmed query construction |
| F4 | High | Changed-document ingestion does not prune obsolete evidence; entity vectors are not invalidated when their input changes. | Fixed |
| F5 | High | Community and index maintenance delete existing resources before replacement succeeds. | Confirmed ordering |
| F6 | High | Embedding failures can be reported as success; graph writes are partial and reported counts are input counts. | Confirmed |
| F7 | High | Default entity resolution merges homonyms; extraction identity mistakes can collapse entities before resolution. | Fixed |
| F8 | High for concurrent use | Synchronous database operations block async callers; core node identities lack uniqueness constraints. | Fixed (mechanisms); live race/heartbeat measurement outstanding |
| F9 | High as data grows | Full materialization, quadratic candidate generation, and unbounded neighborhood expansion create scaling limits. | Confirmed mechanisms; thresholds unmeasured |
| F10 | Medium | Citation failures can disappear silently; one retrieval mode ignores citation-metadata options. | Confirmed |
| F11 | Medium | Invalid concurrency can hang ingestion; malformed externally written reports can cause a non-progress loop. | Reproduced, with qualifications |
| F12 | Medium | Lockfile drift, limited mandatory integration coverage, and inconsistent diagnostic sanitization weaken operational confidence. | Confirmed / conditional risk |

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

**Fix status:** Fixed on `prod-hardening`. `CypherGraphWriter.write_graph_document` now replaces one document's contribution inside its `graph_name`: it reads what the graph holds for that document, writes the new records first, then deletes evidence and claims on removed chunks, detaches removed chunks, prunes relationships whose `source_chunk_ids` become empty, rebuilds merged descriptions for entities no longer mentioned, and deletes entities left with no edges. Description text is stored with per-document provenance (`description_sources` as `[document_id, text]` pairs, which survives APOC `mergeNodes`) and merged descriptions are rebuilt by subtracting the replaced document's contribution, so shared entities, shared relationships, and identical text contributed by another document survive. Stored descriptions without provenance raise a `ValueError` naming the document and asking for re-ingestion instead of silently keeping stale text.

Vectors are now refreshed rather than only filled in: the writer records `embedding_input_fingerprint` for the text it expects to be embedded, `EntityEmbedder` records `embedding_content_fingerprint` and `embedding_model_fingerprint` when it writes a vector, `get_unembedded_entities` selects `embedding IS NULL OR content <> input OR model <> current model` (both backends), and `persist_entity_summaries` nulls the expected fingerprint so a new summary description marks its vector stale. `fail_on_embedding_error` also rejects stale vectors, not just missing ones. Legacy graphs without fingerprints are re-embedded on the next pass.

Behavior is pinned by `tests/pipelines/test_writer_replacement.py` (9 cases: shorter reingestion, shared relationship chunk ids and weight, relationship deletion on evidence loss, shared entity survival, per-document description replacement, identical-text survival, summary-status/vector invalidation, graph-name isolation, legacy-provenance rejection), `tests/embeddings/test_entity_embedder.py` (changed-text and model-change re-embedding, fingerprint propagation), and backend query-shape tests. `uv run pytest -m "not integration"` passes (1097 tests). Live Neo4j and Memgraph runs of the replacement path were **not** executed; `shortcut:` per-document relationship strength is still not tracked (max of contributions is kept).

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

**Fix status:** Sub-problem 1 is fixed (commit `774457f fix(pipelines): fail_on_embedding_error verifies embedding completion`, on `prod-hardening`). A strict build now verifies that no entity in the configured graph still lacks an embedding after the embedding pass and raises `RuntimeError("Entity embedding incomplete ...")` when any remain; failures during the pass are logged and re-raised in strict mode (`recon_graphrag/pipelines/graphrag_pipeline.py:958-981`). A public recovery method `complete_embeddings()` fills only missing vectors without re-running extraction, resolution, or summarization. Behavior is pinned by 10 deterministic regressions in `tests/pipelines/test_graph_builder.py` (all-fail, partial success with retained vectors, no-progress upsert, graph-scoped `limit=1` check, `embed_entities=False` skip, strict/non-strict vector-write errors, final-lookup failure), reflected in `tests/compatibility/fixtures/public_api.json`, and documented in `docs/05-pipelines.md` ("Embedding failures and recovery"). Probe-confirmed live on a disposable graph against Neo4j and Memgraph, including provider failure → retained pending state → successful recovery. Sub-problems 2 (document-wide rollback) and 3 (write-statistic accuracy) remain open.

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

**Fix status (F7):** Fixed in three layers. (1) The assembler now treats extracted node IDs as chunk-local references, not global identities: relationship and claim endpoints resolve through a per-chunk mapping that must contain both endpoints (mirroring validator rules). Entity keys are `label:value` when a non-name identity property is configured (schema `identity_property`), document-scoped surrogates (`document_id:label:local_id`) for name-only records, explicit `canonical_key` when provided, and legacy global keys only as a fallback for endpoints missing from the chunk's extraction. Reused local IDs like `p1` therefore stay separate until resolution distinguishes them on real evidence. (2) Normalized resolution accepts the same optional `conflict_properties` argument as hybrid resolution; groups whose configured properties differ on non-empty values are moved to blocked review groups instead of auto-merging, while missing values do not conflict. The pipeline's default normalized strategy now passes configured `entity_resolution_conflict_properties` through, and derives the identity-property map from the active schema for both extraction paths. (3) Both backends share the resolver base class so the behavior is identical on Neo4j and Memgraph. Coverage: 8 new assembler tests (reused local IDs, endpoint mapping, fallback, identity-property scoping, explicit keys), 3 new normalized-conflict resolver tests, and updated pipeline tests pinning document-scoped LLM keys; full suite 1,107 passed. Not run: live-backend conflict scenarios on an existing graph with previously merged wrong entities (re-ingestion or rebuilds required).

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

**Fix status:** Fixed on `prod-hardening` (mechanisms; live load/race measurement outstanding). Blocking database work is offloaded from the event loop with `asyncio.to_thread` at defined boundaries: graph-document writes, validation, description backfill, the strict unembedded check, description summarizer fetch/persist (including the error path), entity-embedding fetch/upsert, and entity-resolution loads/merges. A heartbeat regression test (`tests/pipelines/test_async_offload.py`) pins that writes and validation no longer block the loop.

Identity is now database-enforced: the writer records a length-prefixed graph-scoped `uid` (`len(graph_name):graph_name:id`) on Document, Chunk, and `__Entity__` (reserved key, so callers cannot spoof it), and both index managers preflight existing duplicates (raising a `ValueError` that names label/graph/id/count and advises re-ingestion), backfill missing `uid` values with the same encoding in Cypher, and create a uniqueness constraint per label (Neo4j `REQUIRE n.uid IS UNIQUE`, Memgraph `ASSERT n.uid IS UNIQUE`). Community constraint failures now fail loudly instead of being swallowed. Live-backend coverage of the contract exists as an integration scenario (`assert_uid_contract`, both smoke suites) that backfills a legacy node, verifies the encoded id including a graph name containing `:`, and asserts a duplicate-uid write is rejected with identical counts before/after; it was **not** executed here (no live database), and simultaneous identical-ID ingestion remains unmeasured.

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

### F11 — A few invalid inputs can prevent progress

**Locations:**

- `recon_graphrag/pipelines/graphrag_pipeline.py:561,833`
- `recon_graphrag/embeddings/community_reports.py:35-113`
- `recon_graphrag/graphdb/store_base.py:247-265`

- `extraction_concurrency=0` creates a semaphore that never admits work. Reproduced and cancelled after a short test timeout.
- A whitespace-only externally stored report can be fetched repeatedly without embedding, mutation, or a no-progress guard.

**Qualification:** Normal SDK report rendering emits a heading, so the whitespace-report loop is a defensive interoperability issue, not an ordinary SDK-generated report path.

**Remediation/validation:** Reject non-positive concurrency at entry; filter blank report text and add a progress guard. Test zero/negative values and malformed externally written rows.

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
3. Make strict failures and persistence counts truthful.
4. ~~Define replacement/replay semantics; reconcile obsolete evidence and invalidate stale vectors.~~ Fixed; sub-problem 3 of F6 (persistence counts) remains.
5. Protect existing communities/indexes during failed maintenance.
6. ~~Establish safe identity-resolution defaults and database identity constraints.~~ Fixed: chunk-local extracted IDs and homonyms stay separate until a merge is safe (defended by conflict checks for both hybrid and default normalized resolution), and Document/Chunk/__Entity__ identities gained a graph-scoped `uid` with uniqueness constraints on both backends.

Concurrent serving additionally requires measuring race behavior under load; blocking database calls are now offloaded from the event loop.

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

**Not checked:** live Neo4j/Memgraph/plugin execution, the 22 integration tests, real-provider behavior, representative load, answer-quality benchmarks, or a dependency vulnerability scan.

**Residual risk:** passing unit tests does not establish safe mutable ingestion, tenant isolation, or production capacity; those remain release gates.
