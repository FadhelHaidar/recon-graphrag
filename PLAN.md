# GraphBuilder-first remediation plan

Scope: production blockers from `PRODUCTION_READINESS_AUDIT.md`, GraphBuilder
first. F1, F2, F3, F4, F6, F7, F8 are treated as completed; the two F6
residuals are still open in the code (evidence below) and stay in scope.

Ordering: everything inside `GraphBuilderPipeline` lands first (Section 2),
community/index maintenance (F5) after it (Section 3, unchanged plan, separate
PR). F9 is explicitly deferred with a reason.

Verified against the working tree at `e4e0f21`. Pre-existing `uv.lock` change
and untracked `AGENTS.md` are left untouched.

## 1. Status, scoped to the builder

| Item | Class | Status | Evidence |
| --- | --- | --- | --- |
| **F6-b misleading write receipts** | blocker | **Open** | `_stats_for` (`pipelines/writer.py:885-893`) returned input lengths; `MATCH`-endpoint writes (`writer.py:495-508`, `510-606`, `608-635`) drop rows silently while the receipt still looks complete, and `graphrag_pipeline.py:923-934` logs it as success. |
| **F6-a partial document writes** | blocker | **Open** | `write_graph_document` (`writer.py:130-155`) is ~10 auto-committing `execute_query` calls; no rollback (`graphdb/neo4j/store.py:40-53`, `graphdb/memgraph/store.py:68-81`). |
| **F11 zero concurrency hangs** | near-term (fold in) | **Open** | `asyncio.Semaphore(extraction_concurrency)` at `graphrag_pipeline.py:572`, `:859`; `Semaphore(0)` admits nothing, so `as_completed` (`:884`) never yields. `document_concurrency` is already clamped (`:278`) — asymmetry, not design. |
| **F12(c) sensitive diagnostics** | near-term (fold in) | **Open** | `embeddings/entities.py:123-127`: `logger.warning("error embedding entity '%s': %s", name, result)` logs the entity name *and* the raw provider exception, on the `fail_on_embedding_error=False` default path. |
| **F9 unbounded ingestion memory** | near-term | **Deferred** | Per-chunk tasks created up front and all extractions retained until assembly (`graphrag_pipeline.py:881`, `:891`, `:915`). Real, but it restructures the ingestion loop; doing it before F6-a conflicts with the transaction boundary work. |
| F5-a community rebuild, F5-b index setup | blocker | Open, not GraphBuilder | `communities/*/detection.py`, `graphdb/*/index_manager.py`. Section 3. |
| F10, F12(a) lock, F12(b) CI matrix | other | See audit | Out of scope here. |

## 2. GraphBuilder work, in order

Order rationale: F6-b touches only the writer and shrinks what F6-a must
protect; F11 and F12(c) are 3-6 line riders on adjacent lines in
`graphrag_pipeline.py` / `embeddings/entities.py` and need no sequencing. F6-a
last, because it is the only one that changes execution semantics.

### Step 1 — truthful write receipts + fail-fast reference validation (F6-b)

**Failure scenario.** An extraction names a relationship endpoint, claim
entity, or evidence entity that does not exist. The write queries `MATCH`
endpoints, so the row is dropped, `write_stats` still reports the input counts,
and the pipeline logs `write complete ... N entities`. Example: a document whose
relationship targets an entity that a concurrent writer removed → the receipt
says 40 relationships, the graph has 32, no error anywhere.

**Root cause.** `_stats_for` reports `len()` of the input document instead of
what the database persisted, and nothing checks endpoints before the first
mutation.

**Smallest change, all in `recon_graphrag/pipelines/writer.py`:**

1. Add private `_validate_references(graph_document, state)`, called immediately
   after the existing pre-mutation validation (`writer.py:146-147`), before
   `_write_documents`.
   - Allowed chunks = document chunks + `state.chunk_ids`; allowed entities =
     document entities + `state.entities`.
   - Anything else resolves through one read, `_existing_entity_ids`
     (`UNWIND $entity_ids` / `MATCH (e:__Entity__)`), so cross-document links
     still work.
   - Unresolved → `ValueError` naming the document and the missing ids, before
     any write.
2. Append `RETURN count(*) AS written` to the document, chunk, entity, evidence,
   relationship, and claim write queries; return
   `_confirmed(result, submitted_rows)` from each helper
   (`_write_entities` accumulates across per-type groups; `_write_relationships`
   counts this document's rows only; `_write_claims` counts rows that survived
   both endpoint `MATCH`es).
3. Delete `_stats_for`; build the same six-key dict in `write_graph_document`
   from the helper returns.
4. `_confirmed()` falls back to `submitted` when the result carries no count,
   with a `shortcut:` comment — production always reads the real aggregate;
   test doubles and custom `execute_query` overrides keep working.

**Deliberately not changed.** `_read_document_state`'s entity-id set stays as
is: `_rebuild_entity_rows` (`writer.py:694-737`) runs `_accessible_entries` over
every `state.entities` entry, so adding lookups for newly referenced endpoints
would raise the legacy-provenance error on unrelated entities.

**Compatibility.** Same six keys, same `dict[str, int]`; `GraphBuilderPipeline`
call sites (`graphrag_pipeline.py:484-500`, `:923-934`) untouched; custom
`GraphWriter` returns still pass through (`tests/compatibility/test_ingestion_contract.py`).
No `GraphStore`/`GraphWriter`/public signature change. Values become persisted
counts — the fix.

**Validation.** New `tests/pipelines/test_writer_receipts.py`: store reporting
`{"written": n}` < submitted makes stats report `n`; unknown relationship
endpoint raises and the store sees no `MERGE`; claim with an absent chunk raises
the same way; a cross-document endpoint present in the store is accepted.
Existing pins must pass unchanged (`tests/pipelines/writer_scenarios.py:493-510`,
`test_writer_characterization.py`, `test_writer_replacement.py`,
`test_graph_builder.py`).

**Acceptance.** Stats equal confirmed writes for all six keys; unresolvable
endpoints fail with a named `ValueError` before any mutation; cross-document
references still work; deterministic suite green; `docs/05-pipelines.md:406`
updated (receipt + fail-fast; the document-wide transaction caveat stays until
Step 4).

### Step 2 — reject non-positive concurrency (F11 rider)

**Failure scenario.** `build_from_documents(docs, schema=..., extraction_concurrency=0)`
→ 12,000 chunk tasks queue, `Semaphore(0)` admits none, `as_completed` never
yields, the progress bar sits at `0/12000`, nothing raises, and the `finally`
cancel path (`:900-904`) is unreachable. Auditor reproduced it and had to cancel.

**Root cause.** `asyncio.Semaphore(0)` is legal; only `document_concurrency` is
clamped (`:278`).

**Smallest change.** One guard at the entry of each public builder:
`build_from_text` (`:145-152`), `build_from_documents` (`:244-252`),
`build_from_rows` (`:426-436`) — reject `extraction_concurrency < 1` (and
`document_concurrency < 1`) with `ValueError`, mirroring the existing
`_require_schema` style (`:679-684`). No clamp-to-1: silently ignoring a caller's
0 hides a config typo, and every other invalid input here already raises.

**Compatibility.** Public signatures unchanged; only a previously-hanging input
now raises. No behavior change for valid callers.

**Validation.** Parametrized over `0` and `-1` on all three entry points: raises
`ValueError` immediately, no task created. Existing concurrency tests pass.

**Acceptance.** No builder path can construct a zero-width semaphore; invalid
values fail in milliseconds instead of hanging.

### Step 3 — redact embedding diagnostics (F12-c rider)

**Failure scenario.** Provider 429 during `_finalize_graph` →
`EntityEmbedder.embed_entities` warns once per failed entity with the entity
name and the raw exception, whose text routinely echoes the request body.
Legal/HR corpora end up with entity names and description fragments in
plaintext logs. Default `fail_on_embedding_error=False` means this is the
normal degraded path, not a rare crash.

**Root cause.** `embeddings/entities.py:123-127` logs `name` and `result`
directly, while extraction errors are deliberately sanitized.

**Smallest change.** Log the opaque entity `id` and `type(result).__name__` at
`WARNING`; keep name and exception text at `DEBUG` only. Reuse the existing
redaction helper used by extraction rather than adding a new one.

**Compatibility.** Log text only; no API, no exception type, no return value
changes. Operators keying on the old message lose the name at `WARNING` — the
intended change; note it in the changelog line for the release.

**Validation.** `caplog` test: a failing embedder produces a `WARNING` that
contains no entity name and no exception message, and a `DEBUG` record that
does.

**Acceptance.** No entity name or provider payload reaches `WARNING`+ logs from
the embedding path.

### Step 4 — bounded per-document transaction (F6-a)

**Failure scenario.** Entity write succeeds, relationship write fails
(connection drop, constraint violation). Chunks and entities persist, the
document is reported failed, and the retry relies on replacement semantics that
were never transactional.

**Root cause.** Every write in `write_graph_document` is its own
`execute_query`, hence its own transaction; there is no per-document commit
boundary.

**Smallest change.** Add a **private** transaction context manager to
`Neo4jGraphStore` and `MemgraphGraphStore` yielding an executor that runs
statements on one open transaction (commit on clean exit, rollback on
exception). `CypherGraphWriter.write_graph_document` uses it via
`getattr(store, "_transaction", None)` and runs state reads + all writes +
reconcile inside; absent hook → today's auto-commit path and the documented
caveat. Extraction, summarization, embedding and `finalize` stay outside the
transaction. No `GraphStore` protocol change, no new public parameter.

**Compatibility.** Public signatures and protocol unchanged; direct
`store.write_graph_document(...)` gets the transaction automatically; custom
`graph_writer=` is unaffected.

**Validation.** Injected mid-write exception → rollback, original error
surfaces, prior document state intact, zero partial nodes observable.
Live disposable backends: `tests/integration/test_writer_invariants.py` on both.

**Acceptance.** Any failure inside a document write leaves no partial document;
both backend invariant suites pass; public API unchanged.

## 3. After the builder: F5 (separate PR, unchanged plan)

**F5-a atomic community publish** — `community_pipeline` deletes every
`Community` (reports + report embeddings live on those nodes,
`graphdb/store_base.py:114-162`) before GDS/MAGE preflight, projection, or
Leiden (`communities/neo4j/detection.py:71-80`,
`communities/memgraph/detection.py:81-89`), and publishes over three
auto-commit queries. Fix: compute Leiden/membership/parent rows first, then
publish in one short transaction using Step 4's hook; preserve the primary
error if the projection drop also fails; keep `detect()`'s signature and
return shape. Validation: fake-store ordering tests (no delete before
preflight, no delete on injected Leiden failure), plus a live
"re-detect on a graph with no relationships → old reports still queryable".

**F5-b non-destructive `create_indexes()`** —
`graphdb/*/index_manager.py:44-55` drops all three retrieval indexes first and
swallows errors (`neo4j/index_manager.py:73-78`). Fix: ensure-semantics rather
than replace — absent → create, matching → no-op, mismatched → raise telling the
operator to use the existing `drop_indexes()`. Validation: two consecutive
`create_indexes()` emit zero `DROP`; mismatch raises without dropping.

**Acceptance (F5).** No path in `create_indexes()` issues `DROP`; any failure
before or during community publish leaves previous communities and reports
intact; no observable zero-community window.

## 4. Recommendation

Implement **Step 1 (F6-b)** next. It is the smallest complete change, confined
to `pipelines/writer.py`, needs no driver work, and closes the visibly open half
of F6 — a receipt that claims success for rows the database dropped. It also
makes Step 4 strictly safer, because the transaction then protects rows already
proven valid.

Steps 2 and 3 are independent 3-6 line riders; they can ship in the same PR or
immediately after with no ordering risk.

## 5. Verified vs assumed

Verified by direct reads: the four defect sites; `_stats_for`'s input counts;
`_rebuild_entity_rows` iterating `state.entities` (Step 1's stated non-change);
`Semaphore(extraction_concurrency)` with no guard at `:572`/`:859` versus the
`:278` clamp; the unredacted log line and the absence of a no-progress guard in
`embeddings/community_reports.py:47-100` (entity embedding already has one at
`embeddings/entities.py:71-80`); existing test doubles' `execute_query` behavior
(`tests/pipelines/writer_scenarios.py:17-25`, `47-100`).

Assumed, to confirm during implementation: Memgraph accepts the same explicit
transaction shape as Neo4j (Step 4); index introspection returns parseable
definition/dimension metadata on supported server versions (F5-b); Memgraph's
index ensure on an existing identical index is a no-op rather than an error.
F9's memory figures are an estimate from the code shape, not measured.
