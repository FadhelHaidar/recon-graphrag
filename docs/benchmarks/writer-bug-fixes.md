# Writer bug fixes: implementation record

This report records the execution of `plans/writer-bug-fixes.md`.
It covers the seven findings in `BUG_SCAN_REPORT.md`.

## 1. Baseline

Recorded before any implementation change.

| Item | Value |
| --- | --- |
| Original worktree | `/home/fadhelh/projects/recon-graphrag`, branch `perf/pipeline-logical-optimizations` |
| Original worktree HEAD | `13f6d35 perf(writer): specialize relationship state read predicates` |
| Implementation worktree | `/home/fadhelh/projects/recon-graphrag-bugscan`, branch `audit/bugscan-2026-11-13` |
| Implementation worktree HEAD | `660f0ee merge: community logical optimizations` |
| Divergence | The implementation worktree carries the same reader optimization as an uncommitted diff. `13f6d35` is that diff committed in the original worktree. No other source difference exists. |
| Python / pytest | 3.12.14 / 9.0.3 |
| Neo4j | 5.26.0 community, APOC 5.26.0, `bolt://localhost:7688` |
| Memgraph | `memgraph/memgraph-mage:latest`, `bolt://localhost:7689` |
| Vector index dimension | 768 for `entity-embeddings` and `community-report-embeddings` |

### Working tree state before implementation

Tracked modifications, all from the reader optimization:

- `.gitignore`
- `recon_graphrag/pipelines/writer.py`
- `tests/integration/test_writer_invariants.py`
- `tests/pipelines/test_writer_characterization.py`

Untracked, nonignored:

- `BUG_SCAN_REPORT.md`
- `docs/benchmarks/2026-11-13-pipeline-performance.md`
- `docs/benchmarks/writer-read-optimizations.md`
- `tests/benchmarks/` (5 files)
- `tests/integration/test_writer_read_performance.py`

`.env` was copied into the implementation worktree and stays gitignored.
Credentials are never printed or recorded here.

### Baseline checks

| Command | Result |
| --- | --- |
| `pytest -m "not integration"` | 1768 passed, 1 skipped, 148 deselected |
| `RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/test_writer_read_performance.py -m database` | passes (reader equivalence and profiling) |

### Pre-existing environment limitation

`tests/integration/test_writer_invariants.py` calls `store.create_indexes()` with no
dimension, so the manager expects 1536 while the disposable databases hold 768
vector indexes. 28 module-scope fixtures fail at setup with
`Index 'entity-embeddings' already exists with an incompatible definition
(dimensions=768, expected 1536)`.

This failure predates the work in this branch and is unrelated to the seven findings.
The plan forbids dropping global indexes. New backend regressions call
`create_indexes(embedding_dim=768)` in their own fixtures. Affected checks are
listed as skipped in section 6 instead of being reported as passes.

### Writer receipt keys and valid input

`CypherGraphWriter.write_graph_document` returns confirmed counts under the keys
`documents`, `chunks`, `entities`, `relationships`, `evidence_links`, and `claims`.
`tests/pipelines/test_writer_replacement.py::build_document` supplies the valid
input shape used by the regressions below.

## 2. B03: retained claim deleted after its source moved

**Trigger:** a replacement document keeps a claim ID but sources it from a
different chunk, and the previous source chunk is removed.

**Changed function:** `recon_graphrag/pipelines/writer.py::CypherGraphWriter._reconcile`.
Stale claims are now selected by identity only:

```text
stale_claims = [row["id"] for row in state.claims if row["id"] not in keep_claim_ids]
```

`removed_chunk_id_set` is gone because no other branch used it.
Claim deletion, chunk deletion, and the `_write_claims` sequence are unchanged.

**Regressions (deterministic):**

- `tests/pipelines/test_writer_replacement.py::test_retained_claim_survives_source_chunk_replacement`
- `tests/pipelines/test_writer_replacement.py::test_omitted_claim_is_deleted_during_replacement`
- `tests/pipelines/test_writer_replacement.py::test_retained_claim_survives_multiple_previous_source_rows`
- `tests/pipelines/test_writer_replacement.py::test_claim_replacement_does_not_touch_other_graph`

**Regressions (database, both backends):**

- `tests/integration/test_writer_bug_regressions.py::test_retained_claim_moves_to_the_replacement_chunk`
- `tests/integration/test_writer_bug_regressions.py::test_omitted_claim_is_deleted_and_retained_claim_kept`
- `tests/integration/test_writer_bug_regressions.py::test_claim_replacement_leaves_other_graph_claims_untouched`

The backend case asserts the claim node, its `SOURCED_FROM` target, and its
`SUBJECT_OF` target, so a surviving claim with a dangling source fails.

**Before/after:** before, the retained claim was deleted while the receipt still
reported `claims: 1`. After, the claim survives, points only at the replacement
chunk, and an omitted claim is still deleted. Both backends failed the backend
regression before the patch and pass after it.

**Recorded limitation, not fixed:** when a retained claim changes source but the
old chunk survives, both `SOURCED_FROM` edges remain. A Neo4j probe confirmed
`sources == ["doc:1:c0", "doc:1:c1"]` after such a replacement. The scan did not
establish this case, so it needs its own finding and fix.

## 3. B06: references accepted for chunks that replacement deletes

**Trigger:** a replacement document omits a chunk, and it submits a claim or
evidence link that names the omitted chunk.

**Changed function:** `recon_graphrag/pipelines/writer.py::CypherGraphWriter._validate_references`.
The accepted chunk set is now the submitted document only:

```text
chunk_ids = {chunk.id for chunk in graph_document.chunks}
```

The previous `chunk_ids.update(state.chunk_ids)` widened the set with chunks that
reconciliation deletes. Entity references keep the existing graph lookup, so
shared entities from other documents still resolve.

The missing-reference `ValueError` keeps its existing wording for entities. For
chunks it adds: "A chunk reference must name a chunk in this document version;
replacement removes omitted chunks."

**Regressions (deterministic):**

- `tests/pipelines/test_writer_replacement.py::test_claim_cannot_reference_removed_previous_chunk`
- `tests/pipelines/test_writer_replacement.py::test_evidence_cannot_reference_removed_previous_chunk`
- `tests/pipelines/test_writer_replacement.py::test_references_to_retained_and_new_chunks_are_valid`
- `tests/pipelines/test_writer_replacement.py::test_shared_entity_reference_remains_valid`
- `tests/pipelines/test_writer_replacement.py::test_empty_replacement_rejects_remaining_chunk_references`

**Regressions (database, both backends):**

- `tests/integration/test_writer_bug_regressions.py::test_removed_chunk_reference_is_rejected_before_mutation` (transaction and auto-commit paths)
- `tests/integration/test_writer_bug_regressions.py::test_evidence_reference_to_removed_chunk_is_rejected`
- `tests/integration/test_writer_bug_regressions.py::test_shared_entity_reference_is_accepted_on_backends`

The mutation checks use a full document snapshot (documents, chunks, entities,
claims) and compare it before and after the rejected write. The auto-commit path
uses `AutoCommitStore`, which hides `_transaction` from the writer.

**Before/after:** before, the claim or evidence link was written, then the chunk
was deleted, leaving a claim with no source chunk. After, `ValueError` names the
chunk before the first write. Neo4j and Memgraph keep identical state.

**Intentional behavior difference:** a caller that referenced a previous chunk
now gets an error instead of a dangling record. Documented in `docs/05-pipelines.md`.

## 4. B05: missing relationship ID crashes cleanup

**Trigger:** an affected edge loses its last contribution and its stored `r.id` is
missing, empty, or not a string, while another affected edge has a string ID.

**Changed functions in `recon_graphrag/pipelines/writer.py::CypherGraphWriter`:**

- `_write_document` runs two preflight steps in the documented order: submitted
  identity validation after reserved-key validation, and affected stored
  identity validation immediately after the state read. Both run per attempt.
- `_validate_incoming_relationship_ids` checks every submitted relationship.
- `_validate_stored_relationship_ids` checks the document's affected stored
  edges only, so unrelated graphs and documents are not scanned.
- `_require_relationship_id` holds the single rule: a nonempty string. It does
  not require UUID formatting.

The error names the graph, source, type, target, and the invalid field. It tells
the caller to repair or reimport the affected data, and states that the writer
does not assign relationship IDs. Cleanup still sorts string IDs, which is now
safe.

**Regressions (deterministic), `tests/pipelines/test_writer_relationship_identity.py`:**

- `test_missing_stored_relationship_id_fails_before_mutation`
- `test_mixed_relationship_id_types_fail_before_sorting`
- `test_empty_incoming_relationship_id_is_rejected`
- `test_non_string_incoming_relationship_id_is_rejected`
- `test_custom_string_relationship_id_remains_supported`
- `test_unrelated_malformed_relationship_is_not_scanned`
- `test_malformed_relationship_preflight_is_retry_local`

The retry-local case aborts attempt one with a retryable error, corrupts the
stored ID between attempts, and requires attempt two to reject its own fresh
state.

**Regressions (database, both backends), `tests/integration/test_writer_bug_regressions.py`:**

- `test_missing_stored_relationship_id_fails_before_mutation`
- `test_empty_incoming_relationship_id_is_rejected`
- `test_custom_string_relationship_id_round_trips`

**Before/after:** before, the same input raised
`TypeError: '<' not supported between instances of 'str' and 'NoneType'` after
writes had started. A throwaway probe reproduced exactly that message. After,
`ValueError` names the edge before the first mutation, and both backends keep the
previous state unchanged.

**Recorded limitation, not fixed:** the writer still does not repair or assign
relationship IDs. Imported data with malformed IDs needs an explicit migration
or reimport.

## 5. B01: child records write outside the document graph

**Trigger:** a caller builds a `GraphDocument` whose document uses graph `tenant-a`
and a child record keeps the default graph `entity-graph` or another graph.

**Changed functions in `recon_graphrag/pipelines/writer.py::CypherGraphWriter`:**

- `_validate_document_scope` runs next to reserved-key validation, before any
  state query and on every retry attempt.
- It requires the document graph name for chunks, entities, relationships,
  evidence links, and claims.
- It requires `ChunkRecord.document_id == document.id`.
- It requires `ClaimRecord.source.document_id == document.id`.
- `_require_document_graph` holds the graph comparison and the error text.
- Source chunk membership keeps the B06 rule from `_validate_references`.

The writer rejects a mismatch instead of rewriting the caller's graph names.
Shared entities from other documents remain valid because the rule is the graph
boundary, not exclusive entity ownership. Dataclass signatures and the custom
writer interface are unchanged.

**Regressions (deterministic), `tests/pipelines/test_writer_document_scope.py`:**

- `test_foreign_graph_name_child_writes_are_rejected` (6 cases: chunk, entity,
  relationship, evidence, claim, default graph name)
- `test_chunk_from_another_document_is_rejected`
- `test_claim_source_document_must_match`
- `test_shared_entity_reference_keeps_working`
- `test_scoped_document_is_valid`
- `test_scope_violation_is_checked_on_every_attempt`

Each rejection asserts that the store received zero queries.

**Regressions (database, both backends), `tests/integration/test_writer_bug_regressions.py`:**

- `test_foreign_graph_entity_is_rejected_before_mutation`
- `test_foreign_graph_preserves_existing_graph`

**Before/after:** before, the foreign entity was written into the foreign graph
while the document was replaced in its own graph. After, `ValueError` names the
record kind, both graph names, and the parent document. Neither graph changes.

**Intentional behavior difference:** a caller that mixed graph names now gets an
error. The SDK assembler and structured ingestion already propagate one graph
name, so their output is unaffected. Documented in `docs/05-pipelines.md`.

**Compatibility check:** `tests/pipelines`, `tests/graphdb`, `tests/extraction`,
and `tests/compatibility` pass without fixture changes.

## 6. B04: entity identity mixed with the domain label

**Trigger:** an entity identity whose stored node lacks the incoming domain
label, or carries a different one. Example: a node created by Cypher import
without `:Person`, then a `Person` document write for the same ID.

**Reproduction (both backends):** the old query
`MERGE (e:__Entity__:{label} {id: row.id, graph_name: row.graph_name})` matched
only nodes that already carried the label. Writing the same identity with a new
label created a second node. Without a uniqueness constraint both nodes
survived; with one, the write failed.

**Changed functions in `recon_graphrag/pipelines/writer.py::CypherGraphWriter`:**

- `_write_entities` now merges on
  `MERGE (e:__Entity__ {id: row.id, graph_name: row.graph_name})` and adds the
  escaped domain label with `SET e:{label},`.
- The remaining property and provenance `SET` clauses are unchanged, including
  `e.uid`, `e.type`, description contributions, and summary status.
- `_read_document_state` keeps the entity rows it read and passes them to the new
  `_reject_ambiguous_entity_identities`.
- `_reject_ambiguous_entity_identities` counts rows per affected ID. Two or more
  nodes for one identity raise `ValueError` before mutation, name the graph and
  the affected IDs with their node counts, and state that the writer does not
  merge nodes. The check covers affected IDs only; it never audits the graph.
- `entity_ids` now keeps only string values, so a null read field cannot reach
  the ambiguity check.

**Regressions (deterministic), `tests/pipelines/test_writer_entities.py`:**

- `test_entity_identity_merge_excludes_the_domain_label`
- `test_entity_write_reuses_untyped_existing_identity`
- `test_entity_type_partition_preserves_repeated_input_rows`
- `test_entity_write_rejects_ambiguous_existing_identity`
- `test_entity_label_is_escaped_without_changing_identity`

**Regressions (database, both backends), `tests/integration/test_writer_bug_regressions.py`:**

- `test_entity_write_reuses_untyped_existing_identity` compares `id(e)` before
  and after the write to prove node reuse.
- `test_entity_write_adds_label_without_duplicate_identity`
- `test_entity_label_change_preserves_edges_and_provenance` keeps one
  relationship row and one evidence row.
- `test_entity_write_rejects_ambiguous_existing_identity` keeps both seeded
  nodes and writes no chunk.
- `test_entity_write_with_escaped_label_reuses_identity`
- `test_concurrent_identity_writes_keep_one_node` repeats the shared-identity
  uniqueness case that the retry loop covers; one node remains.

**Updated characterization:** `tests/pipelines/test_writer_characterization.py::test_entity_type_partition_preserves_duplicate_and_cross_type_rows`
now asserts the identity merge and the `SET e:{label}` form. Label rows, row
partitioning, query count, and the per-group receipt convention are unchanged.

**Intentional behavior differences:**

- One stored node can now carry several domain labels.
- `e.type` keeps the last written type group for a repeated ID, as before per
  group. Multi-label precedence is unchanged; only the node count changed.
- A graph with pre-existing duplicate identities now fails the affected write
  with a diagnostic. Repair stays a separate migration.

**Limits:** existing duplicate nodes are not merged. The report records the
diagnostic instead of an automatic repair.

## 7. B07: replaced chunk text kept a stale vector

**Trigger:** a document replacement that keeps a chunk ID and writes new text.
Chunk IDs are positional (`recon_graphrag/extraction/chunking.py:85-89`) and a
document keeps its identity (`recon_graphrag/pipelines/graphrag_pipeline.py:1145-1152`),
so the same chunk ID commonly carries different text after a replacement.

**Reproduction (fake store and both backends):** the writer updated `text` and
kept `embedding`. The chunk selector
(`recon_graphrag/graphdb/neo4j/store.py` and
`recon_graphrag/graphdb/memgraph/store.py::get_unembedded_chunks`) returned only
chunks with a null vector or a changed model, so the stale vector survived. Local
and DRIFT source ranking then scored the chunk against text it no longer had
(`recon_graphrag/retrieval/_queries.py::_SOURCE_SELECT_RANKED`). The observed
state was `text = "beta"` with the vector built from `"alpha"`.

**Changed functions:**

- `recon_graphrag/pipelines/writer.py::CypherGraphWriter._chunk_rows` adds
  `embedding_input_fingerprint = text_fingerprint(chunk.text)`.
- `CypherGraphWriter._write_chunks` reads the previous text in the same
  statement (`WITH c, d, row, c.text AS previous_text`) and keeps the vector and
  both vector fingerprints only when `previous_text = row.text`. A changed text
  clears `embedding`, `embedding_content_fingerprint`, and
  `embedding_model_fingerprint` atomically with the text write.
- `get_unembedded_chunks` in both stores also returns a chunk when its content
  fingerprint is missing, or when the content and input fingerprints disagree.
  Legacy chunk vectors therefore refresh once.
- `upsert_vectors` in both stores reads `n.text` for the legacy snapshot. It
  certifies a legacy chunk vector only when `text_fingerprint(n.text)` equals the
  supplied fingerprint, through the new `row.legacy_chunk_text` guard. The entity
  legacy path is unchanged.
- `recon_graphrag/embeddings/chunks.py::ChunkEmbedder.embed_chunks` passes
  `content_fingerprints` for the exact text it sent to the provider, only for
  successful results. Its docstring no longer claims that chunk text is
  immutable.
- `_SOURCE_SELECT_RANKED` scores 0.0 when both fingerprints exist and disagree,
  so a stale vector cannot rank. Legacy vectors without fingerprints keep their
  current scoring, so ranking does not shift before the refresh.

**Regressions (deterministic):**

- `tests/pipelines/test_writer_chunk_vectors.py`:
  `test_changed_chunk_text_invalidates_stored_vector`,
  `test_unchanged_chunk_text_keeps_stored_vector`,
  `test_only_the_changed_chunk_loses_its_vector`,
  `test_new_chunk_row_carries_the_text_fingerprint`,
  `test_chunk_write_query_clears_vector_only_for_changed_text`.
- `tests/embeddings/test_chunk_embedder.py`: updated
  `test_embed_chunks_stores_normalized_vectors` and
  `test_embed_chunks_keeps_successes_when_one_chunk_fails` (fingerprints stay
  aligned with the successful ids), plus
  `test_embed_chunks_fingerprints_the_text_it_embedded`.
- `tests/retrieval/test_retrieval_limits.py::test_ranked_source_selection_ignores_a_vector_built_from_other_text`
  pins the ranking guard in the local and DRIFT templates.
- `tests/pipelines/test_graph_builder.py::test_embed_chunks_flag_stores_normalized_chunk_vectors`
  now asserts the passed fingerprint.

**Regressions (database, both backends), `tests/integration/test_writer_bug_regressions.py`:**

- `test_changed_chunk_text_clears_vector_and_certification`
- `test_unchanged_chunk_text_keeps_the_certified_vector`
- `test_legacy_uncertified_chunk_is_refreshed_once`
- `test_chunk_model_change_requires_refresh`
- `test_stale_chunk_vector_cannot_overwrite_new_text` writes a vector built from
  the previous text after the replacement and proves the write is rejected.
- `test_chunk_replacement_embeds_only_the_changed_chunk`

**Observed defect during implementation:** the first version of the new
`_write_chunks` statement dropped `d` through `WITH`. Cypher then created a new
label-less node for `MERGE (c)-[:PART_OF]->(d)`, and the next write could not read
the document's chunks. The regression
`test_retained_claim_moves_to_the_replacement_chunk` failed on both backends and
the fix kept `d` in the projection. The statement now returns the same
`count(*) AS written` receipt.

**Intentional behavior differences:**

- A chunk whose text changes loses its vector immediately, so a build that runs
  `embed_chunks=False` leaves that chunk unembedded until the next embedding pass.
  Ranking then falls back to id order for that chunk.
- A legacy chunk vector is refreshed once, because it has no text fingerprint.
- A stored vector whose fingerprints disagree cannot rank until it is refreshed.

**Limits:** the ranking guard applies only when both fingerprints exist. The
writer never leaves a mismatched state, so the guard covers external writes and
interrupted runs rather than normal replacements.

## 8. B02: parallel edges collapsed, renamed, and adopted across graph scope

**Trigger.** `MERGE (source)-[r:{type}]->(target)` matched every parallel edge
between the two endpoints. The following `SET` wrote all matches, so one write
renamed every parallel edge to the incoming ID and overwrote its provenance. The
`MERGE` had no `graph_name` predicate, so a scoped write could adopt a foreign or
unscoped edge and give it its own graph name. The state read also only found
edges by incoming ID or by old chunk ownership, so a stored edge whose ID differed
from the incoming ID was never seen, and its stored ID was replaced.

**Changed functions.**

- `_read_relationship_targets`: new narrow read of every stored edge for the
  submitted logical keys. It matches endpoints by identity and a fixed type, and
  returns the full contribution projection plus `r.graph_name`. The parameter name
  is `$relationship_keys`, so consumers can distinguish it from the write.
- `_reject_duplicate_relationship_identities`: counts same-scope edges per key and
  stops before mutation when one key has more than one.
- `_adopt_key_relationships`: adds same-scope rows from that read to the document
  state, so the stored identity and its contributions survive. Foreign and
  unscoped rows are never adopted.
- `_reject_relationship_id_collisions`: stops when one stored ID identifies a
  different logical key, because rebuild and cleanup select edges by ID.
- `_relationship_rows`: emits `stored.get("id") or rel.id`.
- `_write_relationships`: `MERGE (source)-[r:{type} {graph_name: row.graph_name}]->(target)`.
- `_relationship_rows` and `_write_document`: adoption order is duplicate check,
  adoption, stored-ID validation, then ID-collision check.
- `tests/pipelines/writer_scenarios.py`: dispatches `UNWIND $relationship_keys AS
  row`, and the relationship label parser tolerates the property map.

**Deterministic regressions** in `tests/pipelines/test_writer_relationship_identity.py`:
`test_relationship_rows_keep_the_stored_id_for_the_same_logical_edge`,
`test_relationship_rows_keep_the_incoming_id_for_a_new_logical_edge`,
`test_duplicate_stored_relationships_are_rejected_before_mutation`,
`test_stored_id_for_an_incompatible_logical_key_is_rejected`.

**Database regressions** in `tests/integration/test_writer_bug_regressions.py`, on
Neo4j and Memgraph: `test_foreign_relationship_is_not_adopted_by_local_write`,
`test_unscoped_relationship_is_not_adopted_by_local_write`,
`test_duplicate_logical_relationships_fail_before_mutation`,
`test_existing_logical_relationship_keeps_stored_id`,
`test_relationship_id_collision_is_rejected`,
`test_relationship_replacement_preserves_other_document_contributions`,
`test_omitted_relationship_is_found_by_previous_chunks`,
`test_relationship_self_loop_and_reverse_direction_are_distinct`,
`test_resolve_then_replace_preserves_logical_edge_identity`,
`test_concurrent_logical_edge_creation_does_not_duplicate`,
`test_concurrent_relationship_contributions_keep_one_edge`.

**Before and after.** Before the fix, both backends adopted a foreign edge and
renamed it to the local ID, gave an unscoped edge the local graph name, applied
`SET` to both duplicate edges, and replaced a stored ID with the incoming one.
After the fix, each case either keeps the stored state untouched or stops before
the first write with an actionable error.

**Concurrency measurement.** Two concurrent documents wrote one logical edge three
times per backend:

| Backend | Outcomes | Surviving contributions |
| --- | --- | --- |
| Neo4j | both commits succeed | stored chunk plus one document chunk; the other document contribution is lost |
| Memgraph | one transient conflict, then the built-in retry | stored chunk plus both document chunks |

The baseline writer loses a contribution the same way on Neo4j, and it also drops
the pre-existing stored chunk, because its ownership-only read never found that
edge. This fix therefore narrows the loss but does not close it: closing it needs
per-key serialization, which the plan excludes without a revision. A duplicate
created by a concurrent create is never silently combined; the next write to the
same key stops with the ambiguity error (covered by
`test_concurrent_logical_edge_creation_does_not_duplicate`).

**Intentional behavior differences:**

- A stored edge keeps its ID when a document resubmits the logical key under a new
  ID. Contributions join the existing edge.
- A graph with two stored edges for one key, or one ID on a different key, now
  fails before writing instead of silently rewriting data.
- Each write adds one narrow key read per relationship type.

**Limits:** the key read covers submitted keys only, so an omitted edge is still
found through old chunk ownership rather than by key. Duplicate edges created
outside the writer stay rejected until an operator deduplicates them.

## 9. Independent review and remediation

A read-only reviewer examined the seven fixes against the plan and the source.
The review reported no defect in document scope, chunk-reference validation, target lookup, or chunk freshness.

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| F1 | P1 | Rebuild and cleanup select edges by `r.id` alone. Two keys that share one stored ID could rename or delete the retained edge. | Fixed. `_read_relationship_ids` reads every graph-scoped edge that carries an affected ID, and `_reject_relationship_id_collisions` rejects an ID that spans more than one logical key. |
| F2 | P1 | The entity embedding text took the first non-`__Entity__` label. A retained older label could change the fingerprint, and both stores then rejected the vector. | Fixed. `entity_embedding_text` and `entity_embedding_fingerprint` accept `entity_type`. The embedder passes the record type, and both stores read and pass `n.type`. |
| F3 | P2 | The duplicate check counted only submitted keys, so an omitted key with two stored edges passed. | Fixed. The check now counts stored rows and submitted rows together, after deduplication. |
| F4 | P2 | The legacy upsert treated any node with a `text` property as a chunk. An entity may keep its own `text`, so its vector was never certified. | Fixed. The stores decide by the `Chunk` label and pass the recorded type for the entity fingerprint. |
| F5 | P3 | Three cases had no regression: repeated-key aggregation, two new keys with one incoming ID, and an adopted stored-ID collision outside previous ownership. | Fixed. The cases in the table below cover them. |

Regressions added during remediation:

| Case | Test |
| --- | --- |
| Shared stored ID blocks a replacement | `tests/integration/test_writer_bug_regressions.py::test_shared_stored_id_rejects_replacement_before_mutation`, `tests/pipelines/test_writer_relationship_identity.py::test_stored_id_for_an_incompatible_logical_key_is_rejected` |
| Omitted key with two stored edges | `test_omitted_key_duplicate_is_rejected_before_mutation`, `test_omitted_key_duplicate_is_rejected_on_both_paths` (transaction and auto-commit) |
| Two new keys with one incoming ID | `test_two_new_keys_sharing_one_incoming_id_are_rejected` |
| Repeated incoming key | `test_repeated_incoming_key_writes_one_edge` |
| Changed domain label keeps certification | `test_changed_entity_label_keeps_embedding_certification` (both backends) |
| Recorded type outranks label order | `tests/graphdb/test_vector_persistence.py::test_legacy_entity_fingerprint_uses_the_recorded_type` |
| Entity with its own `text` property | `test_legacy_entity_with_own_text_property_certifies_entity_fingerprint` |
| Legacy chunk certification | `test_legacy_chunk_attaches_the_text_it_was_built_from` |

Two further defects were found while writing the regressions.
`_read_relationship_targets` inflated the edge count when one document repeated a logical key, because each repeated request returned the same edge.
The writer now requests each `(source_id, target_id)` key once per type.
The ambiguity check also ignored stored-only keys, which F3 covers.

### Second review

A second read-only review of the same diff confirmed the document scope, the chunk invalidation, the retrieval guard, and the ownership conditions of the relationship queries.
It reported five further defects.

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| S1 | P1 | The affected-ID read queried only the relationship types the document submits. An adopted ID could also belong to an edge of another type, which rebuild and cleanup reach by ID alone. | Fixed. `_read_relationship_ids` runs one query over every relationship type in the graph. |
| S2 | P1 | Two physical edges with the same ID and the same key collapsed in the duplicate check, so a later `MERGE` could match both. | Fixed. `_reject_duplicate_relationship_identities` counts physical rows per key and takes the larger of the stored and submitted counts. `_reject_relationship_id_collisions` also rejects any ID that the affected-ID read returns more than once, which covers an omitted key whose duplicate another document owns. |
| S3 | P1 | The entity state read did not return `e.type`, so `_rebuild_entity_rows` fingerprinted reconciled descriptions from the first domain label. | Fixed. The state read returns `e.type`, and the rebuild passes `entity_type`. |
| S4 | P2 | An unscoped edge of the same type and key inflated the duplicate count, so the writer rejected a valid graph-owned write. | Fixed. Only rows whose relationship `graph_name` equals the document graph count. |
| S5 | P2 | Reference-only endpoints skipped the entity ambiguity check. Two stored nodes for one identity produced two edges for one submitted key. | Fixed. `_reject_reference_entity_ambiguity` checks relationship endpoints, evidence entity IDs, and claim entity IDs that the state read does not cover. |

Regressions for the second review:

| Case | Test |
| --- | --- |
| Shared ID across two relationship types | `tests/integration/test_writer_bug_regressions.py::test_shared_id_across_relationship_types_fails_before_mutation` |
| Two physical edges with one ID and one key | `test_two_physical_edges_sharing_id_and_key_are_rejected` |
| Owned duplicate with a shared ID and an omitted key | `test_owned_duplicate_with_shared_id_is_rejected_before_mutation` |
| Omitted key whose same-ID duplicate another document owns | `test_omitted_key_duplicate_owned_by_another_document_is_rejected` |
| Unscoped parallel edge does not block an owned write | `test_unscoped_parallel_edge_does_not_block_owned_write` |
| Reference-only ambiguous endpoint | `test_reference_only_endpoints_are_checked_for_ambiguous_entities` |
| Rebuilt fingerprint uses the recorded type | `tests/pipelines/test_writer_entities.py::test_rebuilt_fingerprint_uses_the_recorded_entity_type` |

The duplicate check now counts physical edges, so a repeated incoming key still writes one edge (`test_repeated_incoming_key_writes_one_edge`).

A focused verification review of the remediation confirmed S1, S3, S4, and S5.
It found one residual S2 path: an omitted key whose same-ID duplicate belongs to another document.
`_reject_relationship_id_collisions` now rejects any ID that the affected-ID read returns more than once.
The regression for that path is in the table above.

## 10. Finding ledger

| ID | Changed functions | Deterministic regressions | Database regressions | Status |
| --- | --- | --- | --- | --- |
| B01 | `_validate_document_scope`, `_require_document_graph` | `tests/pipelines/test_writer_document_scope.py` (11 cases) | `test_foreign_graph_entity_is_rejected_before_mutation`, `test_foreign_graph_preserves_existing_graph` | Fixed |
| B02 | `_read_relationship_targets`, `_reject_duplicate_relationship_identities`, `_adopt_key_relationships`, `_reject_relationship_id_collisions`, `_relationship_rows`, `_write_relationships` | `tests/pipelines/test_writer_relationship_identity.py` (B02 section, 4 cases) | 11 cases, see section 8 | Fixed; the lost-update window is narrowed, not closed |
| B03 | `_reconcile` | `tests/pipelines/test_writer_replacement.py` (4 cases) | `test_retained_claim_moves_to_the_replacement_chunk`, `test_omitted_claim_is_deleted_and_retained_claim_kept`, `test_claim_replacement_leaves_other_graph_claims_untouched` | Fixed |
| B04 | `_write_entities`, `_reject_ambiguous_entity_identities` | `tests/pipelines/test_writer_entities.py` (5 cases) | 6 cases, see section 6 | Fixed |
| B05 | `_validate_incoming_relationship_ids`, `_require_relationship_id`, `_validate_stored_relationship_ids` | `tests/pipelines/test_writer_relationship_identity.py` (7 cases) | `test_missing_stored_relationship_id_fails_before_mutation`, `test_empty_incoming_relationship_id_is_rejected`, `test_custom_string_relationship_id_round_trips` | Fixed; repair is excluded |
| B06 | `_validate_references` | `tests/pipelines/test_writer_replacement.py` (5 cases) | `test_removed_chunk_reference_is_rejected_before_mutation`, `test_evidence_reference_to_removed_chunk_is_rejected`, `test_shared_entity_reference_is_accepted_on_backends` | Fixed |
| B07 | `_chunk_rows`, `_write_chunks`, both stores `get_unembedded_chunks` and `upsert_vectors`, `ChunkEmbedder.embed_chunks`, `_SOURCE_SELECT_RANKED` | `tests/pipelines/test_writer_chunk_vectors.py` (5 cases), `tests/embeddings/test_chunk_embedder.py`, `tests/retrieval/test_retrieval_limits.py` | 6 cases, see section 7 | Fixed |

## 11. Executed checks

| Command | Result |
| --- | --- |
| `pytest -m "not integration"` | 1820 passed, 1 skipped, 240 deselected (24.9 s) |
| `pytest -m database` (Neo4j and Memgraph) | 189 passed, 4 skipped, 2 failed, 28 errors |
| `pytest tests/integration/test_writer_bug_regressions.py -m database` | 92 passed on both backends |
| `pytest tests/integration/test_writer_bug_regressions.py -m database -k label_keeps` | 2 passed, one per backend |
| `pytest tests/graphdb -m "not integration"` | 227 passed, 1 skipped |
| `pytest tests/integration/test_writer_read_performance.py -m database` | 6 passed on both backends |
| `pyright` on every changed production and test file | 0 errors, 0 warnings |

The two failures are the pre-existing `test_neo4j_uid_contract_is_written_backfilled_and_enforced` and its Memgraph twin.
The 28 errors are the pre-existing `store.create_indexes()` default of 1536 dimensions against the configured 768-dimension database.
Both limits exist on the untouched original worktree and are recorded in section 1.

Skipped checks: live LLM and provider embedding runs, the `e2e` tier, and `uv build`.
No packaging or dependency file changed, so the build is out of scope.
No credential, provider, or production database was used.

## 12. Concurrency timelines

Relationship contributions, two threads, three rounds per backend:

| Backend | Observed chunk contribution list | Outcome |
| --- | --- | --- |
| Neo4j | `["doc:0:c0", "seed-chunk"]` | Both transactions commit, one contribution is lost |
| Memgraph | `["doc:0:c0", "doc:1:c0", "seed-chunk"]` | First attempt raises `TransientError ... Cannot resolve conflicting transactions`; the retry keeps both |
| Neo4j, baseline writer | `["doc:0:c0"]` | Baseline also drops the pre-seeded chunk, which this fix keeps |

The fix narrows the lost-update window because the writer now reads the stored edge by key before writing.
It does not close the window, because that needs per-key serialization, which the plan excludes.

Concurrent relationship creation, two threads, raises `ValueError` matching `ambiguous` when both edges exist.
The write never merges them.

Concurrent chunk certification is covered by `test_stale_chunk_vector_cannot_overwrite_new_text`.
The store rejects a fingerprint that no longer matches the stored text, so the newer text wins.

## 13. Migration exclusions

This change diagnoses three malformed states and refuses to repair them.
Each repair stays a separate, operator-run migration:

- A relationship without an `id` property, or with a non-string `id`, fails before mutation.
  The writer does not assign an ID and does not migrate stored rows.
- Two entity nodes with one `(graph_name, id)` identity fail with the ambiguity error.
  The writer does not merge nodes or move edges.
- Two relationship edges with one logical key, or with one ID, fail with the ambiguity error.
  The writer does not merge edges, combine contributions, or delete a duplicate.

Operators repair these states with an explicit scripted migration and then retry the write.

## 14. Added costs

- B02 adds two narrow relationship reads on each write.
  The key read matches endpoints and type.
  The ID read matches graph scope and ID, whatever the endpoints and type, because rebuild and cleanup select by ID alone.
  Neither read returns rows for another graph.
  The key read adds one round trip per submitted relationship type; the ID read adds one round trip in total.
- The entity ambiguity preflight adds one narrow read when a relationship, evidence link, or claim names an entity identity that the writer does not otherwise read.
  It returns only IDs.
- B07 re-embeds each legacy chunk once, because a chunk without a stored input fingerprint cannot prove freshness.
  After the first refresh, the stored fingerprint prevents repeat work.
- B07 adds three `CASE` expressions to the chunk write, so the writer clears a vector and its fingerprints in the same statement.
- No index definition changed, so the change adds no index maintenance cost.

## 15. Intentional behavior differences

Each item below changes observable behavior on purpose.
A write that relied on the old behavior now fails before mutation or keeps different data.

- A child record must use the document graph name. The writer rejects a chunk, entity, relationship, evidence link, or claim in another graph.
- A claim or evidence link must reference a chunk in the current document version. It can no longer reference a chunk that the replacement deletes.
- A relationship must carry a nonempty string `id`. The writer never assigns one.
- Entity identity is `(graph_name, id)`. A domain type is an added label, so a type change reuses the node. Several stored nodes for one identity stop the write.
- Chunk text that changes clears the vector and its fingerprints in the same write. A legacy chunk without a fingerprint is embedded once.
- Relationship identity for replacement is `(graph_name, source_id, type, target_id)`. An existing edge keeps its stored ID. A second stored edge for one key, or one ID on two edges, stops the write.
- The writer never adopts, renames, or deletes an edge whose `graph_name` differs or is missing.
- Every rejection listed here raises `ValueError` before the first mutation. A transaction rollback or a retry cannot leave a partial repair.

## 16. Cross-component plumbing follow-up

Four downstream gaps required follow-up changes:

- Resolution and local or DRIFT display now prefer recorded entity types over retained labels.
  Legacy entities fall back to the first domain label, as entity embedding does.
- Resolver preflight now checks every incident relationship before either backend mutates any candidate group.
  Foreign or unscoped domain edges stop resolution, including collisions at the canonical endpoint.
  Known structural roles allow missing relationship ownership only when endpoint labels, direction, and graph scope match.
  This preserves legacy evidence, claim subjects, and community memberships without adopting foreign edges.
  Local, DRIFT, and community retrieval require explicit graph ownership for domain edges.
- Legacy entity vector guards now compare the snapshot type with a null-safe predicate in the write statement.
  A type change between snapshot read and vector write cannot certify an old-type vector.
  Chunk guards and unguarded legacy vector calls retain their behavior.
- Retained claims replace scoped subject and source links, even when both old endpoints survive.
  Omitted claims, current-version chunk validation, and write receipts retain their behavior.

The resolver adds one incident-scope read per candidate group.
It does not repair malformed legacy domain edges.
The preflight does not serialize concurrent external graph mutations; existing resolution transaction boundaries remain unchanged.
Source inspection supplied verification because the worktree had no available graph index coverage.
Tests use disposable Neo4j and Memgraph databases without AI calls.

Follow-up validation:

- Mandatory deterministic suite: **1851 passed, 1 skipped**, with 276 integration cases deselected in the final rerun.
- Writer regression suite on both disposable databases: **128 passed**, including 36 new plumbing cases.
- Pyright on the affected implementation modules and new regression modules: **0 errors, 0 warnings**.
- Live AI-provider and end-to-end smoke tests were not run.
- No packaging or dependency files changed, so `uv build` was not required.

Independent review found two claim-write gaps in the first follow-up implementation.
An unscoped `MERGE` could reuse a foreign link at a desired endpoint after removing the local link.
Both structural-edge `MERGE` patterns now require the document graph name.
The regression preserves foreign parallel links while checking local ownership, repeated replacements, and receipt counts.

The review also found that identical claim inputs collapsed during deletion fan-out deduplication.
Each input row now carries an ordinal, so deduplication preserves each submitted occurrence.
The duplicate-input regression checks a receipt of two and one local edge per role on both backends.
These regressions are `test_claim_replacement_does_not_adopt_foreign_links_at_current_endpoints` and `test_identical_claim_inputs_preserve_receipt_multiplicity`.

A focused read-only verification review confirmed both claim corrections and found no further issues in their immediate effects.
The reviewer inspected sources; the parent ran the test suites listed above.

## 17. Live synthetic E2E validation

After the plumbing fixes, the user requested live tests.
The synthetic E2E suite used the configured OpenAI-compatible LLM and embedder with both disposable databases.

```bash
RUN_E2E_INTEGRATION_TESTS=1 EMBEDDING_DIM=768 uv run pytest tests/integration/test_smoke_full.py -m e2e -q
```

Result: **2 passed**, 273 deprecation warnings, in **421.79 seconds**.
Both Neo4j and Memgraph completed extraction, graph building, hybrid resolution, embeddings, community reports, and local, global, and DRIFT retrieval.
The tests checked graph evidence, citation fields, and nonempty answers.
These checks do not measure answer accuracy or exhaustively validate every concurrent-write scenario.
Fixture cleanup completed; a separate count confirmed zero remaining nodes in both synthetic E2E graph scopes.
Global indexes were not dropped.
The warnings concern the deprecated Neo4j driver notification setting.
The local command used the existing shared virtual environment with `uv run --no-sync`.
Log: `/tmp/recon-plumbing-live-e2e.log`.

## 18. Stage inspection (live providers, both databases)

The user requested the stage-inspection suite after the synthetic E2E pass.
Command (live LLM and embedder, disposable databases, cleanup verified afterward):

```bash
RUN_E2E_INTEGRATION_TESTS=1 EMBEDDING_DIM=768 uv run pytest tests/integration/test_stage_inspection.py -m e2e -s -v
```

Result: **10 passed** in **409.08 seconds** (5 stages x 2 backends).
Zero remaining nodes in `stage-scenario-neo4j` and `stage-scenario-memgraph` afterward.
Log: `/tmp/recon-stage-inspection.log`.

- Stage 1 (extraction): both backends wrote 9 isolated entities with valid names and descriptions, all endpoints resolvable, no dangling relationships. Neo4j extracted one extra `OPERATES Northwind Health -> Billing API` edge from LLM variance. The `final graph` and `build validation` counts are database-global (`validate_graph_build` counts unscoped nodes), so Neo4j shows 818 entities / 22288 relationships from leftover disposable data while Memgraph shows 9 / 18. The per-graph `entity_rows` (9 entities each) are the correct isolation signal and match.
- Stage 2 (resolution): unresolved 6 rows, normalized and fuzzy each 3 nodes (`alicechen`, `achen`, `bobortiz` kept separate, so "A. Chen" correctly stays apart). Merged Alice node keeps 2+ descriptions. Deterministic, no LLM merge calls.
- Stage 3 (embeddings): 9/9 entities and 2/2 chunks carry vectors on Memgraph; 10/10 and 2/2 on Neo4j (the extra entity is another extraction-variance node). Similarity search for the investigator query returns 3 descending-score matches on both backends.
- Stage 4 (communities): Memgraph 7 communities (several singletons) with 7 reports; Neo4j 4 communities with 4 reports. Every report is nonempty and carries a vector. Fragmentation differs by backend community detection, which is expected variance, not an assertion failure.
- Stage 5 (retrieval): local answers on both backends name Bob Ortiz as investigator and credential rotation plus multi-factor authentication as mitigations, with citations resolving to both stored chunks. Global answers list the involved systems and people. Static level selection uses 5 reports on Memgraph and 3 on Neo4j; dynamic selection picks level-0 on Memgraph and level-1 on Neo4j. Both satisfy the stored-report and metadata invariants.

Limits: the suite asserts structural invariants only, not answer accuracy. Entity, edge, community-count, and level-selection differences between runs and backends are live-LLM and detection variance. Token cost per build is roughly 6-7k tokens plus community and retrieval calls (see per-stage usage in the log). The 422 warnings are the known deprecated Neo4j driver notification setting.
