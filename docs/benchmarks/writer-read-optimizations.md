# Writer read optimizations: equivalence and performance record

Planning evidence for `plans/writer-read-optimizations.md`.
Branch: `perf/pipeline-logical-optimizations`, baseline `660f0ee4e4b05cbf62e05c4d318ce5c731a811fe`.
Scope: `CypherGraphWriter._read_document_state` in `recon_graphrag/pipelines/writer.py`.
Constraint: unchanged result quality, public SDK contracts, and persisted data shapes.

## Baseline

- Working tree clean except untracked `docs/benchmarks/` and `tests/benchmarks/`.
- Baseline focused writer suite green:
  `tests/pipelines/test_writer_characterization.py`, `test_writer_replacement.py`,
  `test_writer_provenance.py`, `test_writer_receipts.py`,
  `tests/graphdb/test_writer_transaction.py` — 71 passed in 1.06s.
- Current pinned behavior:
  - `_read_document_state` always executes the evidence and claim queries; both traverse
    `Document ← PART_OF — Chunk` and cannot return rows when the previous-chunk query returns none.
  - The relationship-state read already skips when both `rel_ids` and `chunk_ids` are empty.
  - `StatefulFakeGraphStore` (`tests/pipelines/writer_scenarios.py`) answers exact query shapes;
    its relationship projection omits `properties` and its edge store cannot represent all
    parallel-edge cases. Real-backend or row-returning stubs cover those cases.

## Optimization 1: staged document-owned reads

Status: rejected by the step 2 concurrency gate. `_read_document_state` keeps the
baseline evidence and claim reads.

### Gate method

A test-local staged reader (candidate behavior: initialize evidence and claims
empty, read them only when the previous-chunk query returned rows) ran under a
controlled interleaving: the state reader pauses after the previous-chunk query,
a second connection commits the same document with one chunk, one evidence link,
and one claim, and the paused reader resumes. Both backends and both write paths
(`_transaction` hook and auto-commit `_write_document`) were covered.

### Gate results

| Backend | Path | Interleaved insert | Interleaved removal |
| --- | --- | --- | --- |
| Neo4j | transaction hook | Divergent | Equal |
| Neo4j | auto-commit | Divergent | Equal |
| Memgraph | transaction hook | Equal | Equal |
| Memgraph | auto-commit | Divergent | Equal |

In every divergent insert run the baseline reader saw the concurrently committed
rows and deleted them as stale, while the staged candidate kept the evidence
link, its `SUBJECT_OF` claim relation, and the `Claim` node. Receipts matched in
all cells; only persisted rows diverged.

Supporting isolation probes: with an open reader transaction whose first read
returns empty, Neo4j read-committed exposes a concurrently committed row to a
later read in the same transaction, while Memgraph's snapshot isolation does
not. Auto-commit runs use a fresh snapshot per statement, so they observe the
commit on both backends.

### Decision

Skipping the evidence and claim reads when previous chunks are empty is only
safe under full snapshot isolation (Memgraph transaction hook). The writer must
keep baseline behavior on all supported paths, so the optimization is rejected.
No production change was made for this step.

The gate is pinned by
`tests/integration/test_writer_invariants.py::test_writer_state_read_concurrency_gate`
(opt-in, `database` tier, 8 backend/path/scenario cells, green twice in a row).
Note for the Memgraph transaction-hook insert cell: Memgraph can abort the
paused conflicting reader under its optimistic concurrency control; the writer
retry then re-reads post-commit state, and both variants converge to the same
final graph. The test pins variant equality there; the three non-isolated
insert cells pin the concrete baseline-deletes / candidate-keeps divergence.

## Optimization 2: relationship predicate specialization

Status: adopted in `_read_document_state` (`recon_graphrag/pipelines/writer.py`).

### Change

The relationship state read selects its predicate from input presence and keeps
the match envelope, the 12-column projection, the three `graph_name` checks, and
both query parameters unchanged:

| `rel_ids` | `chunk_ids` | Predicate |
| --- | --- | --- |
| nonempty | nonempty | original OR predicate |
| nonempty | empty | `r.id IN $rel_ids` |
| empty | nonempty | chunk-overlap `any(...)` |
| empty | empty | read skipped (preexisting) |

An empty list makes the other predicate false in Cypher, so each specialized
predicate returns exactly the rows the combined OR predicate returns.
Chunk-ownership matching stays on `r.source_chunk_ids`; no UNION, dedup, sort,
index, or shape changes were made.

### Equivalence evidence

Both backends returned byte-identical row lists (including order) for the
specialized rel-only and chunk-only predicates versus the baseline OR query on
identical seeded states: 40 entities/450 relationships with 100 parallel edges
sharing `(source, target, type)`, self-relationships, and a foreign-graph row;
then 400 entities/4000 relationships. Parallel-edge rows matched, so the
last-wins `(source, target, type)` merge key is unaffected. Cross-graph rows
never appear in results.

Pinned by `test_relationship_state_read_predicate_matches_available_inputs`
(deterministic recording) and
`tests/integration/test_writer_read_performance.py::test_relationship_state_read_equivalence_on_seeded_state`
(opt-in, both backends).

### Performance evidence

Opt-in harness `test_relationship_state_read_profiles` (alternating warm
rounds, medians over 6 repeats, one seeded graph with 1280 owned plus 1500
unrelated relationships, 170 relationship ids, 15 chunk ids):

| Backend | Read | Baseline OR min/med | Specialized min/med |
| --- | --- | --- | --- |
| Neo4j 5.26.0 | rel-only | 12.0/13.9 ms | 10.4/12.8 ms |
| Memgraph 5.11 | rel-only | 8.5/11.4 ms | 7.4/8.2 ms |
| Neo4j 5.26.0 | chunk-only | 20.1/25.1 ms | 25.0/31.7 ms (noise-level) |
| Memgraph 5.11 | chunk-only | 20.1/23.3 ms | 18.3/26.3 ms (noise-level) |

The rel-only cell wins on both backends; the chunk-only cell is within run
to-run dispersion. End-to-end write timings are reported by the harness and
not used as adoption gates because AI-free write totals vary more between
runs than the read predicates save.

A larger state (4000 relationships, 500 relationship ids) showed the same
ordering: rel-only 19.2 ms vs 20.9 ms (Neo4j) and 11.7 ms vs 13.0 ms
(Memgraph); chunk-only 86.6 ms vs 97.1 ms and 89.2 ms vs 91.5 ms.

### Rejected: UNWIND-driven relationship lookup

A per-id lookup (`UNWIND $rel_ids ... MATCH ()-[r {id: rid}]->()`) returns the
same row set in a different order (dangerous for parallel-edge last-wins
semantics) and is 26-50x slower without relationship-property indexes:
Neo4j min 1042.6 ms vs 20.9 ms; Memgraph min 336.5 ms vs 13.0 ms on the
4000-relationship state. Rejected on both equivalence and performance.
Adding relationship-property indexes is out of scope (no index additions).

### Verification

- Full deterministic suite: `uv run pytest -m "not integration"` — 1768 passed,
  1 skipped.
- Compatibility and backend suites (`tests/compatibility`,
  `tests/graphdb/...`): 252 passed, 1 skipped.
- Opt-in database suite (`test_writer_invariants.py` +
  `test_writer_read_performance.py`, `-m database`): 14 passed.
- Both endpoints are disposable containers; no production data touched.

## Outcome

Adopted: relationship predicate specialization in `_read_document_state`
(optimization 2) — one production file, `recon_graphrag/pipelines/writer.py`,
26 changed lines. Rejected: staged evidence/claim reads (optimization 1,
concurrency gate), UNWIND-driven id lookup (equivalence and performance),
relationship-property indexes (out of scope), and batching of merge/rewire
groups (not attempted; order and survivor semantics).

Checks not run: live AI-provider workflows (`tests/integration` e2e and
workflow tiers) because the optimization must not change AI workloads; large
removal workloads and query-plan profiling (reported as future candidates);
real-provider embedding tests.

Changed files: `recon_graphrag/pipelines/writer.py` (predicate selection),
`tests/pipelines/test_writer_characterization.py` (predicate, envelope, and
projection recording), `tests/integration/test_writer_invariants.py`
(concurrency gate), `tests/integration/test_writer_read_performance.py` (new
opt-in equivalence, differential, and profiling harness), and this record.

## Independent review (step 15)

A fresh reviewer session audited the uncommitted diff against the plan
contracts and returned OK with notes.

### Reviewer-verified contracts

Optimization 1 stays rejected (evidence/claim reads unconditional after the
previous-chunk read); the input-presence matrix selects the right predicate;
empty-list and `NULL` semantics preserve filtering; graph isolation, the
12-column projection, chunk-ownership matching, last-wins merge behavior, both
parameter keys, and the write lifecycle are unchanged. No public signature,
persisted shape, index, export, or compatibility fixture changed.

### Reviewer finding and remediation

Finding (minor): the differential coverage answered relationship reads with
test-local query strings, so the production reader and several edge cases were
not exercised. Remediation in
`tests/integration/test_writer_read_performance.py::test_relationship_state_read_differential_on_seeded_state`:

- The differential now calls the production `_read_document_state` (candidate)
  against a baseline-forced writer and compares exact row lists without
  sorting or normalization, for the both-inputs, rel-only (`chunk_ids == []`
  asserted), and chunk-only cells.
- The seeded state now includes a matching self-loop owned by a kept chunk, a
  relationship with a `NULL` id, a relationship with foreign-graph endpoints
  (must stay excluded by the envelope), and a relationship with no stored
  ownership (must match by id only).
- The write differential writes identical replacement documents through both
  writers on identically seeded graphs and asserts receipt equality plus final
  graph equality via `_dump_graph` (labels plus properties, no order
  normalization).

Result: 14 opt-in database tests green (8 gate cells + 2 equivalence +
2 differential + 2 profile).

### Out-of-scope observations recorded during remediation

These are pre-existing behaviors surfaced while building the differential
coverage; the adopted optimization does not touch them.

- `recon_graphrag/pipelines/writer.py:996`: `_reconcile` sorts
  `empty_relationship_ids` with `sorted(set(...))`; a stored relationship whose
  id is `NULL` can enter that list and crash the write with
  `TypeError: '<' not supported between instances of 'NoneType' and 'str'`.
  The test avoids the crash by giving the id-less seed row an extra
  non-document chunk id so its ownership never empties.
- `MERGE (source)-[r:TYPE]->(target)` with N existing rels on the endpoint
  pair applies `SET` to every matched rel and returns N count rows, so
  relationship receipts can exceed the submitted row count and a replacement
  rename applies to all parallel edges. Both writers behave identically; the
  differential asserts equality rather than absolute counts.
- The entity write merges on `(:__Entity__:{label} {id, graph_name})`; stored
  entities without their type label are treated as different nodes. Seed data
  must carry type labels to represent realistic states.
