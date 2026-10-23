# Plan 1 — Entity resolution: bounded load (no vectors, paged)

Part 1 of 4. Follow-ups: `plans/entity-resolution-blocking.md` (finding 2),
`plans/entity-resolution-failure-propagation.md` (finding 4),
`plans/entity-resolution-hybrid-docs.md` (finding 3).

## Context

`plans/issue.md` records a failed run against v0.10.1 on the dome dataset: 136,038 entities
(768-dim vectors stored on the nodes), 378,005 chunks, Neo4j
`dbms.memory.transaction.total.max = 716.8 MiB`. Resolution ran 45 min, merged nothing, client
RSS 4.8 GB, no LLM calls, then died with
`Neo.TransientError.General.MemoryPoolOutOfMemoryError` ("allocation of an extra 32.0 MiB would
use more than the limit 716.8 MiB … threshold reached").

Cause (finding 1 of the issue): `_Neo4jEntityResolver._load_entities`
(`recon_graphrag/graphdb/neo4j/entity_resolution.py:31-77`) runs one unpaged query returning
`properties(e)` for every entity — i.e. every 768-float vector — and materializes all rows. The
Memgraph twin is `recon_graphrag/graphdb/memgraph/entity_resolution.py:12-50`. All four
strategies and dry runs share this loader.

Outcome wanted: a loader whose peak memory does not depend on vector count, and whose server-side
transaction stays under a small `dbms.memory.transaction.total.max`. Grouping results must not
change.

## Approach

**Two-stage load.** Grouping never reads properties — only `node_id`, `entity_id`, `graph_name`,
`domain_label`, `resolve_value`/`normalized_value`. Properties are needed only for *candidate*
nodes (conflict checks, merge summaries, LLM profiles). So:

1. **Bulk stage** — project only the identity fields, keyset-paged. No `properties(e)`, therefore
   no embeddings: 136,038 × ~400 B ≈ 55 MB instead of ~2.5 GB.
2. **Hydration stage** — fetch `properties(n)` in id-batches for the node ids that actually
   appear in a group or review group, and assign onto the `_EntityRecord`s in place.

No APOC needed for the projection (we simply don't project a map at all), so the dry-run-without-
APOC behaviour and `_preflight` stay untouched.

Grouping parity is exact by construction: every group builder reads only non-property fields, and
every property consumer is hydrated before it runs. Verified consumer list:

| consumer | file:line | needs |
|---|---|---|
| `_collect_observations` (merge summaries) | `graphdb/entity_resolution.py:321-339` | `descriptions` |
| `_merge_groups` canonical choice | `graphdb/neo4j/entity_resolution.py:79-99` | `canonical_key`, `human_readable_id` |
| `conflict_for_group` via `_apply_conflict_rules` | `graphdb/entity_resolution.py:606-650` | configured conflict props |
| `build_entity_profiles` via `_llm_review` | `graphdb/entity_resolution_context.py:33-72` | standard + context props |

Note `graphdb/memgraph/entity_resolution.py:138-149` `_combine_properties` already does a batched
`WHERE id(n) IN $node_ids RETURN properties(n)` — the hydration helper should copy that shape.

Keyset paging (`WHERE elementId(e) > $after ORDER BY elementId(e) LIMIT $page` for Neo4j,
`id(e)` for Memgraph) is stable because *all* loading finishes before any merge: merging happens
in `_finish_resolution` after every group is built, so paging cannot skip candidates.

## Files to modify

- `recon_graphrag/graphdb/entity_resolution.py` — new `_hydrate_properties(records)` helper on
  `BaseEntityResolver`; call it in `_finish_resolution` before `_prepare_merge_summaries`, and in
  `_build_hybrid_groups` after `_build_alias_groups` / before `_apply_conflict_rules`.
- `recon_graphrag/graphdb/neo4j/entity_resolution.py` — paged, property-free `_load_entities`;
  new `_load_properties(node_ids)`.
- `recon_graphrag/graphdb/memgraph/entity_resolution.py` — same, reusing `_combine_properties`
  query shape.
- Tests: `tests/graphdb/neo4j/test_neo4j_entity_resolution.py` (its `FakeGraphStore` matches on
  query text), `tests/graphdb/memgraph/test_memgraph_entity_resolution.py`,
  `tests/graphdb/test_entity_resolution_contract.py`, plus marker-gated
  `tests/integration/neo4j/test_neo4j_entity_resolution_integration.py`.

## Reuse

- `graphdb/memgraph/entity_resolution.py:138` `_combine_properties` batched id-fetch pattern.
- `escape_cypher_identifier` (`recon_graphrag/graphdb/cypher.py`) for `resolve_property`.
- Existing `_EntityRecord` dataclass — hydration mutates `record.properties`, no new type.
- No new dependencies.

## Steps

- [x] `_Neo4jEntityResolver._load_entities`: drop `properties(e)` from the RETURN; loop pages with
  `elementId(e) > $after ORDER BY node_id LIMIT $page_size` (module constant, e.g. 5000), stop on
  a short page.
- [x] Add `_Neo4jEntityResolver._load_properties(node_ids)` using
  `MATCH (n) WHERE elementId(n) IN $ids RETURN elementId(n) AS node_id, properties(n) AS props`,
  chunked at ~1000 ids.
- [x] Mirror both in `_MemgraphEntityResolver` with `id(e)` instead of `elementId(e)`.
- [x] Add `BaseEntityResolver._hydrate_properties(records)` — collect ids where
  `record.properties` is empty, call `_load_properties` once, assign in place; no-op when all
  records are already hydrated (keeps custom subclasses working).
- [x] Wire the two hydration points (`_finish_resolution`, `_build_hybrid_groups`).
- [x] Keep `_load_entities` as an abstract method with the same signature so memgraph/neo4j stay
  interchangeable.
- [x] Update the query-text-matching fakes for the new paging + hydration queries.
- [x] Short docstring note on `_load_entities`: "identity fields only; properties are hydrated
  lazily for candidate nodes".

## Verification

- `uv run pytest tests/graphdb tests/pipelines -q` (all four strategies + dry run).
- **Parity test** (new, `tests/graphdb/test_entity_resolution_contract.py`): a resolver whose
  fakes return the same rows as today's loader; assert `merged_groups`, `candidate_groups`,
  `review_groups` (names + node_ids + scores + conflicts) and `signals` are byte-identical to
  the pre-change expectations captured from the current tests.
- **Paging test** (new, fake store): 10 rows with page size 3; assert `_load_entities` returns all
  10 in elementId order and issues 4 queries with monotonically increasing cursors.
- **Hydration test** (new): assert `properties` is `{}` right after `_load_entities` and that only
  node ids present in a group/review group are hydrated, exactly once.
- Live (marker-gated, optional): `RUN_DATABASE_INTEGRATION_TESTS=1` Neo4j with
  `dbms.memory.transaction.total.max=256m` on a few-thousand-node graph; resolution completes and
  merges match the unconstrained run.

## Out of scope (deliberately)

- Candidate-generation blocking (`_blocked_candidates` still scans the pool) → plan 2.
- Failure propagation through `GraphBuilderPipeline._resolve_entities` → plan 3.
- Documenting that hybrid embeds names rather than stored vectors → plan 4.
- No new public parameters: page size and hydration batch size stay module constants.
