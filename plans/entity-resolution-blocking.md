# Plan 2 — Entity resolution: indexed candidate blocking

Part 2 of 4. Depends on `plans/entity-resolution-bounded-load.md` (plan 1) landing first so the
pool carries names only.

## Context

Finding 2 of `plans/issue.md`: `graphdb/entity_resolution.py:479-507` `_blocked_candidates` loops
the entire same-label singleton pool for every entity and only truncates at the end
(`candidates[:max_candidates]`). `_build_fuzzy_groups:417` calls it per remaining entity; hybrid
uses that path too. An offline counting fake measured 10,000 pool visits for 100 lookups with
`max_candidates=1`. On dome ("mostly unmatched entities", 136,038 entities) this is quadratic and
the 45-minute stall cannot be attributed only to the OOM. Pagination does not fix it.

## Key facts (from plan-1 exploration)

- `_normalize_name:51` strips **all** spaces, so `normalized_value.split()` is always a single
  token and `_is_acronym_match:509` degenerates to "first character of the normalized value is
  equal" (plus single-character-name edge cases). Blocking by first character is therefore almost
  exactly the existing filter.
- Empty `normalized_value` currently matches nothing (both `tok1`/`tok2` guards and
  `_is_acronym_match` fail), so empty names can be excluded from buckets.
- The remaining filters are the length window `abs(len1-len2) <= max(3, 0.4*min(len1,len2))`
  and same `graph_name` + `domain_label`.

## Approach

Precompute blocking buckets once per resolution and look candidates up instead of scanning.
Exact parity is a hard requirement (maintainer's answer): the same pairs must be produced in the
same order, so the scoring path (`_fuzzy_score`, `reviewed_pairs`, `dropped_pairs` for LLM rescue,
star-not-chain grouping) is untouched.

Bucket key = `(first_char, length)` where `first_char = normalized_value[:1]`. A per-entity lookup
iterates only the length values inside its own allowed window:

```python
allowed = range(max(1, len1 - maxd), len1 + maxd + 1)   # maxd = max(3, int(min_len * 0.4))
for length in allowed:
    for other in bucket[(first_char, length)]: ...
```

This preserves the exact predicate set while dropping the pool scan. Ordering: bucket insertion
order is pool order, and iterating `allowed` lengths ascending would reorder candidates — so
either keep the current in-pool order by sorting the shortlist by pool index, or accept the
ordering change and document it (decide during implementation; ordering only affects which
`max_candidates` win when a bucket overflows).

## Files to modify

- `recon_graphrag/graphdb/entity_resolution.py` — `_build_fuzzy_groups` builds the index;
  `_blocked_candidates` takes it instead of the raw pool.
- `tests/graphdb/test_entity_resolution_contract.py` — parity test vs the old scan.
- `tests/graphdb/neo4j/test_neo4j_entity_resolution.py`,
  `tests/graphdb/memgraph/test_memgraph_entity_resolution.py` — fuzzy/hybrid expectations.

## Reuse

- `_normalize_name`, `_is_acronym_match`, `_fuzzy_score` unchanged.
- No new dependency (`rapidfuzz` is already used, but `process.extract` is **not** adopted because
  a `score_cutoff` would silently kill `_rescue_dropped_pairs`).

## Steps

- [ ] Extract the length window predicate into a small helper used by both the index and the
      filter, so they cannot drift.
- [ ] Build `dict[tuple[str, int], list[_EntityRecord]]` once in `_build_fuzzy_groups`.
- [ ] Rewrite `_blocked_candidates(entity, index, max_candidates, used)` to walk the length
      window.
- [ ] Keep `used`/`reviewed_pairs` skipping semantics identical.
- [ ] Remove records from buckets (or skip via `used`) so `used` filtering does no extra copying.

## Verification

- **Parity test**: randomized pool (a few hundred names: exact dups, near dups, initials, empty
  strings, unicode, 1-char names); assert the new index returns exactly the old scan's candidate
  lists for every entity.
- **Scaling test**: assert pool visits are O(n × window) — e.g. count comparisons for 5,000 unique
  names and assert it stays under a fixed bound (regression guard against reinstating the scan).
- `uv run pytest tests/graphdb -q`.
