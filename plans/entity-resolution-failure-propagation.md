# Plan 3 — Entity resolution: observable failures

Part 3 of 4.

## Context

Finding 4 of `plans/issue.md`: `GraphBuilderPipeline._resolve_entities`
(`recon_graphrag/pipelines/graphrag_pipeline.py:915-943`) catches and logs exceptions by default;
`fail_on_resolution_error=True` propagates them. An offline failing-store fake confirmed both
paths. Consequence for the dome workload: a caller's retry wrapper cannot retry an exception that
was swallowed inside the library, and a resolution OOM left the build looking successful.

## Approach (to be detailed when this plan starts)

Decide, with the maintainer, between:

- **A. Document only** — keep the default, document in the docstring/README that failures are
  logged and swallowed unless `fail_on_resolution_error=True`. Zero behavior change.
- **B. Surface in the result** — always return a `resolution` dict that carries `error` when
  resolution failed, so dry-run and pipeline summaries report it; propagation stays opt-in.
- **C. Change the default** to propagate.

Constraint to carry forward: whatever is chosen must keep `_write_finalize_summary` truthful
(it currently prints `entity resolution: skipped (...)` or merge counts, and would print a merge
summary even after a swallowed failure).

## Files to modify

- `recon_graphrag/pipelines/graphrag_pipeline.py` (`_resolve_entities`, `_write_finalize_summary`).
- Possibly `recon_graphrag/graphdb/base.py` (result dict contract).
- Tests: `tests/pipelines/test_graph_builder.py`.

## Verification

- Fake store raising on `resolve_entities_*`: assert logged-and-continued vs raised for both
  `fail_on_resolution_error` values, and that the summary line reflects the failure.
