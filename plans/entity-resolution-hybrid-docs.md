# Plan 4 — Entity resolution: document hybrid embedding semantics

Part 4 of 4.

## Context

Finding 3 of `plans/issue.md`: `_score_with_embeddings`
(`recon_graphrag/graphdb/entity_resolution.py:771-809`) embeds candidate **names** afresh via
`embedder.async_embed_query(...)`; it never reads stored entity vectors. Meanwhile both backends
invalidate the survivor's vector on merge (`graphdb/neo4j/entity_resolution.py:115`
`node.embedding = null`, `graphdb/memgraph/entity_resolution.py:95`). A downstream app that
embedded *before* resolution therefore assumed hybrid consumed stored vectors, and merged
survivors ended up with no vector.

## Approach

Documentation + a small loudness fix, no algorithm change:

- Docstrings on `resolve_hybrid` / `_score_with_embeddings`: embeddings are computed from names at
  review time; stored vectors are never read; merges null the survivor's `embedding` property.
- Note in the entity-resolution docs page that resolving **before** the explicit embedding pass is
  the supported order for a fresh graph, and that this does not help incremental runs (existing
  vectors survive on unmerged nodes).
- Optional: emit one `logger.info` per resolution when `embedder` is set, stating that N stored
  vectors were invalidated by merges and need re-embedding.

## Files to modify

- `recon_graphrag/graphdb/entity_resolution.py` (docstrings, optional log line).
- `docs/` entity-resolution page.

## Verification

- Doc-only change: `uv run pytest tests/graphdb -q` stays green; no behavior assertions.
