# Graph Backend Implementation Checklist

Recon-GraphRAG keeps each graph database backend isolated while sharing backend-neutral logic above the database boundary. Use this checklist when adding a backend such as ArcadeDB.

## Design Rules

- Keep the backend in its own package, for example `recon_graphrag/graphdb/arcadedb/`.
- Keep dialect-specific queries, procedures, internal node ID expressions, index DDL, and merge mechanics inside that package.
- Reuse shared helpers for backend-neutral behavior:
  - `recon_graphrag.graphdb.store_base.BaseGraphStore`
  - `recon_graphrag.graphdb.entity_resolution.BaseEntityResolver`
  - `recon_graphrag.graphdb.cypher`
  - `recon_graphrag.pipelines.writer_base.BaseGraphWriter`
- Preserve the public `GraphStore` protocol in `recon_graphrag.graphdb.base`.
- Do not remove GraphRAG pipeline stages to make a backend easier to add. If a backend cannot support a feature yet, expose that as an explicit limitation and test it.

## Required Backend Package

Create a backend package with these modules:

```text
recon_graphrag/graphdb/<backend>/
  __init__.py
  cypher.py
  entity_resolution.py
  index_manager.py
  store.py
```

Add retrieval and community query modules only when the backend needs dialect-specific templates:

```text
recon_graphrag/retrieval/<backend>/queries.py
recon_graphrag/communities/<backend>/detection.py
```

## Store Checklist

Implement a store class that satisfies `GraphStore`.

- Inherit `BaseGraphStore` when possible.
- Implement the query primitive used by shared store helpers:
  - `execute_query(query, parameters=None)`
- Implement backend-specific write/search/index methods:
  - `write_graph_document`
  - `create_indexes`
  - `drop_indexes`
  - `create_vector_index`
  - `create_fulltext_index`
  - `upsert_vectors`
  - `vector_search`
  - `keyword_search`
- `fetch_entity_context`
  - `search_communities`
  - `detect_communities`
  - `get_unembedded_communities`
  - `get_unembedded_entities`
  - `get_community_ranked_context`
  - `get_community_child_summary_context`
  - `get_community_summaries_by_keys`
  - `get_community_entities_by_keys`
  - `backfill_descriptions`
  - `resolve_chunk_citations`
  - `get_claims_for_entities`

Shared `BaseGraphStore` methods should cover:

- Count helpers and `validate_graph_build`
- Community stats and report persistence
- Safe read helpers such as `get_communities`, `get_claims_for_entities`, and `resolve_chunk_citations`
- Entity-resolution preflight helpers

## Writer Checklist

Keep a backend writer class, for example `ArcadeDBGraphWriter`, but inherit shared row preparation from `BaseGraphWriter`.

The backend writer should own:

- Query strings for documents, chunks, entities, evidence links, relationships, and claims.
- Dynamic label or relationship-type escaping.
- Backend-specific `MERGE`, `MATCH`, or equivalent persistence semantics.

The shared writer base should own:

- `write_graph_document` sequencing.
- Write stats shape.
- Row preparation.
- Grouping records by type.

## Entity Resolution Checklist

Inherit `BaseEntityResolver` for grouping, fuzzy matching, hybrid review, LLM review, conflict checks, and AI promotion.

The backend resolver should implement:

- `_preflight(graph_name, resolve_property)` — backend-specific checks such as procedure availability. Optional; the base resolver runs it before loading entities when defined.
- `_load_entities(graph_name, resolve_property)`
- `_merge_groups(groups, resolve_property)`

Preserve backend-specific merge semantics:

- Neo4j uses APOC merge behavior.
- Memgraph rewires relationships manually.
- ArcadeDB should choose a merge strategy that preserves relationships, aliases, canonical IDs, and source metadata before deleting or consolidating nodes.

## Index Checklist

Implement an index manager that creates the same logical indexes:

- `chunk-embeddings`
- `entity-embeddings`
- `community-embeddings`
- `entity-names`

Document any backend limitation, such as unsupported vector similarity functions, index replacement behavior, or fulltext query syntax.

## Retrieval Checklist

Keep query templates backend-local when syntax diverges.

The backend must provide equivalent behavior for:

- Local entity context retrieval.
- DRIFT entity context retrieval with community keys.
- Community summaries by graph-scoped keys.
- Bridging entities for DRIFT.
- Community ranked context for report generation.
- Child community summary context.

The returned row shapes should match existing Neo4j and Memgraph rows so shared retrievers and community summarizers keep working.

## Community Detection Checklist

Implement community detection using the backend's graph algorithm support.

The implementation must preserve:

- `graph_name` scoping.
- Configurable relationship types.
- Weighted relationship support when `relationship_weight_property` is set.
- Hierarchical community levels, with level `0` as the finest level.
- `Community` nodes and `IN_COMMUNITY` / `PARENT_COMMUNITY` relationships expected by retrieval and summarization.

## Test Checklist

Add focused tests before integration tests:

- Shared base contract tests (already exercise backend-neutral behavior):
  - `tests/graphdb/test_store_base.py`
  - `tests/graphdb/test_entity_resolution_contract.py`
  - `tests/pipelines/test_writer_characterization.py`
- Backend-specific tests:
  - `tests/graphdb/<backend>/test_<backend>_store.py`
  - `tests/graphdb/<backend>/test_<backend>_index_manager.py`
  - `tests/graphdb/<backend>/test_<backend>_entity_resolution.py`
  - `tests/pipelines/<backend>/test_<backend>_writer.py`
  - Backend-specific community detection tests if detection uses custom logic.
  - Retrieval query tests if row shape or ordering differs.

Add integration tests behind explicit run flags:

- `tests/integration/<backend>/test_<backend>_store_smoke.py`
- `tests/integration/<backend>/test_<backend>_community_detection_integration.py`
- `tests/integration/<backend>/test_<backend>_entity_resolution_integration.py`
- `tests/integration/<backend>/test_<backend>_movie_smoke.py`

The full backend smoke should validate:

- Document ingestion.
- Page/window chunking.
- Entity and relationship extraction.
- Graph construction and validation counts.
- Entity resolution.
- Community detection and summarization.
- Embedding and vector indexing.
- Local, global, and DRIFT search.
- Citation metadata survival.

## ArcadeDB Notes To Validate

Before implementing ArcadeDB, confirm:

- Which query language surface will be used for graph operations.
- Whether internal node IDs are stable enough for merge and retrieval logic.
- Whether dynamic labels and relationship types need escaping rules different from Cypher.
- Whether vector indexes and fulltext indexes are native, plugin-based, or require an adapter.
- Whether community detection is native, needs an external algorithm, or should initially be documented as unsupported.
- Whether relationship rewiring can be done atomically enough for entity resolution.

Do not copy a full Neo4j or Memgraph store as a starting point. Implement dialect-specific primitives first, then inherit shared behavior.
