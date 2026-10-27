# Changelog

## Unreleased

### ⚠ BREAKING CHANGES

* **graphdb:** `get_unembedded_entities` now takes a required `graph_name`
  first argument; embedding runs are scoped to a single graph so entities
  from other graphs are never read or overwritten (production audit F1).
* **pipelines:** document IDs derived from `source`/`title` metadata are now
  content hashes (`doc:<16 hex chars>`) instead of slugs; sources that
  slugified identically (e.g. `A/B`, `A-B`) no longer collide. Pre-existing
  graphs built before this change are NOT migrated — re-ingested documents
  with the same source get new document IDs (production audit F2, user
  decision: hash new IDs, no migration). Explicit `metadata["document_id"]`
  still wins.
* **graphdb:** Neo4j `vector_search`/`keyword_search` over-fetch (5×) when
  label or `graph_name` filters are applied and trim to `top_k` after the
  filter, matching Memgraph behavior; result shape is unchanged.

## [0.12.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.12.0...v0.12.1) (2026-10-27)


### Bug Fixes

* repair malformed community report JSON ([d247ef1](https://github.com/FadhelHaidar/recon-graphrag/commit/d247ef174f11d840f06d976f8fcd3d29775a6e98))
* repair malformed community report JSON ([35fdb8d](https://github.com/FadhelHaidar/recon-graphrag/commit/35fdb8d96f3bd3f3ad38e3e1d70a18b1274822b4))

## [0.12.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.11.1...v0.12.0) (2026-10-27)


### Features

* **communities:** surface report errors with opt-in private details ([df6b049](https://github.com/FadhelHaidar/recon-graphrag/commit/df6b049d9c0e1163493e993cefafbad5398aa179))


### Bug Fixes

* **extraction:** preserve assembly identities and report property conflicts ([58cb364](https://github.com/FadhelHaidar/recon-graphrag/commit/58cb364f1bf084fc8a4267231ac53bcab3a70998))
* **extraction:** reject cross-label trusted identity collisions ([6e3fcba](https://github.com/FadhelHaidar/recon-graphrag/commit/6e3fcbad2445fa78f17167f12400e062057ada63))
* **graphdb:** retarget claim object ids during entity merges ([6b0f62d](https://github.com/FadhelHaidar/recon-graphrag/commit/6b0f62dda6afd7a56a959fb9960155f1996190b0))
* harden graph builder integrity and surface pipeline diagnostics ([80df7e3](https://github.com/FadhelHaidar/recon-graphrag/commit/80df7e3705e21ceb2a438882376a7563475a7e1c))
* **pipelines:** surface safe extraction and provider diagnostics ([b8633f1](https://github.com/FadhelHaidar/recon-graphrag/commit/b8633f114d8327c508220f0d288b4f6aee99306a))

## [0.11.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.11.0...v0.11.1) (2026-10-26)


### Bug Fixes

* **extraction:** retry empty summaries and preserve batch successes ([16240c8](https://github.com/FadhelHaidar/recon-graphrag/commit/16240c83882a1a8300a4ee20bd1e7ace61727aa2))
* **extraction:** retry empty summaries and preserve batch successes ([5de2bf6](https://github.com/FadhelHaidar/recon-graphrag/commit/5de2bf6b88c9bb6630dbb7b8c7c12b7c7bc201d9))

## [0.11.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.10.1...v0.11.0) (2026-10-26)


### Features

* return structured graph evidence from Local, DRIFT and Global search ([f45b4ca](https://github.com/FadhelHaidar/recon-graphrag/commit/f45b4ca104295d1623af982e0e069206676c57f8))


### Bug Fixes

* bound graph processing work and prevent report embedding loops ([cf67928](https://github.com/FadhelHaidar/recon-graphrag/commit/cf67928a1f3b3efd0ae2c8de229c15681a5e2b25))
* correct ingestion provenance, identity and vector freshness ([e4e0f21](https://github.com/FadhelHaidar/recon-graphrag/commit/e4e0f21e5cccad9546b53e5da2b6c427dfabd100))
* **extraction:** keep chunk-local IDs and homonyms separate until merges are safe (F7) ([a8f7a46](https://github.com/FadhelHaidar/recon-graphrag/commit/a8f7a463ed896b0a2be721089759b182277e542a))
* **graphdb:** enforce graph-scoped identity constraints and stop blocking the event loop (F8) ([6160f86](https://github.com/FadhelHaidar/recon-graphrag/commit/6160f8658357f966a7114783e4549075e4e4c1ee))
* **graphdb:** graph-scoped embeddings, hashed doc IDs, neo4j overfetch parity ([aee4ba8](https://github.com/FadhelHaidar/recon-graphrag/commit/aee4ba82500b468143a5a65253bbf1068204dce7))
* **graphdb:** verify Memgraph vector indexes with SHOW VECTOR INDEX INFO ([8a055bd](https://github.com/FadhelHaidar/recon-graphrag/commit/8a055bd8ac2ffb436e54c0595c9fa142e9000b89))
* harden graph maintenance and retrieval ([51476ff](https://github.com/FadhelHaidar/recon-graphrag/commit/51476ff920496eb846567c91dd41c8cf6787caaa))
* keep entity names and provider text out of embedding warnings ([83cc613](https://github.com/FadhelHaidar/recon-graphrag/commit/83cc61316c9c4b9f6d87914e9e6534e18a93f182))
* **pipelines:** fail_on_embedding_error verifies embedding completion ([774457f](https://github.com/FadhelHaidar/recon-graphrag/commit/774457f5050f734e6f8b189dfeee0227bd40513f))
* **pipelines:** reject reserved keys before any write in CypherGraphWriter ([49d85c3](https://github.com/FadhelHaidar/recon-graphrag/commit/49d85c31eaa592213cd1e19d4e99e5d872b3cf6d))
* **pipelines:** replace a document's contribution on re-ingestion (F4) ([5cbf132](https://github.com/FadhelHaidar/recon-graphrag/commit/5cbf1324cadf4b40a50dde3e1cfe2a4a7cd012b9))
* reject non-positive concurrency before starting a build ([5f9edb6](https://github.com/FadhelHaidar/recon-graphrag/commit/5f9edb683ae4887450a7fc1c88306b6978c43a4c))
* **release:** drop stale release-as pin for 0.10.1 ([95602e5](https://github.com/FadhelHaidar/recon-graphrag/commit/95602e5663512176f185a29286761a0932049afb))
* **release:** drop stale release-as pin for 0.10.1 ([7b42ac1](https://github.com/FadhelHaidar/recon-graphrag/commit/7b42ac1fdf9dd35e263cd4e4eaf496bcefd50055))
* report confirmed write counts and fail on unresolved references ([5c90cbe](https://github.com/FadhelHaidar/recon-graphrag/commit/5c90cbe751e6bd691cebaab2f50a611802d1a1b3))
* retry a deadlocked document write before failing ([b8d78a6](https://github.com/FadhelHaidar/recon-graphrag/commit/b8d78a63f09f0117c5757b0444afb525490dbe89))
* retry the unique-constraint race between concurrent writers ([b2e8b8e](https://github.com/FadhelHaidar/recon-graphrag/commit/b2e8b8e612a6e2bd885dc3bf61b23b0c710aad6d))
* write one document in one database transaction ([14afa92](https://github.com/FadhelHaidar/recon-graphrag/commit/14afa928ecbca4ae58d0a2347b65206ff11d1ffe))


### Documentation

* **audit:** update checkpoints for F1/F2/F3/F6 fix-status and resolved lockfile drift ([8d40554](https://github.com/FadhelHaidar/recon-graphrag/commit/8d4055483d607a4a4afaca98e0e22635ff08a922))
* describe per-document write transactions and their limits ([b15b268](https://github.com/FadhelHaidar/recon-graphrag/commit/b15b2682251df89cc6b53c5e5348ae80c76750ae))
* record F6, F11 and F12 fix status after the write-path patches ([5109f47](https://github.com/FadhelHaidar/recon-graphrag/commit/5109f478e983e3bd8aa33e2bf5e5f31afc74c4ef))

## [0.10.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.10.1...v0.10.1) (2026-10-26)


### Features

* return structured graph evidence from Local, DRIFT and Global search ([f45b4ca](https://github.com/FadhelHaidar/recon-graphrag/commit/f45b4ca104295d1623af982e0e069206676c57f8))


### Bug Fixes

* bound graph processing work and prevent report embedding loops ([cf67928](https://github.com/FadhelHaidar/recon-graphrag/commit/cf67928a1f3b3efd0ae2c8de229c15681a5e2b25))
* correct ingestion provenance, identity and vector freshness ([e4e0f21](https://github.com/FadhelHaidar/recon-graphrag/commit/e4e0f21e5cccad9546b53e5da2b6c427dfabd100))
* **extraction:** keep chunk-local IDs and homonyms separate until merges are safe (F7) ([a8f7a46](https://github.com/FadhelHaidar/recon-graphrag/commit/a8f7a463ed896b0a2be721089759b182277e542a))
* **graphdb:** enforce graph-scoped identity constraints and stop blocking the event loop (F8) ([6160f86](https://github.com/FadhelHaidar/recon-graphrag/commit/6160f8658357f966a7114783e4549075e4e4c1ee))
* **graphdb:** graph-scoped embeddings, hashed doc IDs, neo4j overfetch parity ([aee4ba8](https://github.com/FadhelHaidar/recon-graphrag/commit/aee4ba82500b468143a5a65253bbf1068204dce7))
* **graphdb:** verify Memgraph vector indexes with SHOW VECTOR INDEX INFO ([8a055bd](https://github.com/FadhelHaidar/recon-graphrag/commit/8a055bd8ac2ffb436e54c0595c9fa142e9000b89))
* harden graph maintenance and retrieval ([51476ff](https://github.com/FadhelHaidar/recon-graphrag/commit/51476ff920496eb846567c91dd41c8cf6787caaa))
* keep entity names and provider text out of embedding warnings ([83cc613](https://github.com/FadhelHaidar/recon-graphrag/commit/83cc61316c9c4b9f6d87914e9e6534e18a93f182))
* **pipelines:** fail_on_embedding_error verifies embedding completion ([774457f](https://github.com/FadhelHaidar/recon-graphrag/commit/774457f5050f734e6f8b189dfeee0227bd40513f))
* **pipelines:** reject reserved keys before any write in CypherGraphWriter ([49d85c3](https://github.com/FadhelHaidar/recon-graphrag/commit/49d85c31eaa592213cd1e19d4e99e5d872b3cf6d))
* **pipelines:** replace a document's contribution on re-ingestion (F4) ([5cbf132](https://github.com/FadhelHaidar/recon-graphrag/commit/5cbf1324cadf4b40a50dde3e1cfe2a4a7cd012b9))
* reject non-positive concurrency before starting a build ([5f9edb6](https://github.com/FadhelHaidar/recon-graphrag/commit/5f9edb683ae4887450a7fc1c88306b6978c43a4c))
* report confirmed write counts and fail on unresolved references ([5c90cbe](https://github.com/FadhelHaidar/recon-graphrag/commit/5c90cbe751e6bd691cebaab2f50a611802d1a1b3))
* retry a deadlocked document write before failing ([b8d78a6](https://github.com/FadhelHaidar/recon-graphrag/commit/b8d78a63f09f0117c5757b0444afb525490dbe89))
* retry the unique-constraint race between concurrent writers ([b2e8b8e](https://github.com/FadhelHaidar/recon-graphrag/commit/b2e8b8e612a6e2bd885dc3bf61b23b0c710aad6d))
* write one document in one database transaction ([14afa92](https://github.com/FadhelHaidar/recon-graphrag/commit/14afa928ecbca4ae58d0a2347b65206ff11d1ffe))


### Documentation

* **audit:** update checkpoints for F1/F2/F3/F6 fix-status and resolved lockfile drift ([8d40554](https://github.com/FadhelHaidar/recon-graphrag/commit/8d4055483d607a4a4afaca98e0e22635ff08a922))
* describe per-document write transactions and their limits ([b15b268](https://github.com/FadhelHaidar/recon-graphrag/commit/b15b2682251df89cc6b53c5e5348ae80c76750ae))
* record F6, F11 and F12 fix status after the write-path patches ([5109f47](https://github.com/FadhelHaidar/recon-graphrag/commit/5109f478e983e3bd8aa33e2bf5e5f31afc74c4ef))

## [0.10.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.10.0...v0.10.1) (2026-10-18)


### Features

* **testing:** configure OpenAI reasoning options in integration factories ([227bd9d](https://github.com/FadhelHaidar/recon-graphrag/commit/227bd9d4a5a4b6196e2b61581eef9711d53bd10b))


### Bug Fixes

* **extraction:** enforce declared property contracts ([18ab7b8](https://github.com/FadhelHaidar/recon-graphrag/commit/18ab7b82a43635f7174b6f6d50251870fe25f6ab))
* **extraction:** preserve Unicode token chunk boundaries ([a7dc679](https://github.com/FadhelHaidar/recon-graphrag/commit/a7dc679c5b03edc24a8c07deaafe3a793f352798))
* **extraction:** propagate required pass failures and drain tasks ([b299a1f](https://github.com/FadhelHaidar/recon-graphrag/commit/b299a1f7eba2f99b1fa0e645d82cfaa796492ed2))
* **pipelines:** reject incomplete documents before persistence ([2fedefb](https://github.com/FadhelHaidar/recon-graphrag/commit/2fedefbb3505c0093df8478b7aa0bc32df600dcc))

## [0.10.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.9.1...v0.10.0) (2026-07-17)


### Features

* custom prompts + token-usage observability ([#85](https://github.com/FadhelHaidar/recon-graphrag/issues/85)) ([0e0b762](https://github.com/FadhelHaidar/recon-graphrag/commit/0e0b762045c94b27d1f1e4533b8dcbfa7337c09b))

## [0.9.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.9.0...v0.9.1) (2026-07-11)


### Bug Fixes

* **extraction:** include entity properties in description summarization ([#83](https://github.com/FadhelHaidar/recon-graphrag/issues/83)) ([11f86c2](https://github.com/FadhelHaidar/recon-graphrag/commit/11f86c2f62250a1aa4e263a90e0a236348b7c337))

## [0.9.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.8.0...v0.9.0) (2026-07-10)


### Features

* **extraction:** enable row-level graph construction from tabular sources ([#81](https://github.com/FadhelHaidar/recon-graphrag/issues/81)) ([3b1f2ef](https://github.com/FadhelHaidar/recon-graphrag/commit/3b1f2ef75f7bf75da5381126ac864dfaa2d19471))

## [0.8.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.7.4...v0.8.0) (2026-07-09)


### Features

* **schema:** add LLM-powered auto-analysis of graph schemas from sample documents ([#79](https://github.com/FadhelHaidar/recon-graphrag/issues/79)) ([051a5f1](https://github.com/FadhelHaidar/recon-graphrag/commit/051a5f1c174a3934c41dea8af9c6274227719f90))

## [0.7.4](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.7.3...v0.7.4) (2026-07-08)


### Bug Fixes

* replace print statements with logging and add tqdm progress bars ([#77](https://github.com/FadhelHaidar/recon-graphrag/issues/77)) ([3b1e654](https://github.com/FadhelHaidar/recon-graphrag/commit/3b1e654c56520f96b6a4bbe3975fcc3f731f4762))

## [0.7.3](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.7.2...v0.7.3) (2026-07-07)


### Bug Fixes

* **entity-resolution:** re-wrap scalar source_chunk_ids collapsed by apoc mergeNodes ([#75](https://github.com/FadhelHaidar/recon-graphrag/issues/75)) ([6cc6548](https://github.com/FadhelHaidar/recon-graphrag/commit/6cc6548cda8247c1fd2a31636dacc6be394a4466))

## [0.7.2](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.7.1...v0.7.2) (2026-07-03)


### Bug Fixes

* entity resolution, retrieval normalization, and graph store consolidation ([#71](https://github.com/FadhelHaidar/recon-graphrag/issues/71)) ([21a2167](https://github.com/FadhelHaidar/recon-graphrag/commit/21a2167ff0022d58713d8602140860d107caa444))


### Documentation

* make README more appealing as an open source project ([#69](https://github.com/FadhelHaidar/recon-graphrag/issues/69)) ([3e8de5c](https://github.com/FadhelHaidar/recon-graphrag/commit/3e8de5caa3ee237cd551fdba3978a7bc689dd386))

## [0.7.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.7.0...v0.7.1) (2026-07-02)


### Bug Fixes

* **ci:** enforce owner approval with a status check ([#67](https://github.com/FadhelHaidar/recon-graphrag/issues/67)) ([401ded7](https://github.com/FadhelHaidar/recon-graphrag/commit/401ded7d64350e35cf02a62ebc7cc93476a338f6))

## [0.7.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.6.2...v0.7.0) (2026-07-02)


### ⚠ BREAKING CHANGES

* remove summary field, use_reports parameter, summary_prompt parameter, store_community_summary method, get_community_summaries_by_keys, get_community_entities_by_keys, community_top_k search parameter, level alias for community_level in global search, ReindexRequiredError, and schema version tracking. Default chunk_size changed from 1000 to 1200, chunk_overlap from 200 to 100, max_gleanings from 0 to 1.

### Features

* hybrid entity-resolution with LLM rescue, concurrent extraction, and GraphRAG-aligned retrieval ([#60](https://github.com/FadhelHaidar/recon-graphrag/issues/60)) ([2ea8dc7](https://github.com/FadhelHaidar/recon-graphrag/commit/2ea8dc7ce26b4f5290dda8f1964fabb4f940cecd))


### Bug Fixes

* **tests:** align index manager vector index count with dropped chunk index ([#61](https://github.com/FadhelHaidar/recon-graphrag/issues/61)) ([86fc27b](https://github.com/FadhelHaidar/recon-graphrag/commit/86fc27b4abd2bad25e43baa3e928459ff9a07353))

## [0.6.2](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.6.1...v0.6.2) (2026-06-28)


### Bug Fixes

* add Anthropic LLM provider and packaging fixes for PyPI readiness ([#57](https://github.com/FadhelHaidar/recon-graphrag/issues/57)) ([139ec3f](https://github.com/FadhelHaidar/recon-graphrag/commit/139ec3f336d65e1dcb01bd768a979dec08de33c9))

## [0.6.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.6.0...v0.6.1) (2026-06-28)


### Bug Fixes

* add synthesize_response flag to skip LLM synthesis and expose retrieved context ([#55](https://github.com/FadhelHaidar/recon-graphrag/issues/55)) ([a83982c](https://github.com/FadhelHaidar/recon-graphrag/commit/a83982cf7089365c68cccaa33e2f898ea39757e2))

## [0.6.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.5.0...v0.6.0) (2026-06-28)


### Features

* global search source metadata ([#49](https://github.com/FadhelHaidar/recon-graphrag/issues/49)) ([2795c7c](https://github.com/FadhelHaidar/recon-graphrag/commit/2795c7c9a955487d91c042a4c88a2ad57702e70a))

## [0.5.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.4.1...v0.5.0) (2026-06-27)


### ⚠ BREAKING CHANGES

* **pipeline:** `build_from_pages()` removed — use `build_from_documents([{"pages": pages}], window_size=..., window_overlap=...)` instead. `chunk_size` and `chunk_overlap` are no longer accepted in the `GraphBuilderPipeline` constructor; pass them per call.

### Features

* **pipeline:** consolidate multi-source input and add token-based chunking ([#47](https://github.com/FadhelHaidar/recon-graphrag/issues/47)) ([a927cfe](https://github.com/FadhelHaidar/recon-graphrag/commit/a927cfe57be514bb9afacfe7da1b523a3b1d9772))

## [0.4.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.4.0...v0.4.1) (2026-06-26)


### Documentation

* remove Louvain mention — only Leiden is supported ([#44](https://github.com/FadhelHaidar/recon-graphrag/issues/44)) ([f768c8c](https://github.com/FadhelHaidar/recon-graphrag/commit/f768c8c706b11fcc3bc8c648b62d86599da8f3b4))

## [0.4.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.3.0...v0.4.0) (2026-06-24)


### Features

* **search:** page metadata propagation and citation metadata in LLM context ([#38](https://github.com/FadhelHaidar/recon-graphrag/issues/38)) ([eb8f0da](https://github.com/FadhelHaidar/recon-graphrag/commit/eb8f0da001615052528197401244e5256ac4a9cc))

## [0.3.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.2.3...v0.3.0) (2026-06-19)


### Features

* add configurable concurrency to entity extraction ([03adc25](https://github.com/FadhelHaidar/recon-graphrag/commit/03adc2576bd36b544eabefa812c9dfa3704e89eb))

## [0.2.3](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.2.2...v0.2.3) (2026-06-18)


### Bug Fixes

* project Neo4j relationship weights for community detection ([#32](https://github.com/FadhelHaidar/recon-graphrag/issues/32)) ([8b69cf1](https://github.com/FadhelHaidar/recon-graphrag/commit/8b69cf1419cd58e447806a39e284a419afd08a55))

## [0.2.2](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.2.1...v0.2.2) (2026-06-18)


### Bug Fixes

* llm entity resolution  ([5fa5cdd](https://github.com/FadhelHaidar/recon-graphrag/commit/5fa5cddfd8059cc664770a3885b4d696878c6c6b))

## [0.2.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.2.0...v0.2.1) (2026-06-17)


### BREAKING CHANGES

* FalkorDB backend support and related extras/env vars/tests are removed in favor of Memgraph.

### Features

* ad memgraph support and refactor example scripts ([c62e55e](https://github.com/FadhelHaidar/recon-graphrag/commit/c62e55e4995e6d733142d918453d3df8e8448e5e))
* add Memgraph backend and artifact-based examples ([62c5efb](https://github.com/FadhelHaidar/recon-graphrag/commit/62c5efb86ab6c7965ffc9e943663e5a0117dc7ea))
* add memgraph database support ([b53c3b4](https://github.com/FadhelHaidar/recon-graphrag/commit/b53c3b4ed733c6ca6a7fc8e7c2a2dde6bc930ba6))
* add Memgraph support ([6635e95](https://github.com/FadhelHaidar/recon-graphrag/commit/6635e95f7046c475a009c6876980eff6c0ca7e4f))
* bootstrap recon-graphrag SDK with domain schema, community detection, and multi-mode search ([e429980](https://github.com/FadhelHaidar/recon-graphrag/commit/e429980049e2937a31aefd5e254e4214a3547f91))
* update documentation for indexing, community levels, and search modes ([bdbee55](https://github.com/FadhelHaidar/recon-graphrag/commit/bdbee55c80dedb070f182f9a32a7a2ddf5bd456c))


### Bug Fixes

* test auto versioning workflow ([7b230d6](https://github.com/FadhelHaidar/recon-graphrag/commit/7b230d6122e1fa45c0e208ad4f7b0f558cd3d682))
* update package metadata ([a1bc080](https://github.com/FadhelHaidar/recon-graphrag/commit/a1bc0804799b3350fb23389b4d1bd12b704e8ed3))
* update package metadata ([e9718c7](https://github.com/FadhelHaidar/recon-graphrag/commit/e9718c72f05e34cbea8118aaabc65c3b2168e799))
* update package metadata ([e9718c7](https://github.com/FadhelHaidar/recon-graphrag/commit/e9718c72f05e34cbea8118aaabc65c3b2168e799))
* update release workflow permissions and add hvcs configuration ([15d7204](https://github.com/FadhelHaidar/recon-graphrag/commit/15d7204524ac538882e14660c86d3267d899f2c0))


### Documentation

* clean up movie example workflow ([df0bb17](https://github.com/FadhelHaidar/recon-graphrag/commit/df0bb172315e5a137784714ba3836fa7dfad0d36))
* document breaking change commit style for major version bumps ([4749c9f](https://github.com/FadhelHaidar/recon-graphrag/commit/4749c9fa4ad43d8dd7c1432c47cbd4706971d7d2))
* fix documentation ([43de87f](https://github.com/FadhelHaidar/recon-graphrag/commit/43de87f38b9bb3660188c9f979520bd90001cb39))
* fix pyproject.toml URL section placement and regenerate uv.lock ([b4a8606](https://github.com/FadhelHaidar/recon-graphrag/commit/b4a860672a3f01ad3dade709658ee09cbea42261))
* reorganize README and add comprehensive documentation ([983b935](https://github.com/FadhelHaidar/recon-graphrag/commit/983b9356bce3ac774d38ec705d814a7d16c60ee7))


### Miscellaneous Chores

* release as 0.2.1 ([#28](https://github.com/FadhelHaidar/recon-graphrag/issues/28)) ([fefb8eb](https://github.com/FadhelHaidar/recon-graphrag/commit/fefb8eb84ea264aa60496c68cdce1868ed31f78c))

## [0.2.0](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.1.2...v0.2.0) (2026-06-17)


### Features

* add Memgraph support

## [0.1.2](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.1.1...v0.1.2) (2026-06-15)


### Documentation

* document breaking change commit style for major version bumps ([4749c9f](https://github.com/FadhelHaidar/recon-graphrag/commit/4749c9fa4ad43d8dd7c1432c47cbd4706971d7d2))
* fix documentation ([43de87f](https://github.com/FadhelHaidar/recon-graphrag/commit/43de87f38b9bb3660188c9f979520bd90001cb39))
* fix pyproject.toml URL section placement and regenerate uv.lock ([b4a8606](https://github.com/FadhelHaidar/recon-graphrag/commit/b4a860672a3f01ad3dade709658ee09cbea42261))
* reorganize README and add comprehensive documentation ([983b935](https://github.com/FadhelHaidar/recon-graphrag/commit/983b9356bce3ac774d38ec705d814a7d16c60ee7))

## [0.1.1](https://github.com/FadhelHaidar/recon-graphrag/compare/v0.1.0...v0.1.1) (2026-06-12)


### Bug Fixes

* test auto versioning workflow ([7b230d6](https://github.com/FadhelHaidar/recon-graphrag/commit/7b230d6122e1fa45c0e208ad4f7b0f558cd3d682))
* update package metadata ([a1bc080](https://github.com/FadhelHaidar/recon-graphrag/commit/a1bc0804799b3350fb23389b4d1bd12b704e8ed3))
* update package metadata ([e9718c7](https://github.com/FadhelHaidar/recon-graphrag/commit/e9718c72f05e34cbea8118aaabc65c3b2168e799))
* update package metadata ([e9718c7](https://github.com/FadhelHaidar/recon-graphrag/commit/e9718c72f05e34cbea8118aaabc65c3b2168e799))
* update release workflow permissions and add hvcs configuration ([15d7204](https://github.com/FadhelHaidar/recon-graphrag/commit/15d7204524ac538882e14660c86d3267d899f2c0))
