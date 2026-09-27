# Changelog

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
