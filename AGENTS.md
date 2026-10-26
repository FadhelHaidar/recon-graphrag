# Repository guide for agents

## What this repository does

Recon-GraphRAG is a Python GraphRAG SDK supporting Neo4j and Memgraph. It serves three main purposes:

1. **Graph building:** `GraphBuilderPipeline` extracts entities and relationships from documents or ingests structured data, writes the knowledge graph, resolves duplicate entities, and embeds entities. See `recon_graphrag/pipelines/` and `recon_graphrag/extraction/`.
2. **Community detection:** `CommunityPipeline` builds hierarchical Leiden communities, generates community reports with an LLM, and embeds those reports. See `recon_graphrag/communities/`.
3. **Retrieval:** local, global, and DRIFT retrievers use the graph and community reports to answer questions with source citations. See `recon_graphrag/retrieval/`.

Keep the SDK domain-agnostic and preserve the shared graph-store contract across both database backends.

## Setup and tooling

- Requires Python 3.11 or newer; CI runs Python 3.11.
- Install development dependencies with `uv sync --extra dev --group dev`.
- Run commands from the repository root with `uv run`.
- CI runs `uv run pytest -m "not integration"` and builds the package with `uv build`. Use the build check when changing packaging or dependencies.
- No lint or type-check command is configured in `pyproject.toml` or `.github/workflows/ci.yml`. Do not assume a local tool cache defines a required check.
- See `CONTRIBUTING.md` for database setup, interface conventions, and contribution rules; see `docs/10-testing.md` for integration tiers.

## Where changes belong

| Area | Implementation | Tests |
| --- | --- | --- |
| Extraction and schema handling | `recon_graphrag/extraction/` | `tests/extraction/` |
| Graph building and writing | `recon_graphrag/pipelines/` | `tests/pipelines/` |
| Graph-store contracts and backends | `recon_graphrag/graphdb/` | `tests/graphdb/` |
| Community detection and reports | `recon_graphrag/communities/` | `tests/communities/` |
| Local, global, and DRIFT retrieval | `recon_graphrag/retrieval/` | `tests/retrieval/` |
| LLM and embedding providers | `recon_graphrag/llm/`, `recon_graphrag/embeddings/` | `tests/llm/`, `tests/embeddings/` |
| Public API and compatibility | `recon_graphrag/__init__.py`, affected modules | `tests/compatibility/` |
| Cross-stage live workflows | Affected SDK modules | `tests/integration/` |

## Compatibility rules

- Preserve public exports, callable signatures, result shapes, and source/citation metadata unless the task explicitly calls for a breaking change. Check callers and compatibility tests before changing a shared contract.
- Keep Neo4j and Memgraph aligned with the shared graph-store contract. For shared behavior changes, check both backend implementations and tests.
- Follow the existing Protocol/ABC conventions in `CONTRIBUTING.md`; reuse canonical interfaces instead of defining duplicates.
- Update compatibility tests and fixtures under `tests/compatibility/` only for intentional contract changes, not just to make a failing test pass. Update affected docs and examples as needed.
- Leave unrelated working-tree changes untouched. Do not manually bump package versions; Release Please manages releases.

## Smoke-test data

Two existing datasets are available for smoke testing. Reuse them instead of inventing a new corpus:

- **Synthetic test data:** `tests/integration/synthetic_e2e_support.py` contains the compact security/incident corpus (`SYNTHETIC_PAGES`), its schema, prompts, and assertions. `tests/integration/test_smoke_full.py` exercises graph building, communities, and retrieval against Neo4j and Memgraph using this data.
- **Example data:** `examples/data.py` contains the movie-industry corpus. Use it with `examples/schema.py` and the extraction, ingestion, community, and search scripts documented in `examples/README.md` for hands-on end-to-end checks.

Tests own their fixtures and factories under `tests/`; do not make tests import data or helpers from `examples/`.

## Validation and safety

- For local, deterministic checks without live services, run `uv run pytest -m "not integration"`.
- To run the synthetic E2E smoke tests, configure a disposable graph database and the selected LLM/embedder providers, then run `RUN_E2E_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_smoke_full.py -m e2e`. See `docs/10-testing.md` and `.env.example` for required settings.
- Live smoke tests write to databases and can incur provider costs. Never point them at production data. Plain `pytest` may run live tests if integration flags are enabled in `.env`.
- Do not expose credentials from `.env`. Report which checks actually ran and which were skipped.

## Before handing off a change

- Add a focused regression test for a bug fix and cover new nontrivial behavior.
- Run the affected deterministic tests with `-m "not integration"`, then the full mandatory suite when feasible. Shared API or graph-store changes should include the relevant compatibility and both backend test suites.
- Check affected callers, exports, fixtures, docs, and examples so the change is complete without unrelated refactoring.
- State which checks passed, any failures, and which live-provider or backend checks were not run. Do not claim end-to-end validation from fake-only tests.
