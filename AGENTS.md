# Repository guide for agents

## What this repository does

Recon-GraphRAG is a Python GraphRAG SDK for Neo4j and Memgraph.
The SDK has three main functions:

1. **Graph building:** `GraphBuilderPipeline` extracts entities and relationships from documents or ingests structured data. It writes graph records, resolves duplicate entities, and embeds entities. The implementation is in `recon_graphrag/pipelines/` and `recon_graphrag/extraction/`.
2. **Community detection:** `CommunityPipeline` detects hierarchical Leiden communities. It generates community reports with an LLM and embeds those reports. The implementation is in `recon_graphrag/communities/`.
3. **Retrieval:** Local, global, and DRIFT retrievers answer questions with graph data, community reports, and source citations. The implementation is in `recon_graphrag/retrieval/`.

- Keep the SDK independent of a specific application domain.
- Preserve the shared graph-store contract for both database backends.

## Setup and tooling

Use Python 3.11 or later. CI uses Python 3.11.
CI runs `uv run pytest -m "not integration"` and `uv build`.
Neither `pyproject.toml` nor `.github/workflows/ci.yml` defines a lint or type-check command.

- Run commands from the repository root.
- Install development dependencies with `uv sync --extra dev --group dev`.
- Use `uv run` for project commands.
- After packaging or dependency changes, run `uv build`.
- Do not treat tools in a local cache as required project checks.
- For database setup, interface rules, and contribution rules, read `CONTRIBUTING.md`.
- For integration test tiers, read `docs/10-testing.md`.

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

- Unless the task explicitly requests a breaking change, preserve public exports, callable signatures, result shapes, and source and citation metadata.
- Before changing a shared contract, check its callers.
- Before changing a shared contract, check its compatibility tests.
- Keep Neo4j and Memgraph consistent with the shared graph-store contract.
- For shared behavior changes, check both backend implementations.
- For shared behavior changes, check both backend test suites.
- Follow the Protocol and ABC rules in `CONTRIBUTING.md`.
- Use existing interfaces.
- Do not create duplicate interfaces.
- Change compatibility tests and fixtures under `tests/compatibility/` only for intentional contract changes.
- Do not change compatibility tests or fixtures only to make a failing test pass.
- Update affected documentation.
- Update affected examples.
- Do not change unrelated files or edits.
- Do not manually increase package versions. Release Please manages releases.

## Smoke-test data

Two datasets are available for smoke tests:

- **Synthetic data:** `tests/integration/synthetic_e2e_support.py` contains `SYNTHETIC_PAGES`, a compact security and incident corpus. The file also contains its schema, prompts, and assertions. `tests/integration/test_smoke_full.py` tests graph building, communities, and retrieval with this data on Neo4j and Memgraph.
- **Example data:** `examples/data.py` contains the movie-industry corpus. Its schema is in `examples/schema.py`. `examples/README.md` describes the extraction, ingestion, community, and search scripts.

- Use these datasets instead of creating a new smoke-test corpus.
- For manual end-to-end checks, use the example data with its schema and documented scripts.
- Keep test fixtures and factories under `tests/`.
- Do not import example data or helpers into tests.

## Validation and safety

- For deterministic checks without live services, run `uv run pytest -m "not integration"`.
- Before synthetic E2E smoke tests, configure a disposable graph database.
- Before synthetic E2E smoke tests, configure the selected LLM and embedder providers.
- For required settings, read `docs/10-testing.md` and `.env.example`.
- To run synthetic E2E smoke tests, use this command:

  ```bash
  RUN_E2E_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_smoke_full.py -m e2e
  ```

**Warning:** Live smoke tests write to databases and can incur provider costs.
Plain `pytest` can run live tests when `.env` enables integration flags.

- Never run live smoke tests against production data.
- Do not expose credentials from `.env`.
- Report which checks ran.
- Report which checks were skipped.

## Documentation writing

- Use ASD-STE100-based wording for maintained documentation, comments, and API docstrings.
- Apply these rules to this file as well.
- Do not claim formal ASD-STE100 compliance or certification.
- Use active voice.
- Name the actor when the reader could misunderstand who performs an action.
- Use the imperative for instructions.
- Put conditions before actions.
- Give one instruction per procedural step.
- Limit each procedural sentence to 20 words.
- Limit each descriptive sentence to 25 words.
- Exclude code blocks and table cells from sentence-length checks.
- Use one term for each concept.
- Preserve code identifiers, paths, commands, API contracts, and safety warnings.
- Retain technical terms such as embedding, Cypher, Leiden, LLM, and DRIFT.
- State prerequisites before commands.
- Verify technical claims against the current code.
- Verify examples against the current code.
- Preserve historical records and licenses.
- Follow the [documentation-writing conventions](CONTRIBUTING.md#documentation-writing) for additional guidance and SDK terminology exceptions.

## Before handing off a change

- For each bug fix, add a focused regression test.
- Add tests for new behavior with branches, loops, parsing, or other nontrivial logic.
- Run the affected deterministic tests with `-m "not integration"`.
- When feasible, run the full mandatory test suite.
- For shared API or graph-store changes, run the relevant compatibility tests.
- For shared API or graph-store changes, run both backend test suites.
- Check affected callers, exports, fixtures, documentation, and examples for missing updates.
- Do not include unrelated refactoring.
- Report which checks passed.
- Report any check failures.
- Report which live-provider or backend checks were not run.
- Do not claim end-to-end validation from tests that use only fake services.
