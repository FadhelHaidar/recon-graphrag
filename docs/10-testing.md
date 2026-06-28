# Testing

This guide explains how to run the Recon-GraphRAG test suite. It covers mandatory tests, optional integration tests, and which commands to run for different kinds of changes.

The short version:

- Run mandatory tests while developing: `pytest -m "not integration"`
- Run provider smoke tests when changing provider setup.
- Run database tests when changing graph store behavior.
- Run workflow tests when changing pipeline orchestration with deterministic fakes.
- Run E2E tests for full confidence against real services.

## Testing model

The suite is split into layers by dependency cost and determinism.

### Mandatory tests

Mandatory tests are fast, local, and deterministic. They do not require real provider credentials or a live graph database.

They use fake graph stores, fake Bolt drivers/sessions, fake LLMs, fake embedders, monkeypatched provider clients, and in-memory assertions.

They answer questions like:

- Does our code call the right method?
- Does it pass the right parameters?
- Does it produce the right result shape?
- Does it handle expected failure paths?
- Does each local algorithm behave correctly?

### Integration test tiers

Integration tests are marked with `@pytest.mark.integration`. They are collected by `pytest`, but skip unless explicit run flags and required environment variables are set.

There are four tiers, each with its own marker and run flag:

| Tier | Marker | Run flag | What it proves |
| --- | --- | --- | --- |
| **Provider** | `@pytest.mark.provider` | `RUN_PROVIDER_INTEGRATION_TESTS` | Real LLM/embedder endpoints return usable responses. |
| **Database** | `@pytest.mark.database` | `RUN_DATABASE_INTEGRATION_TESTS` | Real Neo4j/Memgraph stores persist and query correctly. |
| **Workflow** | `@pytest.mark.workflow` | `RUN_WORKFLOW_INTEGRATION_TESTS` | Full pipeline (extraction → build → community → search) works against real databases with deterministic fake AI. No provider cost. |
| **E2E** | `@pytest.mark.e2e` | `RUN_E2E_INTEGRATION_TESTS` | Full pipeline works end-to-end with real LLM, real embedder, and real database. Highest cost. |

These tests may be slower, flaky if external services are unhealthy, and may incur provider cost.

> **Warning:** If your `.env` file sets integration run flags, those tests will attempt to run against live services under plain `pytest`. Always use `pytest -m "not integration"` for local-only development.

---

## Dependency classification

| Test area | Real LLM | Real embeddings | Real graph DB | What it proves |
| --- | ---: | ---: | ---: | --- |
| `tests/extraction/` | No | No | No | JSON parsing, schema validation, chunking, and graph document assembly. |
| `tests/llm/` | No | No | No | LLM protocol and factory behavior using fake or monkeypatched clients. |
| `tests/embeddings/` | No | No | No | Embedder protocol and factory behavior using fake or monkeypatched clients. |
| `tests/graphdb/neo4j/test_neo4j_entity_resolution.py` | Fake only | Fake only | No | Neo4j entity-resolution strategies and APOC behavior. |
| `tests/graphdb/neo4j/test_neo4j_index_manager.py` | No | No | No | Neo4j index manager and resolver-wrapper behavior. |
| `tests/graphdb/neo4j/test_neo4j_store.py` | No | No | No | Neo4j store query construction using fake driver/session objects. |
| `tests/graphdb/memgraph/` | Fake only | Fake only | No | Memgraph store, index, and entity-resolution behavior using fakes. |
| `tests/pipelines/neo4j/test_neo4j_writer.py` | No | No | No | Shared graph-writer contract against the Neo4j writer. |
| `tests/pipelines/memgraph/test_memgraph_writer.py` | No | No | No | The same graph-writer contract against the Memgraph writer. |
| `tests/pipelines/test_graph_builder.py` | Fake only | Fake only | No | Pipeline orchestration with fake LLM, fake embedder, fake graph store, and fake writer. |
| `tests/pipelines/test_writer_characterization.py` | No | No | No | Shared `BaseGraphWriter` behavior across backends. |
| `tests/graphdb/test_store_base.py` | No | No | No | Shared `BaseGraphStore` helpers and read helpers. |
| `tests/graphdb/test_entity_resolution_contract.py` | No | No | No | Shared `BaseEntityResolver` normalization contract. |
| `tests/communities/` | No | Fake only | No | Shared, Neo4j, and Memgraph community behavior using fakes. |
| `tests/retrieval/` | Fake only | Fake only | No | Retrieval, ranking, and answer-generation flow using fake LLM/embedder/store. |
| `tests/test_provider_compat.py` | No | No | No | Backend-neutral provider compatibility helpers. |
| `tests/integration/test_azure_openai_env.py` | Yes | Yes | No | Real Azure OpenAI LLM and embedding endpoint checks. |
| `tests/integration/test_openrouter_env.py` | Yes | Yes | No | Real OpenRouter LLM and embedding endpoint checks. |
| `tests/integration/test_llm_extraction.py` | Yes | No | No | Real LLM entity extraction through `GraphBuilderPipeline` without a graph database. |
| `tests/integration/neo4j/test_neo4j_entity_resolution_integration.py` | Optional | Optional | Yes | Real Neo4j entity resolution; real LLM/embedder only with the AI flag. |
| `tests/integration/memgraph/test_memgraph_entity_resolution_integration.py` | Optional | Optional | Yes | The same entity-resolution scenarios against Memgraph. |
| `tests/integration/neo4j/test_neo4j_community_detection_integration.py` | No | No | Yes | Weighted Leiden community detection through APOC/GDS. |
| `tests/integration/memgraph/test_memgraph_community_detection_integration.py` | No | No | Yes | The same weighted Leiden scenario through MAGE. |
| `tests/integration/neo4j/test_neo4j_store_smoke.py` | No | No | Yes | Scoped graph-document write/read checks against Neo4j. |
| `tests/integration/test_smoke_full.py` | Yes | Yes | Yes | Synthetic corpus E2E smoke test against Neo4j and Memgraph. |
| `tests/integration/test_workflow_deterministic.py` | Fake only | Fake only | Yes | Deterministic pipeline workflow against real databases. No provider cost. |
| `tests/integration/memgraph/test_memgraph_store_smoke.py` | No | No | Yes | Scoped graph-document write/read checks against Memgraph. |
| `tests/communities/test_summarization.py` | No | Fake only | No | Community summarization and report generation using fake LLM. |
| `tests/communities/test_context.py` | No | No | No | Community context formatting and packing. |
| `tests/communities/test_pipeline.py` | No | Fake only | No | `CommunityPipeline` orchestration with fake store and LLM. |
| `tests/communities/test_reports.py` | No | Fake only | No | Structured report rubric and output shape. |
| `tests/models/test_artifacts.py` | No | No | No | `GraphDocument`, `Citation`, and `DocumentSource` models. |
| `tests/evaluation/test_schemas.py` | No | No | No | Evaluation runner schema definitions. |
| `tests/evaluation/test_runner.py` | No | No | No | Evaluation runner orchestration. |

Meaning:

- **No**: the dependency is not used.
- **Fake only**: the dependency is represented by a fake or mock.
- **Optional**: some tests run without it; extra scenarios use it when a flag is enabled.
- **Yes**: the test requires the real dependency when enabled.

---

## Scenario dependency matrix

| Scenario | Real LLM | Real embedder | Real database | Cost/network risk |
| --- | ---: | ---: | ---: | --- |
| Mandatory unit tests | No | No | No | None |
| Azure OpenAI endpoint checks | Yes | Yes | No | Provider calls |
| OpenRouter endpoint checks | Yes | Yes | No | Provider calls |
| LLM extraction integration | Yes | No | No | Provider calls |
| Store persistence smoke | No | No | Yes | Database only |
| Community detection integration | No | No | Yes | Database only |
| Entity resolution without AI flag | No | No | Yes | Database only |
| Entity resolution with AI flag | Yes | Yes | Yes | Provider calls and database writes |
| Deterministic workflow | Fake | Fake | Yes | Database only; no provider cost |
| Synthetic E2E | Yes | Yes | Yes | Full cost; synthetic corpus |

Provider endpoint tests are fixed to the provider named by the test file. Database-plus-AI tests use `LLM_PROVIDER` and `EMBEDDER_PROVIDER` after their run flags are enabled; they do not probe providers automatically.

Supported workflow selections include:

```text
LLM_PROVIDER=azure_openai       EMBEDDER_PROVIDER=azure_openai
LLM_PROVIDER=openai             EMBEDDER_PROVIDER=openai
LLM_PROVIDER=openrouter         EMBEDDER_PROVIDER=openrouter
LLM_PROVIDER=openrouter         EMBEDDER_PROVIDER=sentence-transformer
```

The selected provider variables must also be configured:

| Selection | Required variables |
| --- | --- |
| Azure OpenAI LLM | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_LLM_DEPLOYMENT_NAME` |
| Azure OpenAI embedder | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME` |
| OpenAI LLM | `OPENAI_API_KEY`, `OPENAI_LLM_MODEL` |
| OpenAI embedder | `OPENAI_API_KEY`, `OPENAI_EMBED_MODEL` |
| OpenRouter LLM | `OPENROUTER_API_KEY`, `OPENROUTER_LLM_MODEL` |
| OpenRouter embedder | `OPENROUTER_API_KEY`, `OPENROUTER_EMBED_MODEL` |
| Sentence-Transformers embedder | No provider credentials; the local model dependency must be installed |

---

## Recommended commands

### Mandatory local suite

```bash
pytest -m "not integration"
```

### Provider CI opt-in

```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest -m provider -q
```

### Database CI opt-in

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest -m database -q
```

### Workflow CI opt-in (deterministic, no provider cost)

```bash
RUN_WORKFLOW_INTEGRATION_TESTS=1 pytest -m workflow -q
```

### Nightly / full E2E

```bash
RUN_E2E_INTEGRATION_TESTS=1 pytest -m e2e -q
```

### Full skip-safe suite

```bash
pytest
```

This runs mandatory tests plus optional tests that skip unless enabled.

---

## Scenario recipes

### Provider only, without a database

```bash
# Azure OpenAI LLM and embedding endpoints
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py -q

# OpenRouter LLM and embedding endpoints
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py -q
```

### LLM extraction without a database

```bash
# Azure OpenAI
LLM_PROVIDER=azure_openai RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py -v

# OpenRouter
LLM_PROVIDER=openrouter RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py -v
```

### Database only, without real LLM or embedding calls

```bash
# Neo4j
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_store_smoke.py -q
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_community_detection_integration.py -q
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py -q

# Memgraph
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_store_smoke.py -q
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_community_detection_integration.py -q
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py -q
```

### Deterministic workflow (real database, fake AI)

```bash
# Neo4j
RUN_WORKFLOW_INTEGRATION_TESTS=1 pytest tests/integration/test_workflow_deterministic.py -q

# Memgraph
RUN_WORKFLOW_INTEGRATION_TESTS=1 pytest tests/integration/test_workflow_deterministic.py -q
```

### Synthetic E2E (real everything, compact corpus)

```bash
# Neo4j
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_E2E_INTEGRATION_TESTS=1 \
pytest tests/integration/test_smoke_full.py -k synthetic -q

# Memgraph
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_E2E_INTEGRATION_TESTS=1 \
pytest tests/integration/test_smoke_full.py -k synthetic -q
```

In PowerShell, set the same values with `$env:NAME="value"` on separate lines before invoking `pytest`.

---

## Standard commands

### Mandatory suite

```bash
pytest -m "not integration"
```

### Optional suite in skip-safe mode

```bash
pytest -m integration
```

Tests will skip unless their run flags and required env vars are configured.

### Full skip-safe suite

```bash
pytest
```

This runs mandatory tests plus optional tests that skip unless enabled.

### Full suite with enabled optional tests (Bash)

```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 \
RUN_DATABASE_INTEGRATION_TESTS=1 \
RUN_WORKFLOW_INTEGRATION_TESTS=1 \
RUN_E2E_INTEGRATION_TESTS=1 \
pytest
```

### Full suite with enabled optional tests (PowerShell)

```powershell
$env:RUN_PROVIDER_INTEGRATION_TESTS="1"
$env:RUN_DATABASE_INTEGRATION_TESTS="1"
$env:RUN_WORKFLOW_INTEGRATION_TESTS="1"
$env:RUN_E2E_INTEGRATION_TESTS="1"
pytest
```

Only enable optional flags for services you have configured. Provider checks may incur API cost.

---

## Provider integration tests

### Azure OpenAI

```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py
```

Required environment variables:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
```

### OpenRouter

```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py
```

Required environment variables:

```text
OPENROUTER_API_KEY
OPENROUTER_LLM_MODEL
OPENROUTER_EMBED_MODEL
```

### Neo4j entity resolution

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py
```

Required environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

### Memgraph entity resolution

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py
```

Required environment variable: `MEMGRAPH_URL`. Enable `RUN_ENTITY_RESOLUTION_AI_TESTS=1` and configure the selected provider variables to include the real AI review scenario.

### Neo4j community detection

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_community_detection_integration.py
```

Required database environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

### Memgraph community detection

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_community_detection_integration.py
```

Required environment variable: `MEMGRAPH_URL`. The configured instance must include MAGE.

### Neo4j store smoke test

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_store_smoke.py
```

Required environment variables: `NEO4J_URL`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.

### Memgraph store smoke test

```bash
RUN_DATABASE_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_store_smoke.py
```

Required environment variable: `MEMGRAPH_URL`. Authentication and database variables are optional.

---

## Environment setup

Install development dependencies. We recommend using `uv`:

```bash
uv sync --extra dev --group dev
```

Or with `pip`:

```bash
pip install -e ".[dev]"
# `dotenv` is managed through uv's dependency group; install it manually
# when running tests or examples that load environment files:
pip install python-dotenv
```

Copy `.env.example` to `.env` and fill only the values needed for the optional tests you intend to run.

Run the test suite with `uv`:

```bash
uv run pytest -m "not integration"
```

or with `pytest` directly if you installed into an activated virtual environment.

---

## Notes

- `pytest` uses `pythonpath = ["."]` from `pyproject.toml`, so the local package imports from the repository checkout.
- Integration tests use test-owned factories (`tests/integration/factories.py`) for graph stores, LLMs, and embedders. They do not depend on `examples/`.
- Integration tests should stay behind explicit run flags.
- `pyproject.toml` also defines a `characterization` marker for tests that document current behavior, including known defects. Use it for tests that pin behavior without asserting ideal correctness.
- New external-service tests should use `@pytest.mark.integration` and skip unless explicitly enabled.
- New graph database backends should get backend-specific integration tests under `tests/integration/<backend>/`.
- Use the [graph backend checklist](../plans/backends.md) when adding a new graph database backend.
