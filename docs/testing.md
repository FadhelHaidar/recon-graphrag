# Testing

This guide explains how to run the Recon-GraphRAG test suite. It covers mandatory tests, optional integration tests, and which commands to run for different kinds of changes.

The short version:

- Run mandatory tests while developing: `pytest -m "not integration"`
- Run provider integration tests when changing provider setup.
- Run focused integration tests for the database whose behavior changed.
- Run the matching Neo4j or Memgraph movie smoke test for end-to-end confidence.

## Testing model

The suite is split into two layers.

### Mandatory tests

Mandatory tests are fast, local, and deterministic. They do not require real provider credentials or a live graph database.

They use fake graph stores, fake Bolt drivers/sessions, fake LLMs, fake embedders, monkeypatched provider clients, and in-memory assertions.

They answer questions like:

- Does our code call the right method?
- Does it pass the right parameters?
- Does it produce the right result shape?
- Does it handle expected failure paths?
- Does each local algorithm behave correctly?

### Optional integration tests

Integration tests are marked with `@pytest.mark.integration`. They are collected by `pytest`, but skip unless explicit run flags and required environment variables are set.

They answer questions like:

- Does this work against real external services?
- Does this work against real Neo4j or Memgraph?
- Does the complete workflow still run end to end?

These tests may be slower, flaky if external services are unhealthy, and may incur provider cost.

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
| `tests/communities/` | No | Fake only | No | Shared, Neo4j, and Memgraph community behavior using fakes. |
| `tests/retrieval/` | Fake only | Fake only | No | Retrieval, ranking, and answer-generation flow using fake LLM/embedder/store. |
| `tests/integration/test_azure_openai_env.py` | Yes | Yes | No | Real Azure OpenAI LLM and embedding endpoint checks. |
| `tests/integration/test_openrouter_env.py` | Yes | Yes | No | Real OpenRouter LLM and embedding endpoint checks. |
| `tests/integration/test_llm_extraction.py` | Yes | No | No | Real LLM entity extraction through `GraphBuilderPipeline` without a graph database. |
| `tests/integration/neo4j/test_neo4j_entity_resolution_integration.py` | Optional | Optional | Yes | Real Neo4j entity resolution; real LLM/embedder only with the AI flag. |
| `tests/integration/memgraph/test_memgraph_entity_resolution_integration.py` | Optional | Optional | Yes | The same entity-resolution scenarios against Memgraph. |
| `tests/integration/neo4j/test_neo4j_community_detection_integration.py` | No | No | Yes | Weighted Leiden community detection through APOC/GDS. |
| `tests/integration/memgraph/test_memgraph_community_detection_integration.py` | No | No | Yes | The same weighted Leiden scenario through MAGE. |
| `tests/integration/neo4j/test_neo4j_store_smoke.py` | No | No | Yes | Scoped graph-document write/read checks against Neo4j. |
| `tests/integration/neo4j/test_neo4j_movie_smoke.py` | Yes | Yes | Yes | Neo4j end-to-end movie example with APOC/GDS. |
| `tests/integration/memgraph/test_memgraph_store_smoke.py` | No | No | Yes | Scoped graph-document write/read checks against Memgraph. |
| `tests/integration/memgraph/test_memgraph_movie_smoke.py` | Yes | Yes | Yes | Memgraph end-to-end movie example with MAGE. |
| `tests/integration/test_smoke_full.py` | Yes | Yes | Yes | Backend-neutral full movie workflow against Neo4j and Memgraph when both run flags are enabled. |

Meaning:

- **No**: the dependency is not used.
- **Fake only**: the dependency is represented by a fake or mock.
- **Optional**: some tests run without it; extra scenarios use it when a flag is enabled.
- **Yes**: the test requires the real dependency when enabled.

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
| Movie workflow smoke | Yes | Yes | Yes | Highest cost; extraction, embeddings, summaries, and searches |
| Backend-neutral full smoke | Yes | Yes | Yes | Runs the same graph build, community build, local/global/DRIFT/paper search path on Neo4j and Memgraph |

Provider endpoint tests are fixed to the provider named by the test file. Database-plus-AI tests use `LLM_PROVIDER` and `EMBEDDER_PROVIDER` after their run flags are enabled; they do not probe providers automatically.

Supported workflow selections include:

```text
LLM_PROVIDER=azure_openai       EMBEDDER_PROVIDER=azure_openai
LLM_PROVIDER=openrouter         EMBEDDER_PROVIDER=openrouter
LLM_PROVIDER=openrouter         EMBEDDER_PROVIDER=sentence-transformer
```

The selected provider variables must also be configured:

| Selection | Required variables |
| --- | --- |
| Azure OpenAI LLM | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_LLM_DEPLOYMENT_NAME` |
| Azure OpenAI embedder | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME` |
| OpenRouter LLM | `OPENROUTER_API_KEY`, `OPENROUTER_LLM_MODEL` |
| OpenRouter embedder | `OPENROUTER_API_KEY`, `OPENROUTER_EMBED_MODEL` |
| Sentence-Transformers embedder | No provider credentials; the local model dependency must be installed |

### Scenario recipes

Provider only, without a database:

```bash
# Azure OpenAI LLM and embedding endpoints
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py

# OpenRouter LLM and embedding endpoints
RUN_OPENROUTER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py
```

### LLM extraction without a database

```bash
# Azure OpenAI
LLM_PROVIDER=azure_openai RUN_LLM_EXTRACTION_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py

# OpenRouter
LLM_PROVIDER=openrouter RUN_LLM_EXTRACTION_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py
```

Database only, without real LLM or embedding calls:

```bash
# Neo4j
RUN_NEO4J_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_store_smoke.py
RUN_NEO4J_COMMUNITY_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_community_detection_integration.py
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py

# Memgraph
RUN_MEMGRAPH_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_store_smoke.py
RUN_MEMGRAPH_COMMUNITY_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_community_detection_integration.py
RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py
```

AI-assisted entity resolution with Azure OpenAI and a real database:

```bash
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py

LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_MEMGRAPH_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py
```

Use OpenRouter for the same scenarios by changing both provider selections to `openrouter` and configuring the OpenRouter variables:

```bash
LLM_PROVIDER=openrouter EMBEDDER_PROVIDER=openrouter \
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py
```

Full movie workflow with the selected provider and database:

```bash
# Azure OpenAI + Neo4j
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest tests/integration/neo4j/test_neo4j_movie_smoke.py

# OpenRouter + Neo4j
LLM_PROVIDER=openrouter EMBEDDER_PROVIDER=openrouter \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest tests/integration/neo4j/test_neo4j_movie_smoke.py

# Azure OpenAI + Memgraph
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest tests/integration/memgraph/test_memgraph_movie_smoke.py

# OpenRouter + Memgraph
LLM_PROVIDER=openrouter EMBEDDER_PROVIDER=openrouter \
RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest tests/integration/memgraph/test_memgraph_movie_smoke.py
```

In PowerShell, set the same values with `$env:NAME="value"` on separate lines before invoking `pytest`.

Backend-neutral full smoke against both graph stores:

```bash
LLM_PROVIDER=azure_openai EMBEDDER_PROVIDER=azure_openai \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest tests/integration/test_smoke_full.py
```

The full smoke validates that extraction, optional claim extraction,
aggregation, community detection, summarization, embeddings, local search,
global search and DRIFT run end to end on both
backends. Because it uses live LLM output, it does not require nondeterministic
behaviors such as `claims_written > 0` or non-empty answer citations. Those
behavioral contracts are covered by focused tests.

Focused coverage for the gaps-with-paper alignment:

| Behavior | Focused tests |
| --- | --- |
| Structured report persistence: `report_json`, `report_text`, `report_status` | `tests/graphdb/neo4j/test_neo4j_store.py`, `tests/graphdb/memgraph/test_memgraph_store.py`, `tests/communities/test_summarization.py` |
| Graph-scoped claim and citation reads | `tests/graphdb/neo4j/test_neo4j_store.py`, `tests/graphdb/memgraph/test_memgraph_store.py`, `tests/retrieval/test_global_search.py` |
| Global explicit references and citations | `tests/retrieval/test_global_search.py` |
| Local and DRIFT citations from source chunks | `tests/retrieval/test_hybrid.py`, `tests/retrieval/test_community_levels.py` |
| Arbitrary source metadata on citations | `tests/models/test_artifacts.py`, `tests/retrieval/test_hybrid.py`, `tests/graphdb/neo4j/test_neo4j_store.py`, `tests/graphdb/memgraph/test_memgraph_store.py` |
| Shared token packing API: `PackItem`, `PackResult`, `pack_items` | `tests/utils/test_tokens.py` |

## What to run by change type

| Change type | Recommended command |
| --------- | --------- |
| Parser, schema, validator, chunking, assembler | `pytest tests/extraction` |
| LLM wrapper/factory code | `pytest tests/llm` |
| Embedding wrapper/factory code | `pytest tests/embeddings` |
| Entity deduplication local logic | `pytest tests/graphdb/neo4j/test_neo4j_entity_resolution.py tests/graphdb/memgraph/test_memgraph_entity_resolution.py tests/pipelines/test_graph_builder.py` |
| Entity deduplication with real Neo4j | `RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py` |
| Entity deduplication with real Memgraph | `RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py` |
| Hybrid dedup with real Neo4j + real LLM/embedder | Enable both entity-resolution integration flags and run `pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py` |
| Neo4j store, indexes, writer queries | `pytest tests/graphdb/neo4j tests/pipelines/neo4j` |
| Memgraph store, indexes, writer queries | `pytest tests/graphdb/memgraph tests/pipelines/memgraph` |
| Graph builder pipeline | `pytest tests/pipelines/test_graph_builder.py tests/extraction tests/graphdb` |
| Real LLM extraction through pipeline | `LLM_PROVIDER=azure_openai RUN_LLM_EXTRACTION_INTEGRATION_TESTS=1 pytest tests/integration/test_llm_extraction.py` |
| Community detection/embedding helpers | `pytest tests/communities` |
| Retrieval or search behavior | `pytest tests/retrieval` |
| Provider credentials or provider request shape | Run `pytest tests/llm tests/embeddings`, then the relevant provider integration test |
| Real Neo4j end-to-end graph build/search | `RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/neo4j/test_neo4j_movie_smoke.py` |
| Real Neo4j store behavior | `RUN_NEO4J_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_store_smoke.py` |
| Real Memgraph store behavior | `RUN_MEMGRAPH_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_store_smoke.py` |
| Real Memgraph end-to-end graph build/search | `RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/memgraph/test_memgraph_movie_smoke.py` |
| Before normal commit | `pytest -m "not integration"` |
| Before release or major workflow change | `pytest`, then enabled integration tests for configured providers/services |

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
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 \
RUN_OPENROUTER_INTEGRATION_TESTS=1 \
RUN_NEO4J_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
RUN_NEO4J_COMMUNITY_INTEGRATION_TESTS=1 \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
RUN_MEMGRAPH_INTEGRATION_TESTS=1 \
RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_MEMGRAPH_ENTITY_RESOLUTION_AI_TESTS=1 \
RUN_MEMGRAPH_COMMUNITY_INTEGRATION_TESTS=1 \
RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest
```

### Full suite with enabled optional tests (PowerShell)

```powershell
$env:RUN_AZURE_OPENAI_INTEGRATION_TESTS="1"
$env:RUN_OPENROUTER_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS="1"
$env:RUN_NEO4J_COMMUNITY_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS="1"
$env:RUN_MEMGRAPH_INTEGRATION_TESTS="1"
$env:RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS="1"
$env:RUN_MEMGRAPH_ENTITY_RESOLUTION_AI_TESTS="1"
$env:RUN_MEMGRAPH_COMMUNITY_INTEGRATION_TESTS="1"
$env:RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS="1"
pytest
```

Only enable optional flags for services you have configured. Provider checks may incur API cost.

## Provider integration tests

### Azure OpenAI

```bash
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py
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
RUN_OPENROUTER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py
```

Required environment variables:

```text
OPENROUTER_API_KEY
OPENROUTER_LLM_MODEL
OPENROUTER_EMBED_MODEL
```

### Neo4j entity resolution

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py
```

Required environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

With the selected real LLM and embedder:

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/neo4j/test_neo4j_entity_resolution_integration.py
```

### Memgraph entity resolution

```bash
RUN_MEMGRAPH_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_entity_resolution_integration.py
```

Required environment variable: `MEMGRAPH_URL`. Enable `RUN_MEMGRAPH_ENTITY_RESOLUTION_AI_TESTS=1` and configure the selected provider variables to include the real AI review scenario.

### Neo4j community detection

```bash
RUN_NEO4J_COMMUNITY_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_community_detection_integration.py
```

Required database environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

### Memgraph community detection

```bash
RUN_MEMGRAPH_COMMUNITY_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_community_detection_integration.py
```

Required environment variable: `MEMGRAPH_URL`. The configured instance must include MAGE.

### Neo4j movie example smoke test

```bash
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/neo4j/test_neo4j_movie_smoke.py
```

Required environment variables:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

The selected LLM and embedder variables from the scenario matrix are also required.

### Neo4j store smoke test

```bash
RUN_NEO4J_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_neo4j_store_smoke.py
```

Required environment variables: `NEO4J_URL`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.

### Memgraph store smoke test

```bash
RUN_MEMGRAPH_INTEGRATION_TESTS=1 pytest tests/integration/memgraph/test_memgraph_store_smoke.py
```

Required environment variable: `MEMGRAPH_URL`. Authentication and database variables are optional.

### Memgraph movie example smoke test

```bash
RUN_MEMGRAPH_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/memgraph/test_memgraph_movie_smoke.py
```

This requires `MEMGRAPH_URL` plus the selected LLM and embedder variables from the scenario matrix. The configured Memgraph instance must include MAGE.

## Environment setup

Install development dependencies. We recommend using `uv`:

```bash
uv sync --group dev
```

Or with `pip`:

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill only the values needed for the optional tests you intend to run.

Run the test suite with `uv`:

```bash
uv run pytest -m "not integration"
```

or with `pytest` directly if you installed into an activated virtual environment.

## Notes

- `pytest` uses `pythonpath = ["."]` from `pyproject.toml`, so the local package imports from the repository checkout.
- Integration tests should stay behind explicit run flags.
- New external-service tests should use `@pytest.mark.integration` and skip unless explicitly enabled.
- New graph database backends should get backend-specific integration tests under `tests/integration/<backend>/`.
