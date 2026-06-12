# Test Guide

This guide explains what each test group proves, which tests use fake dependencies, which tests call real services, and which command to run based on the kind of change you are making.

The short version:

- Run mandatory tests while developing: `pytest -m "not integration"`
- Run provider integration tests when changing provider setup.
- Run focused Neo4j integration tests when changing real database behavior.
- Run the Neo4j movie smoke test when you need full end-to-end confidence.

## Testing Model

The suite is intentionally split into layers.

### Mandatory Tests

Mandatory tests are fast, local, and deterministic. They do not require real provider credentials or a live Neo4j database.

They use:

- fake graph stores
- fake Neo4j drivers/sessions
- fake LLMs
- fake embedders
- monkeypatched provider clients
- in-memory assertions

These tests answer:

```text
Does our code call the right method?
Does it pass the right parameters?
Does it produce the right result shape?
Does it handle expected failure paths?
Does each local algorithm behave correctly?
```

They do not answer:

```text
Does Azure/OpenRouter actually respond?
Does Neo4j/APOC/GDS actually work?
Does the whole graph workflow work against real infrastructure?
```

### Optional Integration Tests

Optional tests are marked with `@pytest.mark.integration`. They are collected by `pytest`, but skip unless explicit run flags and required env vars are set.

They answer:

```text
Does this work against real external services?
Does this work against real Neo4j?
Does the complete workflow still run end to end?
```

These tests may be slower, flaky if external services are unhealthy, and may incur provider cost.

## Dependency Classification

Use this table to understand whether a test area uses real LLM, real embeddings, or a real graph database.

| Test area | Real LLM | Real embeddings | Real graph DB | What it proves |
|---|---:|---:|---:|---|
| `tests/extraction/` | No | No | No | JSON parsing, schema validation, chunking, and graph document assembly. |
| `tests/llm/` | No | No | No | LLM protocol and factory behavior using fake or monkeypatched clients. |
| `tests/embeddings/` | No | No | No | Embedder protocol and factory behavior using fake or monkeypatched clients. |
| `tests/graphdb/neo4j/test_entity_resolution.py` | Fake only | Fake only | No | Entity resolution strategy logic without real Neo4j or real models. |
| `tests/graphdb/neo4j/test_index_manager.py` | No | No | No | Index manager index creation and resolver wrapper behavior. |
| `tests/graphdb/neo4j/test_neo4j_store.py` | No | No | No | Neo4j store query construction using fake driver/session objects. |
| `tests/pipelines/neo4j/test_writer.py` | No | No | No | Neo4j graph writer query generation using a fake graph store. |
| `tests/pipelines/test_graph_builder.py` | Fake only | Fake only | No | Pipeline orchestration with fake LLM, fake embedder, fake graph store, and fake writer. |
| `tests/communities/` | No | Fake only | No | Community embedding and Neo4j community detection query behavior using fakes. |
| `tests/retrieval/` | Fake only | Fake only | No | Retrieval, ranking, and answer-generation flow using fake LLM/embedder/store. |
| `tests/integration/test_azure_openai_env.py` | Yes | Yes | No | Real Azure OpenAI LLM and embedding endpoint checks. |
| `tests/integration/test_openrouter_env.py` | Yes | Yes | No | Real OpenRouter LLM and embedding endpoint checks. |
| `tests/integration/neo4j/test_entity_resolution_integration.py` | Optional | Optional | Yes | Real Neo4j entity resolution; real LLM/embedder only with the AI flag. |
| `tests/integration/test_movie_example_smoke.py` | Yes | Yes | Yes | Neo4j end-to-end movie example: real provider + real Neo4j + APOC/GDS. |
| `tests/manual/` | Yes | Yes | No | Manual diagnostics; not part of normal test guidance. |

Meaning:

- **No**: the dependency is not used.
- **Fake only**: the dependency is represented by a fake or mock.
- **Optional**: some tests in the file run without it; extra scenarios use it when a flag is enabled.
- **Yes**: the test requires the real dependency when enabled.

## What Each Major Test Group Does

### `tests/extraction/`

Tests the text-to-graph preprocessing and validation layer.

Important coverage:

- chunking raw text and paginated text
- parsing model JSON output
- dropping malformed nodes/relationships
- enforcing schema labels and relationship patterns
- choosing identity properties
- assembling chunks, entities, relationships, and evidence links

Run this when changing:

- prompts output shape
- parser behavior
- schema validation
- chunking
- graph document assembly

Command:

```bash
pytest tests/extraction
```

### `tests/graphdb/neo4j/`

Tests Neo4j-specific logic without connecting to a real Neo4j server.

Important coverage:

- Cypher generation
- index creation wrapper behavior
- vector upsert query shape
- entity resolution strategies
- APOC-unavailable behavior
- dry-run behavior

Entity resolution coverage includes:

- `exact`
- `normalized`
- `fuzzy`
- `hybrid`
- user aliases
- fake LLM review with `llm_guidance`
- fake embedding review scoring
- dry-run behavior
- APOC unavailable behavior

Run this when changing:

- `Neo4jGraphStore`
- `IndexManager`
- `graphdb/neo4j/entity_resolution.py`
- Neo4j query construction

Command:

```bash
pytest tests/graphdb/neo4j
```

Focused entity-resolution command:

```bash
pytest tests/graphdb/neo4j/test_entity_resolution.py tests/graphdb/neo4j/test_index_manager.py tests/pipelines/test_graph_builder.py
```

### `tests/pipelines/`

Tests SDK orchestration.

Important coverage:

- `GraphBuilderPipeline.build_from_text`
- `GraphBuilderPipeline.build_from_pages`
- `GraphBuilderPipeline.build_from_documents`
- failure behavior when extraction fails
- entity embedding loop wiring
- graph name behavior
- Neo4j writer query grouping

The pipeline tests use fake LLMs, fake embedders, and fake graph stores. They prove orchestration, not real provider/database behavior.

Run this when changing:

- graph build orchestration
- pipeline constructor parameters
- build-from-pages behavior
- graph writer behavior

Command:

```bash
pytest tests/pipelines
```

### `tests/retrieval/`

Tests retrieval and answer-generation flow with fake dependencies.

Important coverage:

- hybrid score merging
- ranker validation
- local search retriever behavior
- precomputed query vector behavior
- local search LLM answer flow
- community-level aliases for global and DRIFT search

Run this when changing:

- retrieval ranking
- local/global/DRIFT search behavior
- community level handling

Command:

```bash
pytest tests/retrieval
```

### `tests/llm/` And `tests/embeddings/`

Tests provider wrapper construction and protocol behavior without real provider calls.

Important coverage:

- local LLM/embedder protocol compatibility
- Azure deployment-name binding behavior
- explicit deployment override behavior
- model parameter forwarding

Run this when changing:

- provider factories
- OpenAI/Azure/OpenRouter/Ollama wrapper setup
- embedder wrapper behavior

Command:

```bash
pytest tests/llm tests/embeddings
```

## Optional Integration Tests

Integration tests are disabled by default. They use run flags so accidental `pytest` runs do not call external services.

All integration tests load `.env` with `python-dotenv`.

### Azure OpenAI Provider Checks

File:

```text
tests/integration/test_azure_openai_env.py
```

What it does:

- creates a real Azure OpenAI LLM client
- calls the real LLM endpoint
- creates a real Azure OpenAI embedder
- calls the real embedding endpoint
- asserts a non-empty response/vector

It does not use Neo4j.

Run:

```bash
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 pytest tests/integration/test_azure_openai_env.py
```

Required env vars:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
```

### OpenRouter Provider Checks

File:

```text
tests/integration/test_openrouter_env.py
```

What it does:

- creates a real OpenRouter LLM client
- calls the real LLM endpoint
- creates a real OpenRouter embedder
- calls the real embedding endpoint

It does not use Neo4j.

Run:

```bash
RUN_OPENROUTER_INTEGRATION_TESTS=1 pytest tests/integration/test_openrouter_env.py
```

Required env vars:

```text
OPENROUTER_API_KEY
OPENROUTER_LLM_MODEL
OPENROUTER_EMBED_MODEL
```

### Focused Neo4j Entity Resolution Integration

File:

```text
tests/integration/neo4j/test_entity_resolution_integration.py
```

What it does with only Neo4j enabled:

- connects to real Neo4j
- checks APOC availability
- seeds duplicate entity nodes directly
- runs `store.resolve_entities(strategy="normalized")`
- verifies real nodes are merged
- runs `strategy="hybrid"` with user aliases in dry-run mode
- verifies candidates are found without merging

This mode does not call a real LLM or real embedder.

Run:

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_entity_resolution_integration.py
```

Required env vars:

```text
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

What it does with the AI flag enabled:

- uses real Neo4j
- uses real Azure OpenAI embedder
- uses real Azure OpenAI LLM
- runs hybrid entity resolution in dry-run mode
- verifies embedding score is produced
- verifies LLM review is produced

Run:

```bash
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
pytest tests/integration/neo4j/test_entity_resolution_integration.py
```

Additional required env vars:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
```

Run this when changing:

- `strategy="hybrid"`
- embedding-assisted candidate scoring
- LLM review prompt shape
- `llm_guidance`
- alias evidence sent into LLM review

### Neo4j Movie Example Smoke Test

File:

```text
tests/integration/test_movie_example_smoke.py
```

What it does:

- connects to real Neo4j
- checks APOC and GDS availability
- creates indexes
- builds a movie graph from paginated text
- runs entity resolution
- embeds entity nodes
- detects communities
- summarizes communities with real LLM
- embeds communities
- runs Local, Global, and DRIFT search
- asserts answers and contexts are non-empty

This is the primary real Neo4j end-to-end test.

Run:

```bash
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/test_movie_example_smoke.py
```

Required env vars:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT_NAME
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME
NEO4J_URL
NEO4J_USERNAME
NEO4J_PASSWORD
```

It requires Neo4j with APOC and GDS.

Run this when you need to verify the whole real workflow:

```text
GraphBuilderPipeline -> entity resolution -> entity embedding
-> CommunityPipeline -> community embedding
-> GraphRAG local/global/DRIFT search
```

When another graph database backend is added, add a separate backend-specific smoke test instead of reusing this Neo4j smoke test.

## What To Run By Change Type

| Change type | Recommended command |
|---|---|
| Parser, schema, validator, chunking, assembler | `pytest tests/extraction` |
| LLM wrapper/factory code | `pytest tests/llm` |
| Embedding wrapper/factory code | `pytest tests/embeddings` |
| Entity deduplication local logic | `pytest tests/graphdb/neo4j/test_entity_resolution.py tests/graphdb/neo4j/test_index_manager.py tests/pipelines/test_graph_builder.py` |
| Entity deduplication with real Neo4j | `RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 pytest tests/integration/neo4j/test_entity_resolution_integration.py` |
| Hybrid dedup with real Neo4j + real LLM/embedder | enable both entity-resolution integration flags and run `pytest tests/integration/neo4j/test_entity_resolution_integration.py` |
| Neo4j store, indexes, writer queries | `pytest tests/graphdb/neo4j tests/pipelines/neo4j` |
| Graph builder pipeline | `pytest tests/pipelines/test_graph_builder.py tests/extraction tests/graphdb/neo4j/test_entity_resolution.py` |
| Community detection/embedding helpers | `pytest tests/communities` |
| Retrieval or search behavior | `pytest tests/retrieval` |
| Provider credentials or provider request shape | run `pytest tests/llm tests/embeddings`, then the relevant provider integration test |
| Real Neo4j end-to-end graph build/search | `RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 pytest tests/integration/test_movie_example_smoke.py` |
| Before normal commit | `pytest -m "not integration"` |
| Before release or major workflow change | `pytest`, then enabled integration tests for configured providers/services |

## Standard Commands

### Mandatory Suite

Run all mandatory tests:

```bash
pytest -m "not integration"
```

### Optional Suite

Run all optional tests in skip-safe mode:

```bash
pytest -m integration
```

Tests will skip unless their run flags and required env vars are configured.

### Full Skip-Safe Suite

Run mandatory tests plus optional tests that skip unless enabled:

```bash
pytest
```

### Full Suite With Enabled Optional Tests

Bash:

```bash
RUN_AZURE_OPENAI_INTEGRATION_TESTS=1 \
RUN_OPENROUTER_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS=1 \
RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS=1 \
RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS=1 \
pytest
```

PowerShell:

```powershell
$env:RUN_AZURE_OPENAI_INTEGRATION_TESTS="1"
$env:RUN_OPENROUTER_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_INTEGRATION_TESTS="1"
$env:RUN_NEO4J_ENTITY_RESOLUTION_AI_TESTS="1"
$env:RUN_NEO4J_MOVIE_EXAMPLE_SMOKE_TESTS="1"
pytest
```

Only enable optional flags for services you have configured. Provider checks may incur API cost.

## Environment

Install development dependencies:

```bash
pip install -e ".[dev]"
```

or with `uv`:

```bash
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill only the values needed for the optional tests you intend to run.

## Notes

- `pytest` uses `pythonpath = ["."]` from `pyproject.toml`, so the local package imports from the repository checkout.
- Integration tests should stay behind explicit run flags.
- New external-service tests should use `@pytest.mark.integration` and skip unless explicitly enabled.
- New graph database backends should get backend-specific integration tests under `tests/integration/<backend>/`.
