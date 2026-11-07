# Movie Industry Example

An end-to-end movie-domain GraphRAG workflow for Neo4j and Memgraph.

## Setup

Run these setup commands from the repository root:

```bash
uv sync --extra dev --group dev
cp .env.example .env
```

Set provider credentials in `.env`. Use a disposable database. The Compose
services expose Neo4j at `bolt://localhost:7688` and Memgraph at
`bolt://localhost:7689`; the example helpers otherwise default to port 7687.
Keep `NEO4J_URL` and `MEMGRAPH_URL` consistent with your database setup.
Then run the example commands from `examples/` with the environment active,
or prefix each Python command with `uv run`.

## Quick start

```bash
# 0. (Optional) Let the LLM propose a schema from the sample corpus
python analyze_schema.py --llm-provider openrouter

# 1. Extract the movie graph into a neutral JSON artifact
python extract.py --llm-provider openrouter

# 2. Ingest into one or all graph backends
python ingest.py --backend all --embedder-provider openrouter --llm-provider openrouter

# 3. Build communities
python communities.py --backend all --llm-provider openrouter

# 4. Run the movie query suite
python search.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter

# 5. Compare Neo4j and Memgraph outputs (advanced)
python advanced/compare_backends.py
```

## Provider flags

All build and search scripts accept `--llm-provider` and `--embedder-provider`:

| Provider | LLM | Embedder |
| --- | --- | --- |
| `azure_openai` | Yes | Yes |
| `openrouter` | Yes | Yes |
| `openai` | Yes | Yes |
| `sentence-transformer` | No | Yes |

You can also set `LLM_PROVIDER` and `EMBEDDER_PROVIDER` environment variables.
If no provider is specified, scripts default to `openrouter`.

## Artifact workflow

The example separates extraction from ingestion, then builds and queries communities:

1. **Extract once** (`extract.py`) — runs LLM extraction and saves a `GraphDocument` JSON artifact to `artifacts/movie_graph.json`.
2. **Ingest into backends** (`ingest.py`) — loads the artifact and writes it to Neo4j and/or Memgraph, then resolves duplicate entities, embeds entity nodes, and validates the build.
3. **Build communities** (`communities.py`) — runs Leiden community detection and LLM summarization.
4. **Search** (`search.py`) — runs local, global, and DRIFT retrieval against the built graph.
5. **Compare** (`advanced/compare_backends.py`) — side-by-side comparison of Neo4j and Memgraph retrieval quality.

This separation lets you extract once and experiment with multiple backends without re-running the LLM extraction.

## Corpus metadata

Each page in `data.py` carries a `metadata` dict with `source`, `topic`, and `page_index` fields. This metadata is passed through the chunking step into the assembled `GraphDocument` so the artifact preserves provenance for each part of the corpus.

## Custom prompts

`prompts.py` overrides the SDK's neutral defaults with film-analyst language at every stage:

- **Extraction** (`extract.py`) — `EXTRACTION_PROMPT` is passed to `LLMGraphExtractor` via `SchemaPromptBuilder`.
- **Community reports** (`communities.py`) — `COMMUNITY_REPORT_PROMPT` is passed to `CommunityPipeline` as `report_prompt`.
- **Retrieval** (`search.py`) — full prompt templates for the local, global, and DRIFT answer stages.

The build-time prompts are plain instruction strings; the backend appends the standard schema, rules, JSON format, and rubric sections. Run `python custom_prompts.py` for a standalone tour of the custom-prompt API that prints the exact prompts sent to the LLM.

## Note

This directory is sample code for hands-on experimentation. The test suite does not import from `examples/`; integration tests use their own test-owned factories under `tests/integration/factories.py`.
