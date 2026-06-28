# Movie Industry Example

An end-to-end movie-domain GraphRAG workflow for Neo4j and Memgraph.

## Quick start

```bash
# 1. Extract the movie graph into a neutral JSON artifact
python extract.py --llm-provider openrouter

# 2. Ingest into one or all graph backends
python ingest.py --backend all --embedder-provider openrouter

# 3. Build communities
python communities.py --backend all --llm-provider openrouter

# 4. Run the movie query suite
python search.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter

# 5. Compare Neo4j and Memgraph outputs
python compare_backends.py
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

## Artifact workflow

The example uses a two-phase approach:

1. **Extract once** (`extract.py`) — runs LLM extraction and saves a `GraphDocument` JSON artifact to `artifacts/movie_graph.json`.
2. **Ingest into backends** (`ingest.py`) — loads the artifact and writes it to Neo4j and/or Memgraph, with optional entity resolution and embedding.
3. **Build communities** (`communities.py`) — runs Leiden community detection and LLM summarization.
4. **Search** (`search.py`) — runs local, global, and DRIFT retrieval against the built graph.
5. **Compare** (`compare_backends.py`) — side-by-side comparison of Neo4j and Memgraph retrieval quality.

This separation lets you extract once and experiment with multiple backends without re-running the LLM extraction.

## Entity resolution

`ingest.py` supports multiple entity-resolution strategies:

```bash
python ingest.py --backend neo4j --entity-resolution-strategy normalized
python ingest.py --backend neo4j --entity-resolution-strategy hybrid --allow-ai-auto-merge
```

## Note

This directory is sample code for hands-on experimentation. The test suite does not import from `examples/`; integration tests use their own test-owned factories under `tests/integration/factories.py`.
