# Examples

The `examples/` directory contains complete, runnable projects that demonstrate how to use Recon-GraphRAG end-to-end.

## Movie industry example

`examples/movie_industry/` is a full GraphRAG application for the film domain. It shows how to:

- Define a domain schema with many entity and relationship types.
- Configure multiple LLM/embedder providers.
- Customize retriever prompts for a domain.
- Build a graph from paginated text.
- Run local, global, and DRIFT searches.

### Files

| File | Purpose |
| --------- | --------- |
| [schema.py](../examples/movie_industry/schema.py) | Domain schema with 10 entity types and 14 relationship types |
| [config.py](../examples/movie_industry/config.py) | Multi-provider setup (Azure OpenAI, OpenRouter, local) and store getters for Neo4j and FalkorDB |
| [prompts.py](../examples/movie_industry/prompts.py) | Domain-customized prompts for all retrievers |
| [data.py](../examples/movie_industry/data.py) | Sample movie text data |
| [build_neo4j.py](../examples/movie_industry/build_neo4j.py) | Ingestion + community pipeline runner for Neo4j |
| [build_falkordb.py](../examples/movie_industry/build_falkordb.py) | Ingestion + community pipeline runner for FalkorDB |
| [search_neo4j.py](../examples/movie_industry/search_neo4j.py) | Test queries across all three search modes on Neo4j |
| [search_falkordb.py](../examples/movie_industry/search_falkordb.py) | Test queries across all three search modes on FalkorDB |

### Provider selection

All build and search scripts accept `--llm-provider` and `--embedder-provider` flags:

```bash
python build_neo4j.py --llm-provider openrouter --embedder-provider openai
python search_falkordb.py --llm-provider openai --embedder-provider sentence-transformer
```

Supported LLM providers: `openrouter`, `azure_openai`, `openai`.  
Supported embedder providers: `openrouter`, `azure_openai`, `openai`, `sentence-transformer`.

You can also set the `LLM_PROVIDER` and `EMBEDDER_PROVIDER` environment variables instead of passing flags.

### Search modes per test case

`search_neo4j.py` and `search_falkordb.py` no longer run every query through all three search modes. Each test case in the suite declares its own `modes` list (e.g. `["local"]`, `["global", "drift"]`, `["local", "global", "drift"]`). Omitting `modes` defaults to running all three.

### Run the example with Neo4j

1. Start Neo4j with APOC and GDS:

   ```bash
   docker-compose up -d
   ```

2. Copy and fill in your environment variables:

   ```bash
   cp .env.example .env
   ```

3. Build the graph and communities:

   ```bash
   cd examples/movie_industry
   python build_neo4j.py --llm-provider openrouter --embedder-provider openrouter
   ```

4. Run test queries:

   ```bash
   python search_neo4j.py --llm-provider openrouter --embedder-provider openrouter
   ```

### Run the example with FalkorDB

1. Start FalkorDB:

   ```bash
   docker compose --profile falkordb up -d falkordb
   ```

2. Ensure `FALKORDB_HOST`, `FALKORDB_PORT`, and optionally `FALKORDB_GRAPH_NAME` are set in your environment.

3. Build the graph and communities:

   ```bash
   cd examples/movie_industry
   python build_falkordb.py --llm-provider openrouter --embedder-provider openrouter
   ```

4. Run test queries:

   ```bash
   python search_falkordb.py --llm-provider openrouter --embedder-provider openrouter
   ```

### Schema highlights

The movie schema includes entity types such as `Movie`, `Person`, `Company`, `Genre`, `Award`, and `Location`, connected by relationships such as `DIRECTED`, `ACTED_IN`, `PRODUCED_BY`, `WON`, `NOMINATED_FOR`, and `LOCATED_IN`.

### Prompt customization

`prompts.py` overrides the default answer, map, reduce, and retrieval prompts so the retrievers answer in the style of a film analyst. This is the recommended pattern for production use: copy the defaults, then tailor them to your domain.

### Adapting the example to your domain

To create your own example:

1. Copy `examples/movie_industry/` to a new directory.
2. Rewrite `schema.py` with your domain's entities and relationships.
3. Update `config.py` with your preferred providers.
4. Replace `data.py` with your own documents.
5. Adjust `prompts.py` to match your domain's language.
6. Update `build_neo4j.py` / `build_falkordb.py` and `search_neo4j.py` / `search_falkordb.py` to call your new modules.

## Next steps

- Read the [Quick Start](quickstart.md) for a minimal version of the same flow.
- Learn about schema design in [Schema](schema.md).
- Explore search modes in [Search](search.md).
