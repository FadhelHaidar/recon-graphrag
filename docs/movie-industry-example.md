# Movie Industry Example

The `examples/` directory contains a complete, runnable movie-industry project that demonstrates how to use Recon-GraphRAG end-to-end.

## Movie industry example

`examples/` is a full GraphRAG application for the film domain. It shows how to:

- Define a domain schema with many entity and relationship types.
- Configure multiple LLM/embedder providers.
- Customize retriever prompts for a domain.
- Build a graph from paginated text.
- Run local, global, and DRIFT searches.

### Files

| File | Purpose |
| --------- | --------- |
| [schema.py](../examples/schema.py) | Domain schema with 10 entity types and 14 relationship types |
| [config.py](../examples/config.py) | Multi-provider setup (Azure OpenAI, OpenRouter, local) and store getters for Neo4j and Memgraph |
| [prompts.py](../examples/prompts.py) | Domain-customized prompts for all retrievers |
| [data.py](../examples/data.py) | Sample movie text data |
| [extract_movie_graph.py](../examples/extract_movie_graph.py) | Extract once into a database-neutral JSON graph artifact |
| [ingest_movie_graph.py](../examples/ingest_movie_graph.py) | Ingest the shared graph artifact into Neo4j, Memgraph, or both |
| [build_communities.py](../examples/build_communities.py) | Build communities for Neo4j, Memgraph, or both |
| [search_movie_graph.py](../examples/search_movie_graph.py) | Run the shared query suite on Neo4j or Memgraph |
| [compare_memgraph_neo4j.py](../examples/compare_memgraph_neo4j.py) | Compare graph stats and retrieval outputs across both backends |

### Provider selection

All build and search scripts accept `--llm-provider` and `--embedder-provider` flags:

```bash
python extract_movie_graph.py --llm-provider openrouter
python ingest_movie_graph.py --backend both --embedder-provider openai
python search_movie_graph.py --backend neo4j --llm-provider openai --embedder-provider sentence-transformer
```

Supported LLM providers: `openrouter`, `azure_openai`, `openai`.  
Supported embedder providers: `openrouter`, `azure_openai`, `openai`, `sentence-transformer`.

You can also set the `LLM_PROVIDER` and `EMBEDDER_PROVIDER` environment variables instead of passing flags.

### Search modes per test case

`search_movie_graph.py` no longer runs every query through all three search modes. Each test case in the shared suite declares its own `modes` list (e.g. `["local"]`, `["global", "drift"]`, `["local", "global", "drift"]`). Omitting `modes` defaults to running all three.

### Run the example with Neo4j

1. Start Neo4j with APOC and GDS:

   ```bash
   docker-compose up -d
   ```

2. Copy and fill in your environment variables:

   ```bash
   cp .env.example .env
   ```

3. Extract and ingest the shared graph artifact:

   ```bash
   cd examples
   python extract_movie_graph.py --llm-provider openrouter
   python ingest_movie_graph.py --backend neo4j --embedder-provider openrouter
   ```

4. Build communities:

   ```bash
   python build_communities.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter
   ```

5. Run test queries:

   ```bash
   python search_movie_graph.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter
   ```

### Run the example with Memgraph

1. Start Memgraph with MAGE:

   ```bash
   docker-compose up -d memgraph
   ```

2. Ensure `MEMGRAPH_URL` is set in your environment (default is `bolt://localhost:7689`).

3. Extract once if you have not already, then ingest the shared graph artifact:

   ```bash
   cd examples
   python extract_movie_graph.py --llm-provider openrouter
   python ingest_movie_graph.py --backend memgraph --embedder-provider openrouter
   ```

4. Build communities:

   ```bash
   python build_communities.py --backend memgraph --community-gamma 3.0 --llm-provider openrouter --embedder-provider openrouter
   ```

5. Run test queries:

   ```bash
   python search_movie_graph.py --backend memgraph --llm-provider openrouter --embedder-provider openrouter
   ```

### Compare Neo4j and Memgraph

To compare outputs fairly, ingest both databases from the same artifact before building communities:

```bash
python extract_movie_graph.py --llm-provider openrouter
python ingest_movie_graph.py --backend both --embedder-provider openrouter
python build_communities.py --backend both --llm-provider openrouter --embedder-provider openrouter
python compare_memgraph_neo4j.py --limit 5
```

### Schema highlights

The movie schema includes entity types such as `Movie`, `Person`, `Company`, `Genre`, `Award`, and `Location`, connected by relationships such as `DIRECTED`, `ACTED_IN`, `PRODUCED_BY`, `WON`, `NOMINATED_FOR`, and `LOCATED_IN`.

### Prompt customization

`prompts.py` overrides the default answer, map, reduce, and retrieval prompts so the retrievers answer in the style of a film analyst. This is the recommended pattern for production use: copy the defaults, then tailor them to your domain.

### Adapting the example to your domain

To create your own example:

1. Copy `examples/` to a new directory.
2. Rewrite `schema.py` with your domain's entities and relationships.
3. Update `config.py` with your preferred providers.
4. Replace `data.py` with your own documents.
5. Adjust `prompts.py` to match your domain's language.
6. Update the extract, ingest, community, and search scripts to call your new modules.

## Next steps

- Read the [Quick Start](quickstart.md) for a minimal version of the same flow.
- Learn about schema design in [Schema](schema.md).
- Explore search modes in [Search](search.md).
