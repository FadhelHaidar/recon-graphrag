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
| [extract.py](../examples/extract.py) | Extract once into a database-neutral JSON graph artifact |
| [ingest.py](../examples/ingest.py) | Ingest the shared graph artifact into one or all graph backends |
| [communities.py](../examples/communities.py) | Build communities for one or all graph backends |
| [search.py](../examples/search.py) | Run the shared query suite on Neo4j or Memgraph |
| [compare_backends.py](../examples/compare_backends.py) | Compare graph stats and retrieval outputs across Neo4j and Memgraph |

### Provider selection

All build and search scripts accept `--llm-provider` and `--embedder-provider` flags:

```bash
python extract.py --llm-provider openrouter
python ingest.py --backend all --embedder-provider openai
python search.py --backend neo4j --llm-provider openai --embedder-provider sentence-transformer
```

Supported LLM providers: `openrouter`, `azure_openai`, `openai`.  
Supported embedder providers: `openrouter`, `azure_openai`, `openai`, `sentence-transformer`.

You can also set the `LLM_PROVIDER` and `EMBEDDER_PROVIDER` environment variables instead of passing flags.

For hybrid entity resolution during ingest, pass
`--entity-resolution-strategy hybrid`. The LLM and embedder review ambiguous
duplicate candidates. Add `--allow-ai-auto-merge` only when you want
LLM-approved review candidates to be merged into the stored graph:

```bash
python ingest.py --backend neo4j --entity-resolution-strategy hybrid --allow-ai-auto-merge
```

### Search modes per test case

`search.py` runs the query modes declared by each test case in the shared suite (e.g. `["local"]`, `["global", "drift"]`, `["local", "global", "drift"]`). The `--modes` flag filters which cases to run, not which modes are executed within a case.

### Run the example with Neo4j

1. Start Neo4j with APOC and GDS:

   ```bash
   docker compose up -d
   ```

2. Copy and fill in your environment variables:

   ```bash
   cp .env.example .env
   ```

3. Extract and ingest the shared graph artifact:

   ```bash
   cd examples
   python extract.py --llm-provider openrouter
   python ingest.py --backend neo4j --embedder-provider openrouter
   ```

4. Build communities:

   ```bash
   python communities.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter
   ```

5. Run test queries:

   ```bash
   python search.py --backend neo4j --llm-provider openrouter --embedder-provider openrouter
   ```

### Run the example with Memgraph

1. Start Memgraph with MAGE:

   ```bash
   docker compose up -d memgraph
   ```

2. Ensure `MEMGRAPH_URL` is set in your environment (default is `bolt://localhost:7689`).

3. Extract once if you have not already, then ingest the shared graph artifact:

   ```bash
   cd examples
   python extract.py --llm-provider openrouter
   python ingest.py --backend memgraph --embedder-provider openrouter
   ```

4. Build communities:

   ```bash
   python communities.py --backend memgraph --community-gamma 3.0 --llm-provider openrouter --embedder-provider openrouter
   ```

5. Run test queries:

   ```bash
   python search.py --backend memgraph --llm-provider openrouter --embedder-provider openrouter
   ```

### Compare Neo4j and Memgraph

To compare outputs fairly, ingest all configured graph backends from the same artifact before building communities:

```bash
python extract.py --llm-provider openrouter
python ingest.py --backend all --embedder-provider openrouter
python communities.py --backend all --llm-provider openrouter --embedder-provider openrouter
python compare_backends.py --limit 5
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
