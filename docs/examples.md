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
| [config.py](../examples/movie_industry/config.py) | Multi-provider setup (Azure OpenAI, OpenRouter, local) |
| [prompts.py](../examples/movie_industry/prompts.py) | Domain-customized prompts for all retrievers |
| [data.py](../examples/movie_industry/data.py) | Sample movie text data |
| [build.py](../examples/movie_industry/build.py) | Ingestion + community pipeline runner |
| [search.py](../examples/movie_industry/search.py) | Test queries across all three search modes |

### Run the example

1. Start Neo4j with APOC and GDS:

   ```bash
   docker-compose up -d
   ```

2. Copy and fill in your environment variables:

   ```bash
   cp .env.example .env
   ```

   You will need Azure OpenAI credentials, or you can adapt `config.py` to use OpenRouter or Ollama.

3. Build the graph and communities:

   ```bash
   cd examples/movie_industry
   python build.py
   ```

4. Run test queries:

   ```bash
   python search.py
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
6. Update `build.py` and `search.py` to call your new modules.

## Next steps

- Read the [Quick Start](quickstart.md) for a minimal version of the same flow.
- Learn about schema design in [Schema](schema.md).
- Explore search modes in [Search](search.md).
