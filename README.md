# Recon-GraphRAG

Domain-agnostic GraphRAG SDK built on Neo4j and `neo4j-graphrag`, following the [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) philosophy.

Like Microsoft GraphRAG, Recon-GraphRAG uses **community detection with multi-level hierarchical communities** to structure knowledge graphs, and provides the same three search paradigms — **Local**, **Global**, and **DRIFT** search — to answer questions at different levels of specificity.

> **Work in Progress** — This project is under active development and is not yet reliable for production use. APIs may change without notice, and features may be incomplete or unstable.

## Requirements

- **Neo4j** (Community or Enterprise edition) with the following plugins:
  - **APOC** — required for entity resolution (`SinglePropertyExactMatchResolver`)
  - **GDS** (Graph Data Science) — required for community detection (Leiden algorithm)

The Docker Compose setup below includes both plugins.

## Quick Start with Docker

If you just want to try out the SDK without setting up your own Neo4j instance, we provide a Docker Compose setup with Neo4j Enterprise (includes APOC and GDS plugins required for GraphRAG):

```bash
# Clone the repo
git clone https://github.com/FadhelHaidar/recon-graphrag.git
cd recon-graphrag

# Copy and customize the environment file (optional)
cp .env.example .env

# Start Neo4j
docker-compose up -d

# Wait for it to be ready (usually takes ~30 seconds)
docker-compose logs -f neo4j
```

Neo4j will be available at:
- **Browser**: http://localhost:7474 (login: neo4j / password)
- **Bolt**: bolt://localhost:7687

To stop:
```bash
docker-compose down
# Or to also remove volumes (data will be lost):
docker-compose down -v
```

## Install

### From GitHub

```bash
# Using uv (recommended)
uv pip install git+https://github.com/FadhelHaidar/recon-graphrag.git

# Or with pip
pip install git+https://github.com/FadhelHaidar/recon-graphrag.git
```

With optional providers:

```bash
# All providers
uv pip install "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/recon-graphrag.git"

# Individual providers
uv pip install "recon-graphrag[openai] @ git+https://github.com/FadhelHaidar/recon-graphrag.git"
uv pip install "recon-graphrag[sentence-transformers] @ git+https://github.com/FadhelHaidar/recon-graphrag.git"
```

### From local clone

```bash
git clone https://github.com/FadhelHaidar/recon-graphrag.git
cd recon-graphrag

# Editable install (recommended for development)
uv pip install -e ".[all]"

# Or with pip
pip install -e ".[all]"
```

### Local clone without install

If you want to hack on the code and test changes immediately without reinstalling:

```bash
# Clone into your project directory
git clone https://github.com/FadhelHaidar/recon-graphrag.git
```

Then run your script from the parent directory:
```bash
# From your project root (above the recon-graphrag folder)
python your_script.py
```

```python
# your_script.py
from recon_graphrag import GraphRAG, GraphBuilderPipeline
```

No `pip install` needed — Python automatically adds the current working directory to its import path.


## Quick Start

```python
from neo4j import GraphDatabase
from recon_graphrag import GraphRAG, GraphBuilderPipeline, CommunityPipeline, Neo4jGraphStore, IndexManager
from recon_graphrag import create_llm, create_embedder
from recon_graphrag.extraction.schema import GraphSchema, NodeType, RelationshipType

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
store = Neo4jGraphStore(driver)

llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")

# Define your domain schema
schema = GraphSchema(
    node_types=[
        NodeType(label="Person", description="An individual", properties=[]),
        NodeType(label="Movie", description="A film", properties=[]),
    ],
    relationship_types=[
        RelationshipType(label="DIRECTED", description="Person directed a movie"),
    ],
    patterns=[("Person", "DIRECTED", "Movie")],
)

# Step 0: Create indexes
IndexManager(store, embedding_dim=1536).create_indexes()

# Steps 1-3: Extract entities, resolve duplicates, embed
pipeline = GraphBuilderPipeline(store, llm, embedder, schema=schema)
await pipeline.build_from_text("Christopher Nolan directed Inception...")

# Steps 4-6: Detect communities, summarize, embed
community = CommunityPipeline(
    store, llm, embedder,
    relationship_types=["DIRECTED"],  # which rel types form community structure
)
await community.build()

# Search
graph_rag = GraphRAG(store, llm, embedder)
result = await graph_rag.search("What are the key findings?", mode="local")
```

## Pipeline Architecture

The indexing pipeline is split into two stages:

| Stage | Pipeline | Steps |
|---|---|---|
| **Graph Building** | `GraphBuilderPipeline` | 1. Extract entities & relationships via LLM |
| | | 2. Entity resolution (merge duplicates) |
| | | 3. Entity embedding |
| **Community Building** | `CommunityPipeline` | 4. Community detection (GDS Leiden) |
| | | 5. Community summarization (LLM) |
| | | 6. Community embedding |

### GraphBuilderPipeline

```python
from recon_graphrag import GraphBuilderPipeline

pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,              # GraphSchema defining your domain
    chunk_size=1000,            # text chunking size (default: 1000)
    chunk_overlap=200,          # overlap between chunks (default: 200)
)

result = await pipeline.build_from_text(text, metadata={"source": "my-doc"})
```

### CommunityPipeline

```python
from recon_graphrag import CommunityPipeline

community = CommunityPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    relationship_types=["DIRECTED", "ACTED_IN"],  # required: which rels form communities
    max_levels=3,               # hierarchy depth (default: 3)
    gamma=1.0,                  # Leiden resolution (default: 1.0)
    theta=0.01,                 # Leiden tolerance (default: 0.01)
    summary_prompt=None,        # custom summary prompt (uses default if None)
)

result = await community.build()          # all levels
result = await community.build(level=0)   # specific level only
```

> **Important:** `relationship_types` determines which relationships are projected into the community detection graph. Choose relationship types that create meaningful connections between entities. For example, `["DIRECTED", "ACTED_IN", "COMPOSED_MUSIC"]` would group people and movies that are connected through creative roles.

## Defining a Schema

The schema tells the LLM what entities and relationships to extract from your text. You can use the `GraphSchema` class directly or the `build_schema()` helper for a more compact definition.

### Using GraphSchema directly

```python
from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType, RelationshipType

schema = GraphSchema(
    node_types=[
        NodeType(
            label="Movie",
            description="A film or motion picture",
            properties=[
                PropertyType(name="title", type="STRING"),
                PropertyType(name="year", type="STRING"),
            ],
        ),
        NodeType(label="Person", description="An individual in the film industry", properties=[]),
    ],
    relationship_types=[
        RelationshipType(label="DIRECTED", description="Person directed a movie"),
        RelationshipType(label="ACTED_IN", description="Person acted in a movie"),
    ],
    patterns=[
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
    ],
)
```

### Using build_schema()

```python
from recon_graphrag.extraction.schema import build_schema

schema = build_schema(
    node_types=[
        {"label": "Movie", "description": "A film or motion picture",
         "properties": [{"name": "title", "type": "STRING"}]},
        {"label": "Person", "description": "An individual in the film industry"},
    ],
    relationship_types=[
        {"label": "DIRECTED", "description": "Person directed a movie"},
        {"label": "ACTED_IN", "description": "Person acted in a movie"},
    ],
    patterns=[
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
    ],
)
```

## IndexManager

Creates and manages the vector and fulltext indexes needed for retrieval.

```python
from recon_graphrag import IndexManager, Neo4jGraphStore

manager = IndexManager(store, embedding_dim=1536)

# Create all required indexes (run once after setting up the store)
manager.create_indexes()
# Creates:
#   - chunk-embeddings      (vector index on Chunk.embedding)
#   - entity-embeddings     (vector index on __Entity__.embedding)
#   - community-embeddings  (vector index on Community.embedding)
#   - entity-names           (fulltext index on __Entity__.name)

# Verify indexes and print node/relationship counts (debug helper)
manager.verify()
```

## Providers

### LLM Providers

| Provider | Key | Notes |
|---|---|---|
| OpenAI | `"openai"` | Requires `openai` optional dep |
| Azure OpenAI | `"azure_openai"` | Requires `openai` optional dep |
| Ollama | `"ollama"` | Requires `ollama` optional dep |
| OpenRouter | `"openrouter"` | Uses OpenAI-compatible API, requires `openai` dep |

```python
# OpenAI
llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")

# OpenRouter
llm = create_llm("openrouter", model_name="anthropic/claude-sonnet", api_key="sk-or-...")

# Custom OpenAI-compatible endpoint
llm = create_llm("openai", model_name="custom-model",
                  base_url="http://localhost:8000/v1", api_key="dummy")

# Ollama
llm = create_llm("ollama", model_name="llama3")
```

### Embedder Providers

| Provider | Key | Default Dim | Notes |
|---|---|---|---|
| OpenAI | `"openai"` | 1536 | Requires `openai` optional dep |
| Azure OpenAI | `"azure_openai"` | 1536 | Requires `openai` optional dep |
| Ollama | `"ollama"` | varies | Requires `ollama` optional dep |
| OpenRouter | `"openrouter"` | varies | Uses OpenAI-compatible API |
| Sentence-Transformers | `"sentence-transformer"` | auto-detected | Requires `sentence-transformers` dep |

```python
# OpenAI
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")

# OpenRouter
embedder = create_embedder("openrouter", model="openai/text-embedding-3-small", api_key="sk-or-...")

# Custom OpenAI-compatible endpoint (e.g. local Ollama with OpenAI compat)
embedder = create_embedder("openai", model="nomic-embed-text",
                            base_url="http://localhost:11434/v1", api_key="ollama")

# Sentence-Transformers (local, no API key needed)
embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
```

### Passing model_params

For providers that support extra parameters on each embedding call (e.g. OpenAI's `dimensions` or `encoding_format`), use `model_params`:

```python
embedder = create_embedder(
    "openai", model="text-embedding-3-small", api_key="sk-...",
    model_params={"dimensions": 512},  # passed to every embed_query() call
)
```

> **Note:** When using non-OpenAI embedders, the `IndexManager` auto-detects the embedding dimension for sentence-transformers. For other providers, pass `embedding_dim` explicitly.

## Search Modes

| Mode | Strategy | Best For |
|---|---|---|
| **local** | Entity subgraph traversal | Specific questions about entities |
| **global** | Community summaries map-reduce | Broad overviews and landscapes |
| **drift** | Entity + community hybrid | Questions needing detail + context |

### Search Parameters

```python
graph_rag = GraphRAG(store, llm, embedder)

# Local search — top_k entities to retrieve
result = await graph_rag.search("query", mode="local", top_k=10)

# Global search — top_k communities + hierarchy level
result = await graph_rag.search("query", mode="global", top_k=5, level=0)

# DRIFT search — top_k entities + community_top_k communities to expand
result = await graph_rag.search("query", mode="drift", top_k=10, community_top_k=3)
```

### Customizing Prompts

All retrievers expose prompt attributes that you can override for domain-specific behavior:

```python
graph_rag = GraphRAG(store, llm, embedder)

# Local search — override the answer prompt
graph_rag.local.answer_prompt = "You are a film analyst. Answer based on:\n{context}\n\nQuestion: {query}"

# Global search — override map and reduce prompts
graph_rag.global_.map_prompt = "Based on this report segment about films, answer: {query}\n\nSegment: {context}"
graph_rag.global_.reduce_prompt = "Synthesize these film perspectives:\n{partial_answers}\n\nQuestion: {query}"

# DRIFT search — override the answer prompt
graph_rag.drift.answer_prompt = "Given specific findings and broader film context, answer: {query}\n\n{context}"

# Local/DRIFT — override the Cypher retrieval query (advanced)
graph_rag.local.retrieval_query = "OPTIONAL MATCH (node)-[r]-(neighbor) RETURN node, r, neighbor"
```

> **Tip:** You can also pass a custom `summary_prompt` to `CommunityPipeline` to control how community summaries are generated during indexing.

## Example

A complete movie industry example is in [examples/movie_industry/](examples/movie_industry/):

| File | Purpose |
|---|---|
| [schema.py](examples/movie_industry/schema.py) | Domain schema with 10 entity types and 14 relationship types |
| [config.py](examples/movie_industry/config.py) | Multi-provider setup (OpenRouter, Azure, local) |
| [prompts.py](examples/movie_industry/prompts.py) | Domain-customized prompts for all retrievers |
| [build_graph.py](examples/movie_industry/build_graph.py) | Ingestion + community pipeline runner |
| [query.py](examples/movie_industry/query.py) | Test suite across all three search modes |

```bash
cd examples/movie_industry
python build_graph.py    # Build the graph
python query.py          # Run test queries
```

## License

This project is licensed under the [MIT License](LICENSE).
