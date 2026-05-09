# Recon-GraphRAG

Domain-agnostic GraphRAG SDK built on Neo4j and `neo4j-graphrag`, following the [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) philosophy.

Like Microsoft GraphRAG, Recon-GraphRAG uses **community detection with multi-level hierarchical communities** to structure knowledge graphs, and provides the same three search paradigms — **Local**, **Global**, and **DRIFT** search — to answer questions at different levels of specificity.

> **Work in Progress** — This project is under active development and is not yet reliable for production use. APIs may change without notice, and features may be incomplete or unstable.

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
from recon_graphrag import GraphRAG, GraphBuilderPipeline, Neo4jGraphStore
from recon_graphrag import create_llm, create_embedder

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
store = Neo4jGraphStore(driver)

llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")

# Ingest
pipeline = GraphBuilderPipeline(store, llm, embedder, schema=my_schema)
await pipeline.build_from_text("Your text here...")

# Search
graph_rag = GraphRAG(store, llm, embedder)
result = await graph_rag.search("What are the key findings?", mode="local")
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

> **Note:** When using non-OpenAI embedders, the `IndexManager` auto-detects the embedding dimension for sentence-transformers. For other providers, pass `embedding_dim` explicitly.

## Search Modes

| Mode | Strategy | Best For |
|---|---|---|
| **local** | Entity subgraph traversal | Specific questions about entities |
| **global** | Community summaries map-reduce | Broad overviews and landscapes |
| **drift** | Entity + community hybrid | Questions needing detail + context |

## License

This project is licensed under the [MIT License](LICENSE).
