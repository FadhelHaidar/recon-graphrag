# Recon-GraphRAG

Domain-agnostic GraphRAG SDK built on Neo4j and `neo4j-graphrag`, following the [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) philosophy.

Like Microsoft GraphRAG, Recon-GraphRAG uses **community detection with multi-level hierarchical communities** to structure knowledge graphs, and provides the same three search paradigms — **Local**, **Global**, and **DRIFT** search — to answer questions at different levels of specificity.

> **Work in Progress** — This project is under active development and is not yet reliable for production use. APIs may change without notice, and features may be incomplete or unstable.

## Install

```bash
pip install recon-graphrag
```

Or clone into your project:

```bash
git clone https://github.com/FadhelHaidar/recon-graphrag.git recon_graphrag
```

## Quick Start

```python
from neo4j import GraphDatabase
from recon_graphrag import GraphRAG, GraphBuilderPipeline, Neo4jGraphStore
from recon_graphrag.providers import create_llm, create_embedder

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

## Search Modes

| Mode | Strategy | Best For |
| ------ | ---------- | ---------- |
| **local** | Entity subgraph traversal | Specific questions about entities |
| **global** | Community summaries map-reduce | Broad overviews and landscapes |
| **drift** | Entity + community hybrid | Questions needing detail + context |

## License

This project is licensed under the [MIT License](LICENSE).
