# Search

Recon-GraphRAG provides three search modes, matching the Microsoft GraphRAG philosophy: **Local**, **Global**, and **DRIFT**.

For a deep dive into the internals, see [Search Internals](search-internals.md).

## Search modes overview

| Mode | Strategy | Best for |
| --------- | --------- | --------- |
| **Local** | Entity subgraph traversal | Specific questions about named entities |
| **Global** | Community summary map-reduce | Broad overviews and thematic landscapes |
| **DRIFT** | Entity + community hybrid | Questions needing detail plus surrounding context |

All modes are accessed through `GraphRAG.search()`:

```python
from recon_graphrag import GraphRAG

graph_rag = GraphRAG(store, llm, embedder)
result = await graph_rag.search("Your question", mode="local")
```

## Local search

Local search answers specific questions by retrieving relevant entities, their neighbors, and related text chunks.

```python
result = await graph_rag.search(
    "Who directed Inception?",
    mode="local",
    top_k=10,
)
```

| Parameter | Description |
| --------- | --------- |
| `top_k` | Number of top entities to retrieve. |

## Global search

Global search answers broad questions by mapping community summaries and reducing them into a single answer.

```python
result = await graph_rag.search(
    "What are the main themes in this dataset?",
    mode="global",
    top_k=5,
    level=0,
)
```

| Parameter | Description |
| --------- | --------- |
| `top_k` | Number of communities to include. |
| `level` / `community_level` | Which community level to search. |

## DRIFT search

DRIFT search combines local entity retrieval with community context for questions that need both detail and big-picture framing.

```python
result = await graph_rag.search(
    "Explain the relationship between Christopher Nolan and his frequent collaborators.",
    mode="drift",
    top_k=10,
    community_top_k=3,
)
```

| Parameter | Description |
| --------- | --------- |
| `top_k` | Number of entities to retrieve. |
| `community_top_k` | Number of communities to expand into. |
| `community_level` | Which community level to use. |

## Community levels

Recon-GraphRAG stores communities with `level=0` as the **finest / most local** level. Higher numbers are broader parent communities. This is the opposite of some Microsoft GraphRAG descriptions, where level 0 is often the coarsest root level.

The search API supports semantic selectors:

```python
community_level="all"       # No level filter
community_level="finest"    # level 0
community_level="coarsest"  # Highest available level
community_level=0             # Exact stored level
community_level=1             # Exact stored level
```

For full background, see [Community Level Numbering](community-level-numbering.md).

## Customizing prompts

Each retriever exposes prompt attributes that you can override for domain-specific behavior.

### Local search prompt

```python
graph_rag.local.answer_prompt = (
    "You are a film analyst. Answer based on:\n{context}\n\nQuestion: {query}"
)
```

### Global search prompt

```python
graph_rag.global_.map_prompt = (
    "Based on this report segment about films, answer: {query}\n\nSegment: {context}"
)
graph_rag.global_.reduce_prompt = (
    "Synthesize these film perspectives:\n{partial_answers}\n\nQuestion: {query}"
)
```

### DRIFT search prompt

```python
graph_rag.drift.answer_prompt = (
    "Given specific findings and broader film context, answer: {query}\n\n{context}"
)
```

### Custom Cypher retrieval query

Advanced users can override the Cypher query used to fetch context:

```python
graph_rag.local.retrieval_query = (
    "OPTIONAL MATCH (node)-[r]-(neighbor) RETURN node, r, neighbor"
)
```

## Customizing community summaries

You can also pass a custom `summary_prompt` to `CommunityPipeline` to control how community summaries are generated during indexing:

```python
community = CommunityPipeline(
    store, llm, embedder,
    relationship_types=["DIRECTED"],
    summary_prompt="Summarize these film industry connections:\n{context}",
)
```

## When to use each mode

- **Local** when the question names a specific entity or asks for a fact about something.
- **Global** when the question is broad, thematic, or asks for an overview.
- **DRIFT** when the question needs specific evidence but also broader context to interpret it.

## Next steps

- Try the search examples in [Quick Start](quickstart.md).
- See a full domain example in [Movie Industry Example](movie-industry-example.md).
- Understand community level numbering in [Community Level Numbering](community-level-numbering.md).
