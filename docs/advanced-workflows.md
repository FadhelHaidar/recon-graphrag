# Advanced Workflows

Recon-GraphRAG's high-level examples ([`GraphBuilderPipeline`](pipelines.md), [`GraphRAG.search()`](search.md)) cover the most common path, but the library is composed of smaller, interchangeable building blocks. This guide explains those primitives so you can assemble custom workflows, implement new backends, or inspect intermediate artifacts without running a full pipeline.

## When to use the building blocks

Reach for these primitives when you want to:

- Extract entities once and save the structured output to JSON/Parquet before database ingestion.
- Call `LocalSearchRetriever`, `GlobalSearchRetriever`, or `DriftSearchRetriever` directly instead of through `GraphRAG`.
- Implement a custom `GraphStore` backend (Postgres, in-memory, etc.).
- Add side effects (logging, caching, validation) by wrapping `GraphWriter`.
- Swap in your own LLM or embedder that does not fit the factory functions.

## Core data types

All database-neutral types live in [`recon_graphrag.extraction.types`](../recon_graphrag/extraction/types.py).

| Type | Purpose |
| ---- | ------- |
| `ExtractedNode` | One entity node produced by the LLM extractor. |
| `ExtractedRelationship` | One relationship produced by the LLM extractor. |
| `GraphExtraction` | A batch of extracted nodes and relationships for one text chunk. |
| `DocumentRecord` | The root document metadata. |
| `ChunkRecord` | A text chunk linked back to a document. |
| `EntityRecord` | A normalized entity ready for persistence. |
| `RelationshipRecord` | A normalized relationship ready for persistence. |
| `EvidenceLink` | Links a chunk to the entities it spawned. |
| `GraphDocument` | The assembled bundle of document, chunks, entities, relationships, and evidence links. |

`GraphDocument` is the key interchange format: it is what the pipeline writes to the graph store, and it is the artifact you can serialize before ingestion.

### Input / output example

Given this input text:

```text
Christopher Nolan directed Inception. Hans Zimmer composed the score.
```

A single-chunk `GraphExtraction` might look like:

```python
GraphExtraction(
    nodes=[
        ExtractedNode(
            id="ent-1",
            label="Person",
            properties={"name": "Christopher Nolan", "description": "Film director"},
        ),
        ExtractedNode(
            id="ent-2",
            label="Movie",
            properties={"name": "Inception", "description": "Science fiction film"},
        ),
        ExtractedNode(
            id="ent-3",
            label="Person",
            properties={"name": "Hans Zimmer", "description": "Composer"},
        ),
    ],
    relationships=[
        ExtractedRelationship(
            source_id="ent-1",
            target_id="ent-2",
            type="DIRECTED",
            properties={"description": "Christopher Nolan directed Inception"},
        ),
        ExtractedRelationship(
            source_id="ent-3",
            target_id="ent-2",
            type="COMPOSED_MUSIC",
            properties={"description": "Hans Zimmer composed music for Inception"},
        ),
    ],
)
```

After validation and assembly, the corresponding `GraphDocument` contains normalized records:

```python
GraphDocument(
    document=DocumentRecord(
        id="doc:inception-example",
        text_hash="a1b2c3...",
        metadata={"source": "advanced-example"},
        graph_name="entity-graph",
    ),
    chunks=[
        ChunkRecord(
            id="doc:inception-example:chunk:0",
            document_id="doc:inception-example",
            text="Christopher Nolan directed Inception. Hans Zimmer composed the score.",
            index=0,
            graph_name="entity-graph",
        )
    ],
    entities=[
        EntityRecord(
            id="ent-1",
            type="Person",
            properties={"name": "Christopher Nolan", "description": "Film director"},
            graph_name="entity-graph",
        ),
        EntityRecord(
            id="ent-2",
            type="Movie",
            properties={"name": "Inception", "description": "Science fiction film"},
            graph_name="entity-graph",
        ),
        EntityRecord(
            id="ent-3",
            type="Person",
            properties={"name": "Hans Zimmer", "description": "Composer"},
            graph_name="entity-graph",
        ),
    ],
    relationships=[
        RelationshipRecord(
            id="ent-1:DIRECTED:ent-2",
            source_id="ent-1",
            target_id="ent-2",
            type="DIRECTED",
            properties={
                "description": "Christopher Nolan directed Inception",
                "weight": 1.0,
                "source_chunk_ids": ["doc:inception-example:chunk:0"],
            },
            graph_name="entity-graph",
        ),
        RelationshipRecord(
            id="ent-3:COMPOSED_MUSIC:ent-2",
            source_id="ent-3",
            target_id="ent-2",
            type="COMPOSED_MUSIC",
            properties={
                "description": "Hans Zimmer composed music for Inception",
                "weight": 1.0,
                "source_chunk_ids": ["doc:inception-example:chunk:0"],
            },
            graph_name="entity-graph",
        ),
    ],
    evidence_links=[
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="ent-1", graph_name="entity-graph"),
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="ent-2", graph_name="entity-graph"),
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="ent-3", graph_name="entity-graph"),
    ],
)
```

## Extraction primitives

The full extraction path is split into three objects so you can stop at any point.

### 1. `LLMGraphExtractor`

[`recon_graphrag.extraction.extractor.LLMGraphExtractor`](../recon_graphrag/extraction/extractor.py) calls the LLM with a schema-specific prompt and parses the response into a `GraphExtraction`.

```python
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.schema import GraphSchema

extractor = LLMGraphExtractor(llm)
extraction = await extractor.extract(
    text="Christopher Nolan directed Inception.",
    schema=schema,
)

print(len(extraction.nodes), len(extraction.relationships))
```

### 2. `SchemaValidator`

[`recon_graphrag.extraction.validator.SchemaValidator`](../recon_graphrag/extraction/validator.py) filters nodes and relationships against the schema, normalizes identity properties (`name`), and sets defaults (`description`, `weight`).

```python
from recon_graphrag.extraction.validator import SchemaValidator

validator = SchemaValidator()
validated = validator.validate(extraction, schema)
```

### 3. `GraphDocumentAssembler`

[`recon_graphrag.extraction.assembler.GraphDocumentAssembler`](../recon_graphrag/extraction/assembler.py) merges per-chunk extractions into a single `GraphDocument`, deduplicating entities and aggregating relationship weights.

```python
from recon_graphrag.extraction.assembler import GraphDocumentAssembler

assembler = GraphDocumentAssembler()
graph_doc = assembler.assemble(
    document_id="doc:test",
    text_hash="sha256...",
    chunks=chunks,
    chunk_extractions={chunk.id: validated for chunk, validated in zip(chunks, validations)},
    metadata={"source": "advanced-example"},
    graph_name="entity-graph",
)
```

### Chunking primitives

[`recon_graphrag.extraction.chunking`](../recon_graphrag/extraction/chunking.py) provides two chunk builders:

- `TextChunker` — character-level sliding windows.
- `PageWindowBuilder` — page-level sliding windows that join multiple pages into one chunk.

```python
from recon_graphrag.extraction.chunking import TextChunker, PageWindowBuilder

chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_text(text, document_id="doc:test", metadata={"source": "text"})

window_builder = PageWindowBuilder(window_size=2, window_overlap=1)
page_chunks = window_builder.build_windows(pages, document_id="doc:test")
```

## GraphStore / GraphWriter

[`recon_graphrag.graphdb.base`](../recon_graphrag/graphdb/base.py) defines two protocols:

- `GraphStore` — the full database abstraction used by pipelines and retrievers.
- `GraphWriter` — the minimal write-only contract (`write_graph_document`).

`GraphBuilderPipeline` accepts a `graph_writer` parameter that only needs to satisfy `GraphWriter`. This is the cleanest extension point for saving extractions before ingestion.

### Save extractions to JSON before writing

```python
import json
from pathlib import Path
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphWriter


class JsonSavingWriter:
    """Wraps a real writer and dumps each GraphDocument to JSON first."""

    def __init__(self, inner: GraphWriter, out_dir: Path | str):
        self.inner = inner
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        path = self.out_dir / f"{graph_document.document.id}.json"
        path.write_text(
            json.dumps(self._to_dict(graph_document), indent=2, default=str),
            encoding="utf-8",
        )
        return self.inner.write_graph_document(graph_document)

    @staticmethod
    def _to_dict(doc: GraphDocument) -> dict:
        return {
            "document": {
                "id": doc.document.id,
                "text_hash": doc.document.text_hash,
                "metadata": doc.document.metadata,
                "graph_name": doc.document.graph_name,
            },
            "chunks": [
                {"id": c.id, "document_id": c.document_id, "text": c.text, "index": c.index, "metadata": c.metadata}
                for c in doc.chunks
            ],
            "entities": [
                {"id": e.id, "type": e.type, "properties": e.properties}
                for e in doc.entities
            ],
            "relationships": [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type,
                    "properties": r.properties,
                }
                for r in doc.relationships
            ],
            "evidence_links": [
                {"chunk_id": l.chunk_id, "entity_id": l.entity_id}
                for l in doc.evidence_links
            ],
        }
```

Usage:

```python
from recon_graphrag import GraphBuilderPipeline

pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    graph_writer=JsonSavingWriter(store, out_dir="extracted_graphs"),
)
```

### Saved JSON shape

Using the writer above, the file `extracted_graphs/doc:inception-example.json` would contain:

```json
{
  "document": {
    "id": "doc:inception-example",
    "text_hash": "a1b2c3...",
    "metadata": {"source": "advanced-example"},
    "graph_name": "entity-graph"
  },
  "chunks": [
    {
      "id": "doc:inception-example:chunk:0",
      "document_id": "doc:inception-example",
      "text": "Christopher Nolan directed Inception. Hans Zimmer composed the score.",
      "index": 0,
      "metadata": {"char_start": 0, "char_end": 73}
    }
  ],
  "entities": [
    {"id": "ent-1", "type": "Person", "properties": {"name": "Christopher Nolan", "description": "Film director"}},
    {"id": "ent-2", "type": "Movie", "properties": {"name": "Inception", "description": "Science fiction film"}},
    {"id": "ent-3", "type": "Person", "properties": {"name": "Hans Zimmer", "description": "Composer"}}
  ],
  "relationships": [
    {
      "id": "ent-1:DIRECTED:ent-2",
      "source_id": "ent-1",
      "target_id": "ent-2",
      "type": "DIRECTED",
      "properties": {
        "description": "Christopher Nolan directed Inception",
        "weight": 1.0,
        "source_chunk_ids": ["doc:inception-example:chunk:0"]
      }
    },
    {
      "id": "ent-3:COMPOSED_MUSIC:ent-2",
      "source_id": "ent-3",
      "target_id": "ent-2",
      "type": "COMPOSED_MUSIC",
      "properties": {
        "description": "Hans Zimmer composed music for Inception",
        "weight": 1.0,
        "source_chunk_ids": ["doc:inception-example:chunk:0"]
      }
    }
  ],
  "evidence_links": [
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "ent-1"},
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "ent-2"},
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "ent-3"}
  ]
}
```

### Implement a custom GraphStore

Implement every method in `GraphStore` to add a new backend. The unit tests for `Neo4jGraphStore` are the best reference for the expected behavior of each method.

```python
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.extraction.types import GraphDocument

class InMemoryGraphStore(GraphStore):
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        ...

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        ...

    # ... implement remaining GraphStore methods
```

## Retrieval primitives

Instead of `GraphRAG.search()`, you can instantiate retrievers directly. This is useful when you need fine-grained control over index names, prompts, or result formatting.

All retrievers return a [`SearchResult`](../recon_graphrag/models/types.py) with `query`, `mode`, `answer`, and `context`.

```python
SearchResult(
    query="Who directed Inception?",
    mode="local",
    answer="Christopher Nolan directed Inception.",
    context="Finding: Christopher Nolan (Person)\nConnections:\n  Person: Christopher Nolan -[DIRECTED]-> Movie: Inception\nEvidence:\n  Christopher Nolan directed Inception.",
)
```

### `LocalSearchRetriever`

Entity-centric search: vector + keyword retrieval on entities, subgraph traversal, then an LLM answer.

```python
from recon_graphrag.retrieval.local import LocalSearchRetriever

retriever = LocalSearchRetriever(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    answer_prompt=LOCAL_ANSWER_PROMPT,
)
result = await retriever.search("Who directed Inception?", top_k=10)
```

### `GlobalSearchRetriever`

Community-summaries-based map-reduce search.

```python
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever

retriever = GlobalSearchRetriever(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    map_prompt=GLOBAL_MAP_PROMPT,
    reduce_prompt=GLOBAL_REDUCE_PROMPT,
    graph_name="entity-graph",
)
result = await retriever.search("What are the main themes?", top_k=5, community_level="coarsest")
```

### `DriftSearchRetriever`

Hybrid of local entity context and community context.

```python
from recon_graphrag.retrieval.drift import DriftSearchRetriever

retriever = DriftSearchRetriever(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    answer_prompt=DRIFT_ANSWER_PROMPT,
    graph_name="entity-graph",
    community_level="finest",
)
result = await retriever.search(
    "How does Hans Zimmer connect Inception to Dune?",
    top_k=10,
    community_top_k=3,
)
```

### `HybridEntityRetriever`

The underlying entity retriever used by local and DRIFT search. It performs vector and keyword search, merges scores, and fetches formatted entity context.

```python
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever

retriever = HybridEntityRetriever(
    graph_store=store,
    embedder=embedder,
    vector_index_name="entity-embeddings",
    fulltext_index_name="entity-names",
    context_mode="local",  # or "drift"
)
result = await retriever.search("Who directed Inception?", top_k=10)
```

## Community primitives

[`recon_graphrag.communities.pipeline.CommunityPipeline`](../recon_graphrag/communities/pipeline.py) is a convenience wrapper around three steps:

1. `graph_store.detect_communities()` — run the Leiden algorithm.
2. `CommunitySummarizer` — generate a natural-language summary per community.
3. `CommunityEmbedder` — embed those summaries.

You can call these steps individually if you need more control:

```python
from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.communities.embeddings import CommunityEmbedder

summarizer = CommunitySummarizer(store, llm, graph_name="entity-graph")
summaries = await summarizer.summarize_all(level=0)

embedder = CommunityEmbedder(store, embedder, graph_name="entity-graph")
await embedder.embed_communities(level=0)
```

## LLM / embedder base classes

The library depends on protocols, not concrete providers.

### `BaseLLM`

[`recon_graphrag.llm.base.BaseLLM`](../recon_graphrag/llm/base.py) requires only two methods:

```python
from recon_graphrag.llm import LLMResponse
from recon_graphrag.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="...")

    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="...")
```

### `BaseEmbedder`

[`recon_graphrag.embeddings.base.BaseEmbedder`](../recon_graphrag/embeddings/base.py) requires:

```python
from recon_graphrag.embeddings.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def async_embed_query(self, text: str, **kwargs) -> list[float]:
        return self.embed_query(text)
```

## Complete custom flow example

Putting it together: extract, save to JSON, then ingest — all without `GraphBuilderPipeline`.

```python
import asyncio
import json
from pathlib import Path

from recon_graphrag.extraction.chunking import TextChunker
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.validator import SchemaValidator
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.schema import GraphSchema


async def extract_to_json(text: str, schema: GraphSchema, llm, out_path: Path):
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_text(text, document_id="doc:custom")

    extractor = LLMGraphExtractor(llm)
    validator = SchemaValidator()
    assembler = GraphDocumentAssembler()

    chunk_extractions = {}
    for chunk in chunks:
        raw = await extractor.extract(chunk.text, schema=schema)
        chunk_extractions[chunk.id] = validator.validate(raw, schema)

    graph_doc = assembler.assemble(
        document_id="doc:custom",
        text_hash="manual-hash",
        chunks=chunks,
        chunk_extractions=chunk_extractions,
        metadata={"source": "advanced"},
        graph_name="entity-graph",
    )

    out_path.write_text(
        json.dumps(
            {
                "document": {"id": graph_doc.document.id, **graph_doc.document.metadata},
                "entities": [{"id": e.id, "type": e.type, **e.properties} for e in graph_doc.entities],
                "relationships": [
                    {"source": r.source_id, "target": r.target_id, "type": r.type, **r.properties}
                    for r in graph_doc.relationships
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return graph_doc
```

## Next steps

- Combine these primitives with custom `GraphWriter` for logging or caching.
- See [`docs/pipelines.md`](pipelines.md) for the high-level pipeline that wraps these steps.
- See [`docs/search.md`](search.md) for the high-level `GraphRAG.search()` API.
