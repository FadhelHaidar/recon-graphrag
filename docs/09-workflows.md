# Advanced Workflows

Recon-GraphRAG's high-level examples ([`GraphBuilderPipeline`](05-pipelines.md), [`GraphRAG.search()`](06-search.md)) cover the most common path, but the library is composed of smaller, interchangeable building blocks. This guide explains those primitives so you can assemble custom workflows, implement new backends, or inspect intermediate artifacts without running a full pipeline.

## When to use the building blocks

Reach for these primitives when you want to:

- Extract entities once and save the structured output to JSON/Parquet before database ingestion.
- Call `LocalSearchRetriever`, `GlobalSearchRetriever`, or `DriftSearchRetriever` directly instead of through `GraphRAG`.
- Implement a custom `GraphStore` backend (Postgres, in-memory, etc.).
- Add side effects (logging, caching, validation) by wrapping `GraphWriter`.
- Swap in your own LLM or embedder that does not fit the factory functions.

---

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

After validation and assembly, the corresponding `GraphDocument` contains
normalized records. Entity `id` values are UUIDs; the extraction IDs are
preserved as readable `canonical_key` / `human_readable_id` values.

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
            id="<uuid-for-ent-1>",
            type="Person",
            canonical_key="ent-1",
            human_readable_id="ent-1",
            properties={
                "name": "Christopher Nolan",
                "title": "Christopher Nolan",
                "description": "Film director",
                "canonical_key": "ent-1",
                "human_readable_id": "ent-1",
            },
            graph_name="entity-graph",
        ),
        EntityRecord(
            id="<uuid-for-ent-2>",
            type="Movie",
            canonical_key="ent-2",
            human_readable_id="ent-2",
            properties={
                "name": "Inception",
                "title": "Inception",
                "description": "Science fiction film",
                "canonical_key": "ent-2",
                "human_readable_id": "ent-2",
            },
            graph_name="entity-graph",
        ),
        EntityRecord(
            id="<uuid-for-ent-3>",
            type="Person",
            canonical_key="ent-3",
            human_readable_id="ent-3",
            properties={
                "name": "Hans Zimmer",
                "title": "Hans Zimmer",
                "description": "Composer",
                "canonical_key": "ent-3",
                "human_readable_id": "ent-3",
            },
            graph_name="entity-graph",
        ),
    ],
    relationships=[
        RelationshipRecord(
            id="<uuid-for-ent-1-directed-ent-2>",
            source_id="<uuid-for-ent-1>",
            target_id="<uuid-for-ent-2>",
            type="DIRECTED",
            properties={
                "canonical_key": "ent-1:DIRECTED:ent-2",
                "human_readable_id": "ent-1:DIRECTED:ent-2",
                "description": "Christopher Nolan directed Inception",
                "weight": 1.0,
                "source_chunk_ids": ["doc:inception-example:chunk:0"],
            },
            graph_name="entity-graph",
        ),
        RelationshipRecord(
            id="<uuid-for-ent-3-composed-music-ent-2>",
            source_id="<uuid-for-ent-3>",
            target_id="<uuid-for-ent-2>",
            type="COMPOSED_MUSIC",
            properties={
                "canonical_key": "ent-3:COMPOSED_MUSIC:ent-2",
                "human_readable_id": "ent-3:COMPOSED_MUSIC:ent-2",
                "description": "Hans Zimmer composed music for Inception",
                "weight": 1.0,
                "source_chunk_ids": ["doc:inception-example:chunk:0"],
            },
            graph_name="entity-graph",
        ),
    ],
    evidence_links=[
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="<uuid-for-ent-1>", graph_name="entity-graph"),
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="<uuid-for-ent-2>", graph_name="entity-graph"),
        EvidenceLink(chunk_id="doc:inception-example:chunk:0", entity_id="<uuid-for-ent-3>", graph_name="entity-graph"),
    ],
    claims=[],
)
```

---

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
    max_gleanings=0,
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
    chunk_claims=None,  # optional: dict[chunk_id, list[ExtractedClaim]]
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
page_chunks = window_builder.build_windows(
    pages,
    document_id="doc:test",
    metadata={"source": "pages"},
)
```

---

## GraphStore / GraphWriter

[`recon_graphrag.graphdb.base`](../recon_graphrag/graphdb/base.py) defines two protocols:

- `GraphStore` — the full database abstraction used by pipelines and retrievers.
- `GraphWriter` — the minimal write-only contract (`write_graph_document`).

`GraphBuilderPipeline` accepts a `graph_writer` parameter that only needs to
satisfy `GraphWriter`. Use this when you want to intercept the assembled
`GraphDocument` before it is written to the database.

### Save extracted entities and relationships as JSON

After extraction, the assembled `GraphDocument` contains the document, chunks,
entities, relationships, and evidence links. Save it with
`save_graph_document_json()` when you want a reusable JSON artifact before
database ingestion.

```python
from recon_graphrag.extraction.artifacts import save_graph_document_json

save_graph_document_json(graph_document, "artifacts/my_graph.json")
```

Use `load_graph_document_json()` later when you want to ingest the saved
artifact without running extraction again.

### Saved JSON shape

The file produced by `save_graph_document_json()` contains the complete
`GraphDocument`:

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
    {
      "id": "<uuid-for-ent-1>",
      "type": "Person",
      "canonical_key": "ent-1",
      "human_readable_id": "ent-1",
      "properties": {
        "name": "Christopher Nolan",
        "title": "Christopher Nolan",
        "description": "Film director",
        "canonical_key": "ent-1",
        "human_readable_id": "ent-1"
      }
    },
    {
      "id": "<uuid-for-ent-2>",
      "type": "Movie",
      "canonical_key": "ent-2",
      "human_readable_id": "ent-2",
      "properties": {
        "name": "Inception",
        "title": "Inception",
        "description": "Science fiction film",
        "canonical_key": "ent-2",
        "human_readable_id": "ent-2"
      }
    },
    {
      "id": "<uuid-for-ent-3>",
      "type": "Person",
      "canonical_key": "ent-3",
      "human_readable_id": "ent-3",
      "properties": {
        "name": "Hans Zimmer",
        "title": "Hans Zimmer",
        "description": "Composer",
        "canonical_key": "ent-3",
        "human_readable_id": "ent-3"
      }
    }
  ],
  "relationships": [
    {
      "id": "<uuid-for-ent-1-directed-ent-2>",
      "source_id": "<uuid-for-ent-1>",
      "target_id": "<uuid-for-ent-2>",
      "type": "DIRECTED",
      "properties": {
        "canonical_key": "ent-1:DIRECTED:ent-2",
        "human_readable_id": "ent-1:DIRECTED:ent-2",
        "description": "Christopher Nolan directed Inception",
        "weight": 1.0,
        "source_chunk_ids": ["doc:inception-example:chunk:0"]
      }
    },
    {
      "id": "<uuid-for-ent-3-composed-music-ent-2>",
      "source_id": "<uuid-for-ent-3>",
      "target_id": "<uuid-for-ent-2>",
      "type": "COMPOSED_MUSIC",
      "properties": {
        "canonical_key": "ent-3:COMPOSED_MUSIC:ent-2",
        "human_readable_id": "ent-3:COMPOSED_MUSIC:ent-2",
        "description": "Hans Zimmer composed music for Inception",
        "weight": 1.0,
        "source_chunk_ids": ["doc:inception-example:chunk:0"]
      }
    }
  ],
  "evidence_links": [
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "<uuid-for-ent-1>"},
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "<uuid-for-ent-2>"},
    {"chunk_id": "doc:inception-example:chunk:0", "entity_id": "<uuid-for-ent-3>"}
  ],
  "claims": []
}
```

### Save while using GraphBuilderPipeline

If you want the normal pipeline to write to the database and also save each
assembled `GraphDocument`, wrap the store with a small `GraphWriter`.

```python
from pathlib import Path

from recon_graphrag import GraphBuilderPipeline
from recon_graphrag.extraction.artifacts import save_graph_document_json
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphWriter


class ArtifactSavingWriter:
    def __init__(self, inner: GraphWriter, out_dir: str | Path):
        self.inner = inner
        self.out_dir = Path(out_dir)

    def write_graph_document(self, graph_document: GraphDocument) -> dict[str, int]:
        safe_id = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in graph_document.document.id
        )
        save_graph_document_json(
            graph_document,
            self.out_dir / f"{safe_id}.json",
        )
        return self.inner.write_graph_document(graph_document)


pipeline = GraphBuilderPipeline(
    graph_store=store,
    llm=llm,
    embedder=embedder,
    schema=schema,
    graph_writer=ArtifactSavingWriter(store, out_dir="artifacts"),
)
```

### Implement a custom GraphStore

Implement every method in `GraphStore` to add a new backend. The protocol defines around 30 methods spanning index management, entity resolution, vector/keyword search, community detection/persistence, and validation. The Neo4j and Memgraph store tests show the shared contract alongside backend-specific query behavior.

```python
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.extraction.types import GraphDocument

class InMemoryGraphStore(GraphStore):
    def write_graph_document(self, graph_document: GraphDocument) -> dict[str, int]:
        ...

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        ...

    # ... implement remaining GraphStore methods
```

See [`recon_graphrag/graphdb/base.py`](../recon_graphrag/graphdb/base.py) for the complete protocol.

---

## Run entity resolution independently

`GraphBuilderPipeline` calls entity resolution for you after writing extracted
entities, but the same operation is also available directly on any graph store
that implements `GraphStore.resolve_entities()`. Use this when you have already
loaded graph data and want to inspect or merge duplicate entities without
running extraction again.

Start with a dry run:

```python
import asyncio

from examples.config import get_neo4j_store


async def main():
    store = get_neo4j_store()

    result = await store.resolve_entities(
        graph_name="entity-graph",
        strategy="normalized",
        dry_run=True,
    )
    print(result)


asyncio.run(main())
```

Then run the same strategy without `dry_run` to apply merges:

```python
result = await store.resolve_entities(
    graph_name="entity-graph",
    strategy="normalized",
    dry_run=False,
)
```

Supported strategies:

| Strategy | Behavior |
| ---- | ---- |
| `exact` | Merge entities with the same raw `resolve_property` value. |
| `normalized` | Merge case, punctuation, whitespace, and common organization-suffix variants. |
| `fuzzy` | Use string similarity to merge high-confidence matches and return lower-confidence review candidates. |
| `hybrid` | Combine normalized/fuzzy matching with optional aliases, embeddings, and LLM review. |

For `hybrid`, pass the extra signals you want to use:

```python
from examples.config import get_embedder, get_llm, get_neo4j_store

store = get_neo4j_store()
embedder = get_embedder()
llm = get_llm()

result = await store.resolve_entities(
    graph_name="entity-graph",
    strategy="hybrid",
    dry_run=True,
    embedder=embedder,
    llm=llm,
    aliases={
        "IBM": ["International Business Machines"],
        "Person": {
            "Bob Iger": ["Robert Iger"],
        },
    },
    llm_guidance="Only merge people when the context clearly refers to the same individual.",
)
```

Hybrid LLM review uses compact entity profiles rather than only candidate
names. Profiles include safe defaults such as display names, descriptions,
readable keys, aliases, labels, and non-internal properties. You can narrow the
extra properties sent to the LLM:

```python
result = await store.resolve_entities(
    graph_name="entity-graph",
    strategy="hybrid",
    dry_run=True,
    embedder=embedder,
    llm=llm,
    context_properties={
        "Movie": ["year", "description"],
        "Person": ["description", "birth_date"],
    },
)
```

Use conflict properties when same-name entities should stay separate if a
domain key differs:

```python
result = await store.resolve_entities(
    graph_name="entity-graph",
    strategy="hybrid",
    dry_run=True,
    embedder=embedder,
    llm=llm,
    conflict_properties={
        "Movie": ["year"],
        "Person": ["birth_date"],
    },
    llm_guidance="Do not merge entities when configured properties conflict.",
)
```

For example, `Movie` entities named `Titanic` with `year=1953` and `year=1997`
are returned as blocked review candidates when `conflict_properties={"Movie":
["year"]}` is configured. Missing values do not block; explicit unequal values
do.

By default, the LLM review is returned for audit but does not merge the review
candidates. To let hybrid resolution apply LLM-approved merges, opt in
explicitly:

```python
result = await store.resolve_entities(
    graph_name="entity-graph",
    strategy="hybrid",
    embedder=embedder,
    llm=llm,
    allow_ai_auto_merge=True,
)
```

Auto-merge only promotes candidates whose LLM review returns
`same_entity=true`, `merge_allowed=true`, and a confidence at or above
`merge_threshold / 100`. Use `dry_run=True` with the same options first when you
want to inspect `ai_merged_review_groups` before mutating the graph.

The returned dictionary includes `merged_groups`, `merged_nodes`,
`candidate_groups`, `review_groups`, `ai_merged_review_groups`, and `signals`.
In a dry run, `merged_groups` reports groups that would be merged, while
`merged_nodes` stays `0` because no write occurs.

### How this relates to pipeline deduplication

There are two deduplication moments in the graph-building flow:

| Step | Where it happens | Scope |
| ---- | ---- | ---- |
| Assembly deduplication | `GraphDocumentAssembler` | In-memory, within the `GraphDocument` being assembled from one extraction run. It collapses repeated extracted entities and relationships before database write. |
| Entity resolution | `GraphStore.resolve_entities()` | Database-level, across persisted `__Entity__` nodes for a `graph_name`. This is the same resolver that `GraphBuilderPipeline` calls when `perform_entity_resolution=True`. |

So independent entity resolution is not a different algorithm from the
GraphBuilderPipeline step. It is the same store-level resolver, called manually.
It is different from `GraphDocumentAssembler` deduplication, which happens
earlier and only sees the current extraction artifact.

For best results, run store-level entity resolution after graph writes and
before entity embedding or community detection. If you resolve entities after
building embeddings, communities, or summaries, treat those downstream artifacts
as stale and rebuild or validate them before relying on search quality.

Backend notes:

- Neo4j non-dry-run merges require APOC. If APOC is unavailable, the resolver
  returns a skipped result instead of failing the whole build.
- Memgraph uses its own merge implementation and does not require APOC.
- `graph_name` matters. Resolution only compares entities inside the requested
  graph.

---

## Retrieval primitives

Instead of `GraphRAG.search()`, you can instantiate retrievers directly. This is useful when you need fine-grained control over index names, prompts, or result formatting.

All retrievers return a [`SearchResult`](../recon_graphrag/models/types.py)
with `query`, `mode`, `answer`, `context`, and optional `citations`.

```python
SearchResult(
    query="Who directed Inception?",
    mode="local",
    answer="Christopher Nolan directed Inception.",
    context="Finding: Christopher Nolan (Person)\nConnections:\n  Person: Christopher Nolan -[DIRECTED]-> Movie: Inception\nEvidence:\n  Christopher Nolan directed Inception.",
    citations=[
        Citation(
            document_id="doc:custom",
            chunk_id="doc:custom:chunk:0",
            metadata={"record_id": "row-42", "collection": "movies"},
        )
    ],
)
```

Citation metadata is intentionally flexible. It is copied from document and
chunk metadata, so callers can use row IDs, item IDs, file IDs, API object IDs,
ticket IDs, page ranges, or any other source keys without changing the graph
schema.

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
    map_prompt=GLOBAL_MAP_PROMPT,
    reduce_prompt=GLOBAL_REDUCE_PROMPT,
    graph_name="entity-graph",
)
result = await retriever.search("What are the main themes?", community_level="coarsest")
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

---

## Community primitives

[`recon_graphrag.communities.pipeline.CommunityPipeline`](../recon_graphrag/communities/pipeline.py) is a convenience wrapper around two steps:

1. `graph_store.detect_communities()` — run the Leiden algorithm.
2. `CommunitySummarizer` — generate a natural-language summary per community.

You can call these steps individually if you need more control:

```python
from recon_graphrag.communities.summarization import CommunitySummarizer

summarizer = CommunitySummarizer(store, llm, graph_name="entity-graph")
summaries = await summarizer.summarize_all(level=0)
```

---

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

---

## Manual extraction flow example

Putting it together: extract and save a reusable JSON artifact without
`GraphBuilderPipeline`. This is useful when you want to inspect the extraction,
reuse the same artifact across backends, or ingest it later.

```python
import hashlib
from pathlib import Path

from recon_graphrag.extraction.artifacts import save_graph_document_json
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunker
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.validator import SchemaValidator


async def extract_to_json(text: str, schema: GraphSchema, llm, out_path: Path):
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_text(
        text,
        document_id="doc:custom",
        metadata={"record_id": "row-42", "collection": "movies"},
    )

    extractor = LLMGraphExtractor(llm)
    validator = SchemaValidator()
    assembler = GraphDocumentAssembler()

    chunk_extractions = {}
    for chunk in chunks:
        raw = await extractor.extract(chunk.text, schema=schema)
        chunk_extractions[chunk.id] = validator.validate(raw, schema)

    graph_doc = assembler.assemble(
        document_id="doc:custom",
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chunks=chunks,
        chunk_extractions=chunk_extractions,
        metadata={"source": "advanced", "record_id": "row-42"},
        graph_name="entity-graph",
    )

    save_graph_document_json(graph_doc, out_path)
    return graph_doc
```

---

## Next steps

- Combine these primitives with custom `GraphWriter` for logging or caching.
- See [`docs/05-pipelines.md`](05-pipelines.md) for the high-level pipeline that wraps these steps.
- See [`docs/06-search.md`](06-search.md) for the high-level `GraphRAG.search()` API.
