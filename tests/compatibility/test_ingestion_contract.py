"""Successful ingestion contracts through the public, duck-typed extension seam."""

import hashlib
import json
from typing import Any, cast

import pytest

from recon_graphrag import (
    ColumnEntity,
    GraphBuilderPipeline,
    GraphSchema,
    LLMResponse,
    NodeType,
    PropertyType,
    RelationshipType,
    RowMapping,
    RowRelationship,
    TextColumn,
)
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.utils.tokens import TokenCounter
from tests.pipelines.test_graph_builder import FakeGraphStore


class CharacterCounter:
    """Legacy counter implementing only count/truncate, not encode/decode."""

    def __init__(self):
        self.count_calls = []
        self.truncate_calls = []

    def count(self, text: str) -> int:
        self.count_calls.append(text)
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        self.truncate_calls.append((text, max_tokens))
        return text[:max_tokens]


class ExtractionLLM:
    async def ainvoke(self, prompt):
        return LLMResponse(content=json.dumps({
            "nodes": [
                {"id": "p1", "label": "Person", "properties": {"name": "Alice", "canonical_key": "Person:Alice"}},
                {"id": "m1", "label": "Movie", "properties": {"title": "Inception"}},
            ],
            "relationships": [{"source_id": "p1", "target_id": "m1", "type": "DIRECTED"}],
        }))


class RecordingWriter:
    def __init__(self):
        self.documents = []
        self.results = []

    def write_graph_document(self, graph_document):
        assert isinstance(graph_document, GraphDocument)
        self.documents.append(graph_document)
        stats = {
            "documents": 1,
            "chunks": len(graph_document.chunks),
            "entities": len(graph_document.entities),
            "relationships": len(graph_document.relationships),
            "evidence_links": len(graph_document.evidence_links),
            "extension": {"backend": "custom", "accepted": True},
        }
        self.results.append(stats)
        return stats


class SynchronousStore(FakeGraphStore):
    """No GraphStore inheritance or new capabilities needed by old callers."""

    def __init__(self):
        super().__init__()
        self.writer = RecordingWriter()
        self.validation_calls = 0
        self.validation = {**super().validate_graph_build(), "extension_count": 7}

    def write_graph_document(self, graph_document):
        return self.writer.write_graph_document(graph_document)

    def validate_graph_build(self):
        self.validation_calls += 1
        return self.validation


@pytest.fixture
def schema():
    return GraphSchema(
        [NodeType("Person", properties=[PropertyType("name"), PropertyType("canonical_key")]),
         NodeType("Movie", properties=[PropertyType("title")], identity_property="title")],
        [RelationshipType("DIRECTED"), RelationshipType("MENTIONS")],
        [("Person", "DIRECTED", "Movie"), ("Person", "MENTIONS", "Movie")],
    )


def _hashed_doc_id(source):
    return "doc:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def assert_result(result, store, writer, index, source, chunks, *, llm_rows=None, finalize=True):
    extraction = {
        "document_id": _hashed_doc_id(source),
        "chunks": chunks,
        "write_stats": writer.results[index],
        "assembly_review_groups": [],
    }
    if llm_rows is not None:
        extraction["llm_rows"] = llm_rows
    expected = {"extraction": extraction}
    if finalize:
        expected["validation"] = store.validation
    assert result == expected
    assert result["extraction"]["write_stats"] is writer.results[index]
    if finalize:
        assert result["validation"] is store.validation
    document = writer.documents[index]
    assert document.document.id == _hashed_doc_id(source)
    assert document.document.graph_name == "compat-graph"
    assert len(document.document.text_hash) == 64
    assert all(chunk.document_id == document.document.id for chunk in document.chunks)
    assert all(entity.graph_name == "compat-graph" for entity in document.entities)
    assert len(document.entities) == 2
    assert len(document.relationships) >= 1
    assert len(document.evidence_links) >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword_call", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("injected_writer", [False, True], ids=["store-writes", "writer-writes"])
async def test_text_documents_rows_preserve_results_and_writer_dispatch(schema, keyword_call, injected_writer):
    store = SynchronousStore()
    writer = RecordingWriter() if injected_writer else store.writer
    # The deliberate partial fake locks runtime duck-typing, not protocol conformance.
    legacy_store = cast(Any, store)
    llm = cast(Any, ExtractionLLM())
    unused_embedder = cast(Any, object())
    options: dict[str, Any] = {
        "perform_entity_resolution": False, "embed_entities": False,
        "summarize_descriptions": False, "track_token_usage": False,
    }
    if keyword_call:
        pipeline = GraphBuilderPipeline(
            graph_store=legacy_store, llm=llm, embedder=unused_embedder,
            graph_name="compat-graph", graph_writer=writer if injected_writer else None, **options,
        )
    else:
        pipeline = GraphBuilderPipeline(
            legacy_store, llm, unused_embedder, "compat-graph", writer if injected_writer else None, **options,
        )
    counter = CharacterCounter()
    assert isinstance(counter, TokenCounter)
    chunking: dict[str, Any] = {
        "schema": schema, "max_gleanings": 0, "token_counter": counter,
        "chunk_size": 20, "chunk_overlap": 0,
    }
    if keyword_call:
        text_result = await pipeline.build_from_text(text="Alice made a movie.", metadata={"source": "Text"}, **chunking)
        documents = await pipeline.build_from_documents(documents=[
            {"text": "Alice made a movie.", "metadata": {"source": "Document"}},
            {"pages": ["Page one", "Page two", "Page three"], "metadata": {"source": "Pages"}},
        ], **chunking)
    else:
        text_result = await pipeline.build_from_text("Alice made a movie.", {"source": "Text"}, **chunking)
        documents = await pipeline.build_from_documents([
            {"text": "Alice made a movie.", "metadata": {"source": "Document"}},
            {"pages": ["Page one", "Page two", "Page three"], "metadata": {"source": "Pages"}},
        ], **chunking)
    assert_result(text_result, store, writer, 0, "Text", 1)
    assert len(documents) == 2
    assert_result(documents[0], store, writer, 1, "Document", 1)
    assert_result(documents[1], store, writer, 2, "Pages", 2)
    assert [chunk.id for chunk in writer.documents[0].chunks] == [
        f"{_hashed_doc_id('Text')}:chunk:0",
    ]
    assert counter.count_calls and counter.truncate_calls

    mapping = RowMapping(
        [ColumnEntity("Person", "person"), ColumnEntity("Movie", "movie", {"title": "movie"})],
        [RowRelationship("Person", "DIRECTED", "Movie")],
    )
    rows = [{"person": "Alice", "movie": "Inception"}]
    if keyword_call:
        row_result = await pipeline.build_from_rows(rows=rows, mapping=mapping, metadata={"source": "Rows"})
    else:
        row_result = await pipeline.build_from_rows(rows, mapping, {"source": "Rows"})
    assert_result(row_result, store, writer, 3, "Rows", 1, llm_rows=0)
    assert store.validation_calls == 3  # once per text, document batch, and rows
    assert len(writer.documents) == 4
    if injected_writer:
        assert store.writer.documents == []


@pytest.mark.asyncio
async def test_row_text_extension_without_finalize_preserves_result_shape(schema):
    store = SynchronousStore()
    pipeline = GraphBuilderPipeline(
        cast(Any, store), cast(Any, ExtractionLLM()), cast(Any, object()),
        "compat-graph", track_token_usage=False,
    )
    mapping = RowMapping(
        [ColumnEntity("Person", "person")],
        text_columns=[TextColumn("notes", "Person")],
    )
    result = await pipeline.build_from_rows(
        [{"person": "Alice", "notes": "Alice directed Inception."}], mapping,
        {"source": "Row-text"}, extraction_schema=schema, max_gleanings=0, finalize=False,
    )
    # Direct anchor and extracted Person have distinct IDs until resolution.
    assert set(result) == {"extraction"}
    assert result["extraction"] == {
        "document_id": _hashed_doc_id("Row-text"), "chunks": 1, "llm_rows": 1,
        "write_stats": store.writer.results[0],
        "assembly_review_groups": [],
    }
    assert result["extraction"]["write_stats"] is store.writer.results[0]
    assert store.validation_calls == 0
    assert len(store.writer.documents[0].entities) >= 2


def test_custom_counter_chunk_offsets_and_ids():
    from recon_graphrag.extraction.chunking import TextChunker

    counter = CharacterCounter()
    chunks = TextChunker(5, 1, "tokens", counter).chunk_text("abcdefghij", "doc:custom", {"source": "custom"})
    assert [(chunk.id, chunk.text, chunk.index, chunk.metadata) for chunk in chunks] == [
        ("doc:custom:chunk:0", "abcde", 0, {"source": "custom", "char_start": 0, "char_end": 5, "token_start": 0, "token_end": 5}),
        ("doc:custom:chunk:1", "efghi", 1, {"source": "custom", "char_start": 4, "char_end": 9, "token_start": 4, "token_end": 9}),
        ("doc:custom:chunk:2", "ij", 2, {"source": "custom", "char_start": 8, "char_end": 10, "token_start": 8, "token_end": 10}),
    ]
    assert counter.count_calls and counter.truncate_calls


def _usage_leaf_types(value):
    if isinstance(value, dict):
        return {key: _usage_leaf_types(item) for key, item in value.items()}
    return type(value).__name__


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["text", "documents", "rows", "row-text"])
async def test_ingestion_default_tracking_preserves_complete_usage_shape(schema, entry_point):
    store = SynchronousStore()
    # Omit track_token_usage: characterize the public default, not an opt-in.
    pipeline = GraphBuilderPipeline(
        cast(Any, store), cast(Any, ExtractionLLM()), cast(Any, object()), "compat-graph",
        perform_entity_resolution=False, embed_entities=False, summarize_descriptions=False,
    )
    chunking = {
        "schema": schema, "max_gleanings": 0, "token_counter": CharacterCounter(),
    }
    if entry_point == "text":
        results = [await pipeline.build_from_text(
            "Alice made a movie.", {"source": "Text"}, **chunking,
        )]
        sources = ["Text"]
    elif entry_point == "documents":
        results = await pipeline.build_from_documents([
            {"text": "Alice made a movie.", "metadata": {"source": "Document"}},
            {"pages": ["Alice made a movie."], "metadata": {"source": "Pages"}},
        ], **chunking)
        sources = ["Document", "Pages"]
    else:
        mapping = RowMapping(
            [ColumnEntity("Person", "person"), ColumnEntity("Movie", "movie")],
            [RowRelationship("Person", "DIRECTED", "Movie")],
            text_columns=[TextColumn("notes")] if entry_point == "row-text" else [],
        )
        results = [await pipeline.build_from_rows(
            [{"person": "Alice", "movie": "Inception", "notes": "Alice made a movie."}],
            mapping, {"source": "Rows"}, extraction_schema=schema, max_gleanings=0,
        )]
        sources = ["Rows"]

    totals_shape = {
        "calls": "int", "input_tokens": "int", "output_tokens": "int",
        "total_tokens": "int", "estimated_calls": "int",
    }
    expected_usage = {
        "totals": totals_shape,
        "stages": {"construction.extract": totals_shape},
    }
    # Document batches expose per-document deltas, not a run-level identifier.
    if entry_point != "documents":
        expected_usage["run_id"] = "str"
    for index, (result, source) in enumerate(zip(results, sources, strict=True)):
        if entry_point == "rows":
            # No LLM call means no usage record, even with default tracking on.
            assert "token_usage" not in result
        else:
            assert _usage_leaf_types(result["token_usage"]) == expected_usage
        extraction = {
            "document_id": _hashed_doc_id(source), "chunks": 1,
            "write_stats": store.writer.results[index],
            "assembly_review_groups": [],
        }
        if entry_point in {"rows", "row-text"}:
            extraction["llm_rows"] = int(entry_point == "row-text")
        assert {key: value for key, value in result.items() if key != "token_usage"} == {
            "extraction": extraction, "validation": store.validation,
        }
        assert result["extraction"]["write_stats"] is store.writer.results[index]
        assert result["validation"] is store.validation
