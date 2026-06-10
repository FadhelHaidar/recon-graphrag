"""Tests for text chunking and page window building."""

import pytest

from recon_graphrag.extraction.chunking import PageWindowBuilder, TextChunker


def test_text_chunker_basic():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunker.chunk_text(text, document_id="doc1")

    assert len(chunks) > 0
    assert all(c.text for c in chunks)
    assert chunks[0].id == "doc1:chunk:0"
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == 10


def test_text_chunker_empty_text():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_text("", document_id="doc1")
    assert chunks == []


def test_text_chunker_metadata_propagation():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_text(
        "hello world foo bar baz",
        document_id="doc1",
        metadata={"source": "test"},
    )
    assert chunks[0].metadata["source"] == "test"
    assert "char_start" in chunks[0].metadata


def test_text_chunker_invalid_config():
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
        TextChunker(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_page_window_builder_basic():
    builder = PageWindowBuilder(window_size=2, window_overlap=1)
    pages = ["Page one", "Page two", "Page three", "Page four"]
    chunks = builder.build_windows(pages, document_id="doc1")

    assert len(chunks) == 3
    assert chunks[0].id == "doc1:pages:1-2"
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2
    assert chunks[1].id == "doc1:pages:2-3"
    assert chunks[2].id == "doc1:pages:3-4"


def test_page_window_builder_empty_pages():
    builder = PageWindowBuilder(window_size=2, window_overlap=1)
    chunks = builder.build_windows([], document_id="doc1")
    assert chunks == []


def test_page_window_builder_invalid_config():
    with pytest.raises(ValueError, match="window_size must be > 0"):
        PageWindowBuilder(window_size=0)
    with pytest.raises(ValueError, match="window_overlap must be >= 0"):
        PageWindowBuilder(window_size=2, window_overlap=-1)
    with pytest.raises(ValueError, match="window_overlap must be < window_size"):
        PageWindowBuilder(window_size=2, window_overlap=2)
