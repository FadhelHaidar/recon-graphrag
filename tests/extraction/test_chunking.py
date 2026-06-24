"""Tests for text chunking and page window building."""

import pytest

from recon_graphrag.extraction.chunking import PageWindowBuilder, TextChunker
from recon_graphrag.utils.tokens import ApproximateTokenCounter


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


def test_page_window_builder_preserves_page_record_metadata():
    builder = PageWindowBuilder(window_size=2, window_overlap=1)
    pages = [
        {
            "text": "Page one",
            "metadata": {
                "record_id": "page-1",
                "source": "source-1",
                "collection": "movies",
            },
        },
        {
            "text": "Page two",
            "metadata": {
                "record_id": "page-2",
                "source": "source-2",
                "collection": "movies",
            },
        },
        {
            "text": "Page three",
            "metadata": {
                "record_id": "page-3",
                "source": "source-3",
                "collection": "movies",
            },
        },
    ]

    chunks = builder.build_windows(
        pages,
        document_id="doc1",
        metadata={"source": "document-source", "tenant": "acme"},
    )

    assert chunks[0].text == "Page one\n\nPage two"
    assert chunks[0].metadata["source"] == "document-source"
    assert chunks[0].metadata["tenant"] == "acme"
    assert chunks[0].metadata["record_ids"] == ["page-1", "page-2"]
    assert chunks[0].metadata["source_ids"] == ["source-1", "source-2"]
    assert chunks[0].metadata["collections"] == ["movies"]
    assert chunks[1].metadata["record_ids"] == ["page-2", "page-3"]


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


# ---------------------------------------------------------------------------
# Token-aware chunking tests
# ---------------------------------------------------------------------------


def test_text_chunker_default_unit_is_characters():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    assert chunker.unit == "characters"
    assert chunker.token_counter is None


def test_text_chunker_token_unit_requires_counter():
    with pytest.raises(ValueError, match="token_counter is required"):
        TextChunker(chunk_size=10, chunk_overlap=2, unit="tokens")


def test_text_chunker_invalid_unit():
    with pytest.raises(ValueError, match="unit must be"):
        TextChunker(chunk_size=10, chunk_overlap=2, unit="words")


def test_text_chunker_token_basic():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    # 40 chars = 10 tokens at ratio=4
    text = "a" * 40
    chunks = chunker.chunk_text(text, document_id="doc1")

    assert len(chunks) > 0
    assert all(c.text for c in chunks)
    # First chunk should be ~20 chars (5 tokens * 4 chars/token)
    assert chunks[0].text == "a" * 20
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == 20
    assert chunks[0].metadata["token_start"] == 0
    assert chunks[0].metadata["token_end"] == 5


def test_text_chunker_token_overlap():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    text = "a" * 40
    chunks = chunker.chunk_text(text, document_id="doc1")

    # Step = 5 - 1 = 4 tokens = 16 chars
    # Chunk 0: chars 0-20, Chunk 1: chars 16-36, etc.
    assert len(chunks) >= 2
    assert chunks[1].metadata["char_start"] == 16
    assert chunks[1].metadata["token_start"] == 4


def test_text_chunker_token_empty_text():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    chunks = chunker.chunk_text("", document_id="doc1")
    assert chunks == []


def test_text_chunker_token_preserves_metadata():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    text = "a" * 40
    chunks = chunker.chunk_text(text, document_id="doc1", metadata={"source": "test"})
    assert chunks[0].metadata["source"] == "test"
    assert "char_start" in chunks[0].metadata
    assert "token_start" in chunks[0].metadata


def test_text_chunker_token_whitespace_only_skipped():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    text = "   \n   "
    chunks = chunker.chunk_text(text, document_id="doc1")
    assert chunks == []


def test_text_chunker_token_stable_ids():
    counter = ApproximateTokenCounter(ratio=4.0)
    chunker = TextChunker(chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter)
    text = "a" * 40
    chunks = chunker.chunk_text(text, document_id="doc1")
    assert chunks[0].id == "doc1:chunk:0"
    assert chunks[1].id == "doc1:chunk:1"


def test_text_chunker_token_char_mode_unchanged():
    """Existing character-mode tests must still work."""
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunker.chunk_text(text, document_id="doc1")

    assert len(chunks) > 0
    assert chunks[0].id == "doc1:chunk:0"
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == 10
    assert "token_start" not in chunks[0].metadata
