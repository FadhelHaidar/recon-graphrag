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


class _WhitespaceIdCounter:
    """Token counter exposing encode/decode (triggers the fast path).

    Models a byte-level tokenizer: each token carries its own trailing
    whitespace, so ``decode`` is exact concatenation and slicing token IDs
    reconstructs the original substring (unlike partial tiktoken byte tokens).
    """

    def encode(self, text):
        import re

        return re.findall(r"\S+\s*", text)

    def decode(self, tokens):
        return "".join(tokens)

    def count(self, text):
        return len(self.encode(text))

    def truncate(self, text, max_tokens):
        return "".join(self.encode(text)[:max_tokens])


def test_text_chunker_token_fast_path_reconstructs_source():
    counter = _WhitespaceIdCounter()
    chunker = TextChunker(
        chunk_size=5, chunk_overlap=1, unit="tokens", token_counter=counter
    )
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunker.chunk_text(text, document_id="doc1")

    assert len(chunks) > 1
    # char offsets must slice the original text back to each chunk verbatim
    for chunk in chunks:
        m = chunk.metadata
        assert text[m["char_start"]:m["char_end"]] == chunk.text
    # overlap: consecutive chunks share tokens (token_start advances by step=4)
    assert chunks[1].metadata["token_start"] == 4


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


def _assert_token_chunks(text, chunks, counter, budget, overlap):
    assert chunks
    previous_start = -1
    previous_end = 0
    for index, chunk in enumerate(chunks):
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert previous_start < start <= previous_end
        assert start < end <= len(text)
        assert chunk.text == text[start:end]
        assert counter.count(chunk.text) <= budget
        assert chunk.metadata["token_end"] - chunk.metadata["token_start"] == counter.count(chunk.text)
        assert chunk.id == f"unicode:chunk:{index}"
        assert chunk.index == index
        if previous_start >= 0:
            assert counter.count(text[start:previous_end]) <= overlap
        previous_start, previous_end = start, end
    assert previous_end == len(text)


@pytest.mark.parametrize("text", ["🚀" * 1500, "x🚀" * 1000])
def test_text_chunker_original_1200_100_emoji_boundary(text):
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter()
    chunks = TextChunker(1200, 100, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 1200, 100)
    assert len(chunks) > 1
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    assert all(chunk.metadata["char_start"] < chunks[i - 1].metadata["char_end"] for i, chunk in enumerate(chunks) if i)


@pytest.mark.parametrize("text", ["漢字東京 日本語测试" * 25, "e\u0301a\u0308" * 50, "🙂🚀字e\u0301" * 40, "literal \ufffd marker " * 20])
@pytest.mark.parametrize("overlap", [0, 1, 4, 10])
def test_text_chunker_unicode_boundaries_budgets_and_offsets(text, overlap):
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter()
    chunks = TextChunker(11, overlap, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 11, overlap)


def test_text_chunker_exact_counter_encodes_source_once(monkeypatch):
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter()
    text = "🚀漢字e\u0301 source " * 200
    original_encode = counter.encode
    encoded_texts = []

    def encode(value):
        encoded_texts.append(value)
        return original_encode(value)

    monkeypatch.setattr(counter, "encode", encode)
    chunks = TextChunker(120, 10, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 120, 10)
    assert encoded_texts == [text]


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_text_chunker_exact_empty_or_whitespace(text):
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    assert TextChunker(5, 1, "tokens", TiktokenTokenCounter()).chunk_text(text, "doc") == []


def test_text_chunker_leading_whitespace_does_not_drop_later_content():
    counter = ApproximateTokenCounter(1)
    text = " " * 10 + "abcdefghij"
    chunks = TextChunker(5, 1, "tokens", counter).chunk_text(text, "doc")
    assert chunks
    assert chunks[-1].metadata["char_end"] == len(text)
    covered = {i for c in chunks for i in range(c.metadata["char_start"], c.metadata["char_end"])}
    assert set(range(10, len(text))) <= covered
    assert all(c.text == text[c.metadata["char_start"]:c.metadata["char_end"]] for c in chunks)


class _CodepointCounter:
    def count(self, text):
        return len(text)

    def truncate(self, text, max_tokens):
        return text[:max_tokens]


class _MisleadingCodecCounter(_CodepointCounter):
    def encode(self, text):
        return [text]

    def decode(self, tokens):
        return "\ufffd"


@pytest.mark.parametrize("counter", [_CodepointCounter(), _MisleadingCodecCounter(), ApproximateTokenCounter(1.5)])
def test_text_chunker_custom_counter_not_custom_codec_controls_budget(counter):
    text = "🙂漢e\u0301abcdef" * 10
    chunks = TextChunker(5, 1, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 5, 1)
    assert "\ufffd" not in "".join(c.text for c in chunks)


class _NonmonotonicCounter:
    def count(self, text):
        return len(text) if len(text) % 2 == 0 else len(text) + 3

    def truncate(self, text, max_tokens):
        # The TokenCounter contract supplies the longest fitting prefix even
        # when a shorter prefix has a larger token count.
        return next((text[:n] for n in range(len(text), -1, -1) if self.count(text[:n]) <= max_tokens), "")


def test_text_chunker_custom_counter_counts_need_not_be_monotonic():
    counter = _NonmonotonicCounter()
    text = "abcdefghijklmnop"
    chunks = TextChunker(4, 2, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 4, 2)
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "cdef"


@pytest.mark.parametrize("exact", [False, True])
def test_text_chunker_impossible_budget_raises_safe_value_error(exact):
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter() if exact else ApproximateTokenCounter(0.5)
    with pytest.raises(ValueError, match=r"chunk_size=1.*character offset 0") as error:
        TextChunker(1, 0, "tokens", counter).chunk_text("🚀private-content", "secret-doc")
    assert "private-content" not in str(error.value)
    assert "secret-doc" not in str(error.value)


def test_text_chunker_impossible_later_codepoint_reports_original_offset():
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    with pytest.raises(ValueError, match=r"character offset 1"):
        TextChunker(1, 0, "tokens", TiktokenTokenCounter()).chunk_text("a🚀", "doc")


class _NormalizingWordCounter:
    def count(self, text):
        return len(text.split())

    def truncate(self, text, max_tokens):
        return " ".join(text.split()[:max_tokens])


def test_text_chunker_legacy_whitespace_normalizing_counter_preserves_source():
    text = "  Alice   directed Inception in 2010 with Christopher Nolan. \n" * 3 + "End."
    counter = _NormalizingWordCounter()
    chunks = TextChunker(5, 1, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 5, 1)
    assert chunks[0].text.startswith("  Alice   ")


def test_text_chunker_exact_overlap_is_preserved_when_representable():
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter()
    text = "你" * 40
    chunks = TextChunker(5, 2, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 5, 2)
    assert chunks[1].metadata["char_start"] == 3
    assert chunks[1].metadata["token_start"] == 3
    assert all(counter.count(text[b.metadata["char_start"]:a.metadata["char_end"]]) == 2 for a, b in zip(chunks, chunks[1:], strict=False))


def test_text_chunker_unrepresentable_overlap_still_makes_progress():
    from recon_graphrag.utils.tokens import TiktokenTokenCounter

    counter = TiktokenTokenCounter()
    text = "🚀" * 4
    chunks = TextChunker(3, 2, "tokens", counter).chunk_text(text, "unicode")
    _assert_token_chunks(text, chunks, counter, 3, 2)
    assert [c.metadata["char_start"] for c in chunks] == [0, 1, 2, 3]
    assert [c.text for c in chunks] == ["🚀"] * 4
