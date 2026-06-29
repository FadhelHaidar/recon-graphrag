"""Tests for token counting utilities."""

from __future__ import annotations

import pytest

from recon_graphrag.config.settings import PipelineConfig
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    PackItem,
    TiktokenTokenCounter,
    count_tokens,
    create_token_counter,
    pack_items,
    truncate_text,
)


class TestApproximateTokenCounter:
    def test_empty_text_returns_zero(self):
        counter = ApproximateTokenCounter()
        assert counter.count("") == 0

    def test_ascii_text_estimate(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        # 8 chars / 4 = 2 tokens
        assert counter.count("abcdefgh") == 2

    def test_unicode_text_counts_characters(self):
        counter = ApproximateTokenCounter(ratio=2.0)
        # Each emoji is one character
        assert counter.count("🙂🙂🙂🙂") == 2

    def test_custom_ratio(self):
        counter = ApproximateTokenCounter(ratio=2.0)
        assert counter.count("abcd") == 2

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            ApproximateTokenCounter(ratio=0)

    def test_truncate_empty_text(self):
        counter = ApproximateTokenCounter()
        assert counter.truncate("", 10) == ""

    def test_truncate_zero_tokens(self):
        counter = ApproximateTokenCounter()
        assert counter.truncate("hello", 0) == ""

    def test_truncate_negative_raises(self):
        counter = ApproximateTokenCounter()
        with pytest.raises(ValueError):
            counter.truncate("hello", -1)

    def test_truncate_respects_ratio(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        # 8 chars max for 2 tokens
        assert counter.truncate("abcdefghij", 2) == "abcdefgh"

    def test_multiline_text(self):
        counter = ApproximateTokenCounter(ratio=4.0)
        text = "line one\nline two\nline three"
        assert counter.count(text) == math_ceil(len(text) / 4.0)


def math_ceil(value: float) -> int:
    import math

    return math.ceil(value)


class TestCreateTokenCounter:
    def test_create_approximate(self):
        counter = create_token_counter("approximate")
        assert isinstance(counter, ApproximateTokenCounter)

    def test_create_approximate_with_ratio(self):
        counter = create_token_counter("approximate", ratio=3.0)
        assert counter.count("abc") == 1

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            create_token_counter("unknown")

    def test_create_tiktoken(self):
        counter = create_token_counter("tiktoken")
        assert isinstance(counter, TiktokenTokenCounter)

    def test_create_tiktoken_with_encoding(self):
        counter = create_token_counter("tiktoken", model="cl100k_base")
        assert isinstance(counter, TiktokenTokenCounter)
        # "hello world" is two tokens in cl100k_base
        assert counter.count("hello world") == 2

    def test_create_tiktoken_unknown_encoding_raises(self):
        with pytest.raises(ImportError):
            create_token_counter("tiktoken", model="unknown-encoding-xyz")


class TestConvenienceFunctions:
    def test_count_tokens_default_counter(self):
        assert count_tokens("abcdefgh") == 2  # default ratio 4

    def test_truncate_text_default_counter(self):
        assert truncate_text("abcdefghij", 2) == "abcdefgh"


class TestPackItems:
    def test_pack_items_uses_existing_order(self):
        items = [
            PackItem(id="low", text="aaaa", priority=0.0),
            PackItem(id="high", text="bbbb", priority=10.0),
            PackItem(id="third", text="cccc", priority=5.0),
        ]

        result = pack_items(
            items,
            max_tokens=2,
            counter=ApproximateTokenCounter(ratio=4.0),
        )

        assert [item.id for item in result.included] == ["low", "high"]
        assert [item.id for item in result.excluded] == ["third"]
        assert result.used_tokens == 2

    def test_pack_items_can_truncate_oversized_item(self):
        items = [PackItem(id="large", text="abcdefghij")]

        result = pack_items(
            items,
            max_tokens=2,
            counter=ApproximateTokenCounter(ratio=4.0),
            truncate_oversized=True,
        )

        assert len(result.included) == 1
        assert result.included[0].text == "abcdefgh"
        assert result.excluded == []
        assert result.truncated_item_ids == ["large"]

    def test_pack_items_rejects_negative_budget(self):
        with pytest.raises(ValueError):
            pack_items([], max_tokens=-1)

    def test_pack_items_empty_input_returns_empty(self):
        from recon_graphrag.utils.tokens import PackItem

        result = pack_items([], max_tokens=100, counter=ApproximateTokenCounter())
        assert result.included == []
        assert result.excluded == []
        assert result.truncated_item_ids == []


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.chunk_size == 1200
        assert cfg.chunk_overlap == 100
        assert cfg.max_gleanings == 1
        assert cfg.use_mixed_context is False
        assert cfg.token_counter is None

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_overlap=1200)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_overlap=-1)
