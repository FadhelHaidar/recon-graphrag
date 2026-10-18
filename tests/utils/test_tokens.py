"""Tests for token counting utilities."""

from __future__ import annotations

import pytest

from recon_graphrag.config.settings import PipelineConfig
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    PackItem,
    pack_items,
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


class TestTiktokenTokenCounter:
    @pytest.mark.parametrize("text", ["🚀" * 500, "漢字東京 日本語测试" * 4, "e\u0301a\u0308" * 5, "🙂🚀字e\u0301" * 8, "literal \ufffd value"])
    @pytest.mark.parametrize("budget", [0, 1, 2, 3, 5, 11, 1200])
    def test_truncate_preserves_original_unicode_prefix_and_budget(self, text, budget):
        from recon_graphrag.utils.tokens import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        prefix = counter.truncate(text, budget)
        assert text.startswith(prefix)
        assert counter.count(prefix) <= budget
        if "\ufffd" not in text:
            assert "\ufffd" not in prefix
        if counter.count(text) <= budget:
            assert prefix == text

    def test_truncate_emoji_mid_token_returns_empty(self):
        from recon_graphrag.utils.tokens import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        assert counter.count("🚀") == 3
        assert counter.truncate("🚀", 1) == ""
        assert counter.truncate("🚀", 2) == ""
        assert counter.truncate("🚀", 3) == "🚀"

    def test_truncate_negative_and_empty(self):
        from recon_graphrag.utils.tokens import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        assert counter.truncate("", 10) == ""
        with pytest.raises(ValueError, match="max_tokens must be >= 0"):
            counter.truncate("🚀", -1)

    def test_pack_items_exact_truncation_does_not_corrupt_unicode(self):
        from recon_graphrag.utils.tokens import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
        text = "🚀" * 20
        result = pack_items([PackItem("rocket", text)], 4, counter, truncate_oversized=True)
        assert result.included == [PackItem("rocket", "🚀")]
        assert result.used_tokens == 3
        assert result.truncated_item_ids == ["rocket"]
