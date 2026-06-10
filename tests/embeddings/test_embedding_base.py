"""Tests for local embedding interface types."""

import pytest

from recon_graphrag.embeddings import BaseEmbedder, ModelParamsEmbedder


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_query(self, text: str, **kwargs):
        self.calls.append(("sync", text, kwargs))
        return [1.0, 2.0]

    async def async_embed_query(self, text: str, **kwargs):
        self.calls.append(("async", text, kwargs))
        return [3.0, 4.0]


def test_fake_embedder_matches_base_protocol():
    embedder = FakeEmbedder()

    assert isinstance(embedder, BaseEmbedder)
    assert embedder.embed_query("hello") == [1.0, 2.0]


@pytest.mark.asyncio
async def test_model_params_embedder_merges_params_for_sync_and_async():
    inner = FakeEmbedder()
    embedder = ModelParamsEmbedder(inner, {"dimensions": 3, "encoding_format": "float"})

    assert embedder.embed_query("hello", dimensions=2) == [1.0, 2.0]
    assert await embedder.async_embed_query("world") == [3.0, 4.0]

    assert inner.calls == [
        ("sync", "hello", {"dimensions": 2, "encoding_format": "float"}),
        ("async", "world", {"dimensions": 3, "encoding_format": "float"}),
    ]
