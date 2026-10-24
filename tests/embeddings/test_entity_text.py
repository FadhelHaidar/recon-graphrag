"""Model identities use the actual configured provider model string."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from recon_graphrag.embeddings.factory import OpenAIEmbeddings, OllamaEmbeddings
from recon_graphrag.entity_text import _model_name, embedder_fingerprint


@pytest.mark.parametrize("model", [None, "", "text-embedding-3-small", "nomic-embed-text:latest"])
def test_model_name_preserves_string_identifiers(model):
    assert _model_name(model) == model


@pytest.mark.parametrize("adapter,module,clients,models", [
    (OpenAIEmbeddings, "openai", ["OpenAI", "AsyncOpenAI"],
     ["text-embedding-3-small", "text-embedding-3-large"]),
    (OllamaEmbeddings, "ollama", ["Client", "AsyncClient"],
     ["nomic-embed-text:latest", "mxbai-embed-large"]),
])
def test_real_provider_configurations_have_distinct_model_fingerprints(
    monkeypatch, adapter, module, clients, models
):
    # Construct real adapters, but never create a real client or read credentials.
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(
        **{client: MagicMock() for client in clients}
    ))
    first = adapter(model=models[0])
    second = adapter(model=models[1])
    assert _model_name(first.model) == models[0]
    assert _model_name(second.model) == models[1]
    assert embedder_fingerprint(first) != embedder_fingerprint(second)
    assert embedder_fingerprint(first) == embedder_fingerprint(adapter(model=models[0]))


def test_object_model_name_and_class_fallback_remain_supported():
    assert _model_name(SimpleNamespace(model_name_or_path="local-model")) == "local-model"
    assert _model_name(object()) == "builtins.object"
