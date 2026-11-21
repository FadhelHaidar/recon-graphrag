"""Built-in embedder identities are part of the report freshness contract."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from recon_graphrag.embeddings.factory import OllamaEmbeddings, OpenAIEmbeddings
from recon_graphrag.entity_text import embedder_fingerprint


def test_builtin_embedding_model_fingerprints_are_pinned(monkeypatch):
    # The report freshness path compares this fingerprint with stored vectors.
    # Keep built-in identities byte-identical while adding optional hooks.
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=MagicMock(), AsyncOpenAI=MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=MagicMock(), AsyncClient=MagicMock()),
    )

    assert embedder_fingerprint(OpenAIEmbeddings(model="text-embedding-3-small")) == (
        "d33e01e9d7b59554e6be030f9a57b4cbfff571e79ae344b3bb04176838145ca7"
    )
    assert embedder_fingerprint(OpenAIEmbeddings(model="text-embedding-3-large")) == (
        "048046fca714d304fb8a88b0ab884499498cf2e1bb23380db3e71168f2d960ff"
    )
    assert embedder_fingerprint(OllamaEmbeddings(model="nomic-embed-text:latest")) == (
        "e1303957dee27b14fe683243d3964dc0e53c0e2753e126f86f994a8dddc131b3"
    )
