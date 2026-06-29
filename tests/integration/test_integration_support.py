"""Unit tests for integration-provider selection helpers."""

import pytest

from tests.integration.support import require_selected_provider_env


def test_selected_provider_env_accepts_azure_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", "chat")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME", "embedding")

    selected = require_selected_provider_env("test scenario")

    assert selected == ("azure_openai", "azure_openai")


def test_selected_provider_env_accepts_openrouter_with_local_embedder(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "sentence-transformer")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "provider/model")

    selected = require_selected_provider_env("test scenario")

    assert selected == ("openrouter", "sentence-transformer")


def test_selected_provider_env_reports_missing_variables(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "")

    with pytest.raises(pytest.fail.Exception, match="OPENROUTER_EMBED_MODEL"):
        require_selected_provider_env("test scenario")
