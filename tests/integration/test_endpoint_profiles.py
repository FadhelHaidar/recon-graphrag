"""Offline regressions for the test-owned endpoint profiles (not integration)."""

import os
from itertools import product
from unittest.mock import Mock

import pytest

from tests.integration import factories, support


@pytest.fixture(autouse=True)
def provider_creators(monkeypatch):
    prefixes = (
        "LLM_", "EMBEDDING_", "EMBEDDER_", "OPENAI_", "AZURE_OPENAI_",
        "OPENROUTER_", "SENTENCE_TRANSFORMER_",
    )
    for name in os.environ:
        if name.startswith(prefixes):
            monkeypatch.delenv(name)
    monkeypatch.setattr(support, "load_dotenv", lambda: None)
    monkeypatch.setenv("RUN_PROVIDER_INTEGRATION_TESTS", "1")
    creators = {"llm": Mock(), "embedder": Mock()}
    monkeypatch.setattr(factories, "create_llm", creators["llm"])
    monkeypatch.setattr(factories, "create_embedder", creators["embedder"])
    return creators


def select_legacy_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_LLM_MODEL", "legacy-chat")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "legacy-embedding")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-legacy-key")


def validate_openai():
    assert support.require_selected_provider_env("profile test") == ("openai", "openai")
    for role in ("llm", "embedder"):
        support.require_integration_env(
            "RUN_PROVIDER_INTEGRATION_TESTS",
            support.PROVIDER_REQUIREMENTS["openai"][role],
            "profile test",
            fail_on_missing=True,
        )


def test_full_profiles_use_separate_models_urls_and_keys(monkeypatch, provider_creators):
    monkeypatch.setenv("LLM_PROVIDER", "OPENAI")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "OPENAI")
    for name, value in {
        "LLM_MODEL": "profile-chat",
        "LLM_BASE_URL": "http://chat.invalid/v1",
        "LLM_API_KEY": "synthetic-chat-key",
        "EMBEDDING_MODEL": "profile-embedding",
        "EMBEDDING_BASE_URL": "http://embedding.invalid/v1",
        "EMBEDDING_API_KEY": "synthetic-embedding-key",
        "OPENAI_BASE_URL": "http://shared.invalid/v1",
    }.items():
        monkeypatch.setenv(name, value)

    assert factories.get_llm() is provider_creators["llm"].return_value
    assert factories.get_embedder() is provider_creators["embedder"].return_value
    provider_creators["llm"].assert_called_once_with(
        "openai", model_name="profile-chat", base_url="http://chat.invalid/v1",
        api_key="synthetic-chat-key",
    )
    provider_creators["embedder"].assert_called_once_with(
        "openai", model="profile-embedding", base_url="http://embedding.invalid/v1",
        api_key="synthetic-embedding-key",
    )
    validate_openai()


@pytest.mark.parametrize("role", ["llm", "embedder"])
@pytest.mark.parametrize("model_value,key_value", list(product([None, "", "profile"], repeat=2)))
def test_partial_and_blank_profiles_fall_back_per_field(
    monkeypatch, provider_creators, role, model_value, key_value,
):
    select_legacy_openai(monkeypatch)
    prefix = "LLM" if role == "llm" else "EMBEDDING"
    if model_value is not None:
        monkeypatch.setenv(f"{prefix}_MODEL", model_value)
    if key_value is not None:
        monkeypatch.setenv(f"{prefix}_API_KEY", key_value)
    monkeypatch.setenv(f"{prefix}_BASE_URL", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://shared.invalid/v1")

    factory = factories.get_llm if role == "llm" else factories.get_embedder
    create = provider_creators[role]
    factory()
    model_arg = "model_name" if role == "llm" else "model"
    legacy_model = "legacy-chat" if role == "llm" else "legacy-embedding"
    create.assert_called_once_with(
        "openai", **{
            model_arg: model_value or legacy_model,
            "api_key": key_value or "synthetic-legacy-key",
            # None leaves OPENAI_BASE_URL / the SDK endpoint default in control.
            "base_url": None,
        },
    )
    validate_openai()


@pytest.mark.parametrize("role", ["llm", "embedder"])
@pytest.mark.parametrize("profile_field", ["MODEL", "API_KEY"])
def test_partial_profiles_can_replace_one_missing_legacy_field(
    monkeypatch, provider_creators, role, profile_field,
):
    select_legacy_openai(monkeypatch)
    prefix = "LLM" if role == "llm" else "EMBEDDING"
    legacy_model = "OPENAI_LLM_MODEL" if role == "llm" else "OPENAI_EMBED_MODEL"
    legacy_field = legacy_model if profile_field == "MODEL" else "OPENAI_API_KEY"
    monkeypatch.delenv(legacy_field)
    monkeypatch.setenv(f"{prefix}_{profile_field}", "synthetic-profile-value")
    # If the shared legacy key is removed, the other role needs its own key.
    if profile_field == "API_KEY":
        other_prefix = "EMBEDDING" if role == "llm" else "LLM"
        monkeypatch.setenv(f"{other_prefix}_API_KEY", "synthetic-other-key")

    factory = factories.get_llm if role == "llm" else factories.get_embedder
    factory()
    model_arg = "model_name" if role == "llm" else "model"
    replaced_arg = model_arg if profile_field == "MODEL" else "api_key"
    assert provider_creators[role].call_args.kwargs[replaced_arg] == "synthetic-profile-value"
    validate_openai()


def test_openai_factory_defaults_remain_available(monkeypatch, provider_creators):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openai")
    factories.get_llm()
    factories.get_embedder()
    provider_creators["llm"].assert_called_once_with(
        "openai", model_name="gpt-4o-mini", base_url=None, api_key=None,
    )
    provider_creators["embedder"].assert_called_once_with(
        "openai", model="text-embedding-3-small", base_url=None, api_key=None,
    )


@pytest.mark.parametrize("provider", [None, "OPENAI"])
@pytest.mark.parametrize("value", [None, "", "true", "false"])
@pytest.mark.parametrize("effort", [None, "", "none", "minimal", "low", "medium", "high", "xhigh"])
def test_openai_reasoning_settings_preserve_independent_embedding_profile(
    monkeypatch, provider_creators, provider, value, effort,
):
    select_legacy_openai(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "profile-embedding")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embedding.invalid/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "synthetic-embedding-key")
    if value is not None:
        monkeypatch.setenv("LLM_ENABLE_THINKING", value)
    if effort is not None:
        monkeypatch.setenv("LLM_REASONING_EFFORT", effort)

    factories.get_llm(provider)
    factories.get_embedder(provider)
    expected = {
        "model_name": "legacy-chat", "base_url": None,
        "api_key": "synthetic-legacy-key",
    }
    model_params = {}
    if value in ("true", "false"):
        model_params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": value == "true"}
        }
    if effort:
        model_params["reasoning_effort"] = effort
    if model_params:
        expected["model_params"] = model_params
    provider_creators["llm"].assert_called_once_with("openai", **expected)
    provider_creators["embedder"].assert_called_once_with(
        "openai", model="profile-embedding", base_url="http://embedding.invalid/v1",
        api_key="synthetic-embedding-key",
    )
    validate_openai()


@pytest.mark.parametrize("value", ["1", "0", "yes", "TRUE", "False", " ", "synthetic-secret"])
def test_openai_thinking_toggle_rejects_invalid_values_without_leaking_them(
    monkeypatch, provider_creators, value,
):
    select_legacy_openai(monkeypatch)
    monkeypatch.setenv("LLM_ENABLE_THINKING", value)
    with pytest.raises(ValueError) as exc:
        factories.get_llm()
    assert str(exc.value) == "LLM_ENABLE_THINKING must be true or false"
    provider_creators["llm"].assert_not_called()
    factories.get_embedder()
    provider_creators["embedder"].assert_called_once_with(
        "openai", model="legacy-embedding", base_url=None, api_key="synthetic-legacy-key",
    )


@pytest.mark.parametrize("value", ["HIGH", "false", "1", " ", "synthetic-secret"])
def test_openai_reasoning_effort_rejects_invalid_values_without_leaking_them(
    monkeypatch, provider_creators, value,
):
    select_legacy_openai(monkeypatch)
    monkeypatch.setenv("LLM_REASONING_EFFORT", value)
    with pytest.raises(ValueError) as exc:
        factories.get_llm()
    assert str(exc.value) == "Invalid LLM_REASONING_EFFORT"
    provider_creators["llm"].assert_not_called()
    factories.get_embedder()
    provider_creators["embedder"].assert_called_once_with(
        "openai", model="legacy-embedding", base_url=None, api_key="synthetic-legacy-key",
    )


@pytest.mark.parametrize("validator", ["selected", "tier"])
@pytest.mark.parametrize("fail_on_missing", [True, False])
@pytest.mark.parametrize("missing_field", ["model", "key"])
def test_validators_reject_incomplete_profiles_without_leaking_values(
    monkeypatch, validator, fail_on_missing, missing_field,
):
    select_legacy_openai(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://chat.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    # The embedder's credentials/model must not satisfy the LLM's requirements.
    monkeypatch.setenv("EMBEDDING_API_KEY", "synthetic-embedding-secret")
    monkeypatch.setenv("EMBEDDING_MODEL", "profile-embedding")
    if missing_field == "model":
        monkeypatch.delenv("OPENAI_LLM_MODEL")
        expected = "LLM_MODEL or OPENAI_LLM_MODEL"
    else:
        monkeypatch.delenv("OPENAI_API_KEY")
        expected = "LLM_API_KEY or OPENAI_API_KEY"

    exception = pytest.fail.Exception if fail_on_missing else pytest.skip.Exception
    with pytest.raises(exception, match=expected) as exc:
        if validator == "selected":
            support.require_selected_provider_env("profile test", fail_on_missing=fail_on_missing)
        else:
            support.require_integration_env(
                "RUN_PROVIDER_INTEGRATION_TESTS",
                support.PROVIDER_REQUIREMENTS["openai"]["llm"],
                "profile test",
                fail_on_missing=fail_on_missing,
            )
    assert "synthetic-" not in str(exc.value)


@pytest.mark.parametrize("validator", ["selected", "tier"])
def test_no_profile_preserves_legacy_model_validation(monkeypatch, validator):
    select_legacy_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_EMBED_MODEL")
    with pytest.raises(pytest.fail.Exception, match="OPENAI_EMBED_MODEL"):
        if validator == "selected":
            support.require_selected_provider_env("profile test")
        else:
            support.require_integration_env(
                "RUN_PROVIDER_INTEGRATION_TESTS",
                support.PROVIDER_REQUIREMENTS["openai"]["embedder"],
                "profile test",
                fail_on_missing=True,
            )


@pytest.mark.parametrize("provider", [None, "azure_openai", "openrouter", "sentence-transformer"])
def test_profiles_do_not_change_other_provider_paths(monkeypatch, provider_creators, provider):
    monkeypatch.setenv("LLM_ENABLE_THINKING", "ignored-invalid-value")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "ignored-invalid-value")
    for prefix in ("LLM", "EMBEDDING"):
        monkeypatch.setenv(f"{prefix}_MODEL", "ignored-profile-model")
        monkeypatch.setenv(f"{prefix}_API_KEY", "synthetic-ignored-key")
        monkeypatch.setenv(f"{prefix}_BASE_URL", "http://ignored.invalid/v1")

    factories.get_embedder(provider)
    if provider == "sentence-transformer":
        provider_creators["embedder"].assert_called_once_with(
            "sentence-transformer", model="all-MiniLM-L6-v2",
        )
        return

    factories.get_llm(provider)
    if provider == "openrouter":
        provider_creators["llm"].assert_called_once_with(
            "openrouter", model_name="qwen/qwen3.6-flash", api_key=None,
            model_params={"extra_body": {"reasoning": {"enabled": False}}},
        )
        provider_creators["embedder"].assert_called_once_with(
            "openrouter", model="qwen/qwen3-embedding-4b", api_key=None,
            model_params={"encoding_format": "float", "dimensions": factories.EMBEDDING_DIM},
        )
    else:
        provider_creators["llm"].assert_called_once_with(
            "azure_openai", model_name="gpt-4o", api_key=None,
            azure_endpoint=None, api_version="2024-06-01",
        )
        provider_creators["embedder"].assert_called_once_with(
            "azure_openai", model="text-embedding-3-small", api_key=None,
            azure_endpoint=None, api_version="2024-06-01",
        )


@pytest.mark.parametrize("provider", ["azure_openai", "openrouter"])
def test_openai_profiles_cannot_satisfy_other_provider_validation(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("EMBEDDER_PROVIDER", provider)
    for prefix in ("LLM", "EMBEDDING"):
        monkeypatch.setenv(f"{prefix}_MODEL", "profile-model")
        monkeypatch.setenv(f"{prefix}_API_KEY", "synthetic-profile-key")
    with pytest.raises(pytest.fail.Exception, match=provider.upper()):
        support.require_selected_provider_env("profile test")


def test_explicit_openai_selection_overrides_provider_env(monkeypatch, provider_creators):
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "profile-chat")
    monkeypatch.setenv("EMBEDDING_MODEL", "profile-embedding")
    factories.get_llm("OPENAI")
    factories.get_embedder("OPENAI")
    assert provider_creators["llm"].call_args.args == ("openai",)
    assert provider_creators["llm"].call_args.kwargs["model_name"] == "profile-chat"
    assert provider_creators["embedder"].call_args.args == ("openai",)
    assert provider_creators["embedder"].call_args.kwargs["model"] == "profile-embedding"


def test_disabled_tier_skips_before_profile_validation(monkeypatch):
    monkeypatch.delenv("RUN_PROVIDER_INTEGRATION_TESTS")
    with pytest.raises(pytest.skip.Exception, match="Set RUN_PROVIDER_INTEGRATION_TESTS=1"):
        support.require_integration_env(
            "RUN_PROVIDER_INTEGRATION_TESTS",
            support.PROVIDER_REQUIREMENTS["openai"]["llm"],
            "profile test",
            fail_on_missing=True,
        )
