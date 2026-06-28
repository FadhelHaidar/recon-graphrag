"""Shared helpers for opt-in integration tests.

Tier run flags
--------------
RUN_PROVIDER_INTEGRATION_TESTS   - real LLM/embedder endpoint checks
RUN_DATABASE_INTEGRATION_TESTS   - real graph database checks (no real AI)
RUN_WORKFLOW_INTEGRATION_TESTS   - real database + deterministic fake AI
RUN_E2E_INTEGRATION_TESTS        - full real LLM/embedder/database workflow
RUN_ENTITY_RESOLUTION_AI_TESTS   - secondary gate for AI-assisted entity resolution
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv


ENABLED_VALUES = {"1", "true", "yes"}

PROVIDER_REQUIREMENTS = {
    "azure_openai": {
        "llm": [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
        ],
        "embedder": [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
        ],
    },
    "openrouter": {
        "llm": ["OPENROUTER_API_KEY", "OPENROUTER_LLM_MODEL"],
        "embedder": ["OPENROUTER_API_KEY", "OPENROUTER_EMBED_MODEL"],
    },
    "openai": {
        "llm": ["OPENAI_API_KEY", "OPENAI_LLM_MODEL"],
        "embedder": ["OPENAI_API_KEY", "OPENAI_EMBED_MODEL"],
    },
    "sentence-transformer": {
        "embedder": [],
    },
}


def require_integration_env(
    run_flag: str,
    required_env: list[str],
    description: str,
    *,
    fail_on_missing: bool = False,
) -> None:
    """Skip disabled integrations and validate their required environment."""
    load_dotenv()

    if os.getenv(run_flag, "").lower() not in ENABLED_VALUES:
        pytest.skip(f"Set {run_flag}=1 to run {description}.")

    missing = [name for name in required_env if not os.getenv(name)]
    if not missing:
        return

    message = f"Missing required {description} env vars: {', '.join(missing)}"
    if fail_on_missing:
        pytest.fail(message)
    pytest.skip(message)


def require_selected_provider_env(
    description: str,
    *,
    fail_on_missing: bool = True,
) -> tuple[str, str]:
    """Validate the explicitly selected LLM and embedder provider settings."""
    load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "azure_openai").lower()

    selections = (("llm", llm_provider), ("embedder", embedder_provider))
    required: set[str] = set()
    for role, provider in selections:
        provider_requirements = PROVIDER_REQUIREMENTS.get(provider)
        if provider_requirements is None or role not in provider_requirements:
            pytest.fail(f"Unsupported {role} provider for {description}: {provider}")
        required.update(provider_requirements[role])

    missing = sorted(name for name in required if not os.getenv(name))
    if missing:
        message = f"Missing required {description} env vars: {', '.join(missing)}"
        if fail_on_missing:
            pytest.fail(message)
        pytest.skip(message)

    return llm_provider, embedder_provider


def cleanup_graph(store, graph_name: str) -> None:
    """Delete only data owned by one test graph."""
    store.execute_query(
        "MATCH (n {graph_name: $graph_name}) DETACH DELETE n",
        {"graph_name": graph_name},
    )


def single_count(store, query: str, graph_name: str) -> int:
    """Run a scalar count query scoped to *graph_name* and return the integer result."""
    result = store.execute_query(query, {"graph_name": graph_name})
    if not result:
        return 0
    return int(next(iter(result[0].values())))
