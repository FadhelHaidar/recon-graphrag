import math
import pytest

from recon_graphrag import create_embedder, create_llm
from tests.integration.support import require_integration_env


RUN_FLAG = "RUN_PROVIDER_INTEGRATION_TESTS"
REQUIRED_ENV = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_LLM_MODEL",
    "OPENROUTER_EMBED_MODEL",
]


def _openrouter_env_or_skip():
    import os

    require_integration_env(RUN_FLAG, REQUIRED_ENV, "OpenRouter integration tests")

    return {
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "llm_model": os.environ["OPENROUTER_LLM_MODEL"],
        "embed_model": os.environ["OPENROUTER_EMBED_MODEL"],
    }


@pytest.mark.integration
@pytest.mark.provider
async def test_openrouter_llm_env_endpoint():
    env = _openrouter_env_or_skip()
    llm = create_llm(
        "openrouter",
        model_name=env["llm_model"],
        api_key=env["api_key"],
    )

    try:
        response = await llm.ainvoke(
            "Reply with one short sentence confirming the endpoint works.",
            max_tokens=128,
        )
    finally:
        await llm.aclose()

    assert response.content.strip()


@pytest.mark.integration
@pytest.mark.provider
async def test_openrouter_embedding_env_endpoint():
    env = _openrouter_env_or_skip()
    embedder = create_embedder(
        "openrouter",
        model=env["embed_model"],
        api_key=env["api_key"],
    )

    try:
        embedding = await embedder.async_embed_query("OpenRouter embedding check")
    finally:
        close = getattr(embedder, "aclose", None)
        if callable(close):
            await close()

    assert embedding
    assert all(math.isfinite(value) for value in embedding)
