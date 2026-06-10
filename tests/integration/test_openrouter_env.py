import math
import os

import pytest
from dotenv import load_dotenv

from recon_graphrag import create_embedder, create_llm


RUN_FLAG = "RUN_OPENROUTER_INTEGRATION_TESTS"
REQUIRED_ENV = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_LLM_MODEL",
    "OPENROUTER_EMBED_MODEL",
]


def _openrouter_env_or_skip():
    load_dotenv()

    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to call the real OpenRouter endpoint.")

    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required OpenRouter env vars: {', '.join(missing)}")

    return {
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "llm_model": os.environ["OPENROUTER_LLM_MODEL"],
        "embed_model": os.environ["OPENROUTER_EMBED_MODEL"],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_llm_env_endpoint():
    env = _openrouter_env_or_skip()
    llm = create_llm(
        "openrouter",
        model_name=env["llm_model"],
        api_key=env["api_key"],
    )

    try:
        response = await llm.ainvoke(
            "Reply with exactly: openrouter ok",
            max_tokens=128,
        )
    finally:
        await llm.aclose()

    assert response.content.strip().lower() == "openrouter ok"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_embedding_env_endpoint():
    env = _openrouter_env_or_skip()
    embedder = create_embedder(
        "openrouter",
        model=env["embed_model"],
        api_key=env["api_key"],
    )

    embedding = await embedder.async_embed_query("OpenRouter embedding check")

    assert embedding
    assert all(math.isfinite(value) for value in embedding)
