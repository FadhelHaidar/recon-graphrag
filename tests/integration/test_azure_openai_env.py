import math
import os

import pytest
from dotenv import load_dotenv

from recon_graphrag import create_embedder, create_llm


RUN_FLAG = "RUN_AZURE_OPENAI_INTEGRATION_TESTS"
REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
]


def _azure_env_or_skip():
    load_dotenv()

    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to call the real Azure OpenAI endpoint.")

    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required Azure OpenAI env vars: {', '.join(missing)}")

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    return {
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "base_url": f"{endpoint}/openai/v1/",
        "llm_deployment": os.environ["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
        "embed_deployment": os.environ["AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME"],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_azure_openai_llm_env_endpoint():
    env = _azure_env_or_skip()
    llm = create_llm(
        "openai",
        model_name=env["llm_deployment"],
        api_key=env["api_key"],
        base_url=env["base_url"],
    )

    try:
        response = await llm.ainvoke(
            "Reply with one short sentence confirming the endpoint works.",
            max_completion_tokens=32,
        )
    finally:
        await llm.aclose()

    assert response.content.strip()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_azure_openai_embedding_env_endpoint():
    env = _azure_env_or_skip()
    embedder = create_embedder(
        "openai",
        model=env["embed_deployment"],
        api_key=env["api_key"],
        base_url=env["base_url"],
    )

    embedding = await embedder.async_embed_query("Azure OpenAI endpoint check")

    assert embedding
    assert all(math.isfinite(value) for value in embedding)
