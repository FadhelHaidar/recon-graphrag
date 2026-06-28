import math
import pytest

from recon_graphrag import create_embedder, create_llm
from tests.integration.support import require_integration_env


RUN_FLAG = "RUN_PROVIDER_INTEGRATION_TESTS"
REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME",
]


def _azure_env_or_skip():
    import os

    require_integration_env(RUN_FLAG, REQUIRED_ENV, "Azure OpenAI integration tests")

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    return {
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "base_url": f"{endpoint}/openai/v1/",
        "llm_deployment": os.environ["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
        "embed_deployment": os.environ["AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME"],
    }


@pytest.mark.integration
@pytest.mark.provider
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
@pytest.mark.provider
async def test_azure_openai_embedding_env_endpoint():
    env = _azure_env_or_skip()
    embedder = create_embedder(
        "openai",
        model=env["embed_deployment"],
        api_key=env["api_key"],
        base_url=env["base_url"],
    )

    try:
        embedding = await embedder.async_embed_query("Azure OpenAI endpoint check")
    finally:
        close = getattr(embedder, "aclose", None)
        if callable(close):
            await close()

    assert embedding
    assert all(math.isfinite(value) for value in embedding)
