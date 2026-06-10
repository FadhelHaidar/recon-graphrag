"""Tests for local LLM interface types."""

import pytest

from recon_graphrag.llm import BaseLLM, LLMResponse, LLMUsage


class FakeLLM:
    def invoke(self, prompt: str, **kwargs):
        return LLMResponse(content=f"sync:{prompt}")

    async def ainvoke(self, prompt: str, **kwargs):
        return LLMResponse(content=f"async:{prompt}")


@pytest.mark.asyncio
async def test_fake_llm_matches_base_protocol():
    llm = FakeLLM()

    assert isinstance(llm, BaseLLM)
    assert llm.invoke("hello").content == "sync:hello"
    assert (await llm.ainvoke("hello")).content == "async:hello"


def test_llm_response_usage():
    response = LLMResponse(
        content="ok",
        usage=LLMUsage(request_tokens=1, response_tokens=2, total_tokens=3),
    )

    assert response.content == "ok"
    assert response.usage.total_tokens == 3
