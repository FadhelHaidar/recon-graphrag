from types import SimpleNamespace

from recon_graphrag.llm.factory import _anthropic_response_to_llm_response
from recon_graphrag.providers import (
    OpenAICompatibleProviderError,
    _response_error,
    _safe_response_summary,
)


def test_response_error_reads_dict_payload():
    response = {"error": {"message": "rate limited", "code": 429}}

    assert _response_error(response) == {"message": "rate limited", "code": 429}


def test_response_error_reads_model_dump_payload():
    response = SimpleNamespace(
        model_dump=lambda: {
            "error": {"message": "model busy", "code": "busy"}
        }
    )

    assert _response_error(response) == {"message": "model busy", "code": "busy"}


def test_provider_error_message_uses_shared_summary():
    response = {"error": {"message": "model busy", "code": "busy"}}

    error = OpenAICompatibleProviderError.from_error(
        "chat",
        response["error"],
        response,
    )

    message = str(error)
    assert "OpenAI-compatible chat provider returned error (busy): model busy" in message
    assert "Response:" in message


def test_safe_response_summary_handles_none():
    assert _safe_response_summary(None) == "None"


def test_anthropic_response_to_llm_response():
    text_block = SimpleNamespace(type="text", text="Hello world")
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    response = SimpleNamespace(content=[text_block], usage=usage)

    result = _anthropic_response_to_llm_response(response)

    assert result.content == "Hello world"
    assert result.usage.request_tokens == 10
    assert result.usage.response_tokens == 5
    assert result.usage.total_tokens == 15


def test_anthropic_response_multiple_text_blocks():
    blocks = [
        SimpleNamespace(type="text", text="Part 1. "),
        SimpleNamespace(type="text", text="Part 2."),
    ]
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    response = SimpleNamespace(content=blocks, usage=usage)

    result = _anthropic_response_to_llm_response(response)

    assert result.content == "Part 1. Part 2."
