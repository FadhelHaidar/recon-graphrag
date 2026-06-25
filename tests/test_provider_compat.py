from types import SimpleNamespace

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
