"""Privacy-safe diagnostics from the shared extraction/claims boundary."""

import traceback
from json import JSONDecodeError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from recon_graphrag.extraction.schema import GraphSchema, NodeType
from recon_graphrag.extraction.types import ChunkRecord, ExtractedNode, GraphExtraction
from recon_graphrag.llm.factory import _openai_chat_response_to_llm_response
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from recon_graphrag.providers._compat import OpenAICompatibleProviderError

PRIVATE_TEXT = "private-document-provider-payload"


def http_error(status):
    return APIStatusError(
        PRIVATE_TEXT,
        response=httpx.Response(
            status, request=httpx.Request("POST", f"https://example.com/{PRIVATE_TEXT}")
        ),
        body={"secret": PRIVATE_TEXT},
    )


@pytest.fixture
def pipeline():
    return GraphBuilderPipeline(
        graph_store=MagicMock(),
        llm=MagicMock(),
        embedder=MagicMock(),
        graph_writer=MagicMock(),
        track_token_usage=False,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )


@pytest.mark.parametrize("claims", [False, True])
@pytest.mark.parametrize(
    "error, label, status, hint",
    [
        (http_error(401), "APIStatusError", 401, "credentials and model access"),
        (http_error(403), "APIStatusError", 403, "credentials and model access"),
        (http_error(429), "APIStatusError", 429, "quota/rate limits"),
        (http_error(503), "APIStatusError", 503, "retry later"),
        (http_error(400), "APIStatusError", 400, "request limits and configuration"),
        (TimeoutError(PRIVATE_TEXT), "TimeoutError", None, "connectivity/timeouts"),
        (
            ConnectionError(PRIVATE_TEXT),
            "ConnectionError",
            None,
            "connectivity/timeouts",
        ),
        (
            APITimeoutError(request=httpx.Request("POST", "https://example.com")),
            "APITimeoutError",
            None,
            "connectivity/timeouts",
        ),
        (
            APIConnectionError(request=httpx.Request("POST", "https://example.com")),
            "APIConnectionError",
            None,
            "connectivity/timeouts",
        ),
        (
            JSONDecodeError(PRIVATE_TEXT, PRIVATE_TEXT, 0),
            "JSONDecodeError",
            None,
            "schema, prompt templates",
        ),
        (ValueError(PRIVATE_TEXT), "ValueError", None, "provider response format"),
        (
            OpenAICompatibleProviderError(PRIVATE_TEXT),
            "OpenAICompatibleProviderError",
            None,
            "Provider returned an error payload",
        ),
        (RuntimeError(PRIVATE_TEXT), "RuntimeError", None, "provider configuration"),
        (
            type(PRIVATE_TEXT, (Exception,), {})(PRIVATE_TEXT),
            "Exception",
            None,
            "provider configuration",
        ),
    ],
)
async def test_extract_chunk_reports_safe_diagnostics(
    pipeline, claims, error, label, status, hint, caplog
):
    extraction = GraphExtraction(nodes=[ExtractedNode("entity", "Person")])
    pipeline.extractor.extract = AsyncMock(return_value=extraction)
    pipeline.extractor.extract_claims = AsyncMock()
    failing = (
        pipeline.extractor.extract_claims if claims else pipeline.extractor.extract
    )
    failing.side_effect = error
    chunk = ChunkRecord("chunk", PRIVATE_TEXT, PRIVATE_TEXT, 20)

    with pytest.raises(RuntimeError) as caught:
        await pipeline._extract_chunk(
            chunk, PRIVATE_TEXT, GraphSchema([NodeType("Person")], []), 0, claims
        )

    message = str(caught.value)
    assert message.startswith(
        "Claims extraction failed" if claims else "Extraction failed"
    )
    assert "document=" in message and "chunk=21" in message
    assert f"exception={label}" in message
    assert hint in message
    assert (
        (f"status={status}" in message)
        if status is not None
        else ("status=" not in message)
    )
    assert PRIVATE_TEXT not in "".join(traceback.format_exception(caught.value))
    assert PRIVATE_TEXT not in caplog.text
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    pipeline.graph_writer.write_graph_document.assert_not_called()


@pytest.mark.parametrize("status", [True, "429", 429.0, 0, 399, 600, PRIVATE_TEXT])
async def test_extract_chunk_does_not_coerce_untrusted_status(pipeline, status):
    error = RuntimeError(PRIVATE_TEXT)
    error.__dict__["status_code"] = status
    pipeline.extractor.extract = AsyncMock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        await pipeline._extract_chunk(
            ChunkRecord("chunk", PRIVATE_TEXT, PRIVATE_TEXT, 0),
            PRIVATE_TEXT,
            GraphSchema([], []),
            0,
            False,
        )
    assert "exception=RuntimeError" in str(caught.value)
    assert "status=" not in str(caught.value)
    assert PRIVATE_TEXT not in str(caught.value)


@pytest.mark.parametrize(
    "response, label, hint",
    [
        (
            {"error": {"code": 503, "message": PRIVATE_TEXT}},
            "OpenAICompatibleProviderError",
            "Provider returned an error payload",
        ),
        ({}, "ValueError", "provider response format"),
        (
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            "ValueError",
            "provider response format",
        ),
        (
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=f"{{{PRIVATE_TEXT}")
            )]),
            "JSONDecodeError",
            "schema, prompt templates",
        ),
    ],
)
async def test_real_adapter_and_parser_failures_are_reported_safely(
    pipeline, response, label, hint, caplog
):
    # Exercise the real adapter/extractor/parser path, not a mocked extractor.
    pipeline.extractor.llm.ainvoke = AsyncMock(
        side_effect=lambda *args, **kwargs: _openai_chat_response_to_llm_response(response)
    )
    with pytest.raises(RuntimeError) as caught:
        await pipeline._extract_chunk(
            ChunkRecord("chunk", PRIVATE_TEXT, PRIVATE_TEXT, 20),
            PRIVATE_TEXT,
            GraphSchema([], []),
            0,
            False,
        )

    assert f"exception={label}" in str(caught.value)
    assert hint in str(caught.value)
    assert "chunk=21" in str(caught.value)
    assert "status=" not in str(caught.value)
    assert PRIVATE_TEXT not in "".join(traceback.format_exception(caught.value))
    assert PRIVATE_TEXT not in caplog.text
    pipeline.graph_writer.write_graph_document.assert_not_called()


async def test_extract_chunk_does_not_inspect_provider_properties_or_message(pipeline):
    class PrivateProviderError(Exception):
        @property
        def status_code(self):
            raise AssertionError("must not inspect provider properties")

        @property
        def response(self):
            raise AssertionError("must not inspect provider response")

        def __str__(self):
            raise AssertionError("must not render provider message")

    pipeline.extractor.extract = AsyncMock(side_effect=PrivateProviderError())
    with pytest.raises(RuntimeError, match="exception=Exception"):
        await pipeline._extract_chunk(
            ChunkRecord("chunk", PRIVATE_TEXT, PRIVATE_TEXT, 0),
            PRIVATE_TEXT,
            GraphSchema([], []),
            0,
            False,
        )
