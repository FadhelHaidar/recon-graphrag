"""Claim validation and strict-output regressions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.parser import ClaimParser


def _claim(**changes):
    return {
        "subject_entity_id": "alice", "claim_type": "role",
        "description": "Alice leads Acme.", "status": "active",
        "object_entity_id": None, "start_date": None, "end_date": None,
        "source_text": None, "text_unit_id": None, **changes,
    }


@pytest.mark.parametrize("description", [None, {}, [], 42, False, "", "   "])
def test_parser_does_not_invent_claim_descriptions(description):
    assert ClaimParser().parse(json.dumps([_claim(description=description)])) == []


@pytest.mark.asyncio
async def test_claim_request_uses_strict_schema():
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"claims": [_claim()]}), finish_reason="stop",
    )))
    result = await LLMGraphExtractor(llm).extract_claims("source", ["alice"])
    assert len(result) == 1
    fmt = llm.ainvoke.call_args.kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["claims"]["items"]
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["description"]["type"] == "string"


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["[]", "{}", '{"claims": null}', '{"claims": {}}',
    '{"claims": [], "extra": 1}', json.dumps({"claims": [_claim(description=None)]}),
    json.dumps({"claims": [{"subject_entity_id": "alice"}]}), "not JSON"])
async def test_claim_extractor_rejects_invalid_envelope_without_fallback(content):
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=content)))
    with pytest.raises(ValueError):
        await LLMGraphExtractor(llm).extract_claims("source", ["alice"])
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "content_filter", "refusal"])
async def test_claim_extractor_rejects_incomplete_completion(reason):
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content='{"claims": []}', finish_reason=reason,
    )))
    with pytest.raises(ValueError):
        await LLMGraphExtractor(llm).extract_claims("source", ["alice"])


@pytest.mark.asyncio
async def test_claim_extractor_allows_empty_claims_and_filters_unknown_references():
    llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        SimpleNamespace(content='{"claims": []}'),
        SimpleNamespace(content=json.dumps({"claims": [_claim(object_entity_id="unknown")]})),
    ]))
    extractor = LLMGraphExtractor(llm)
    assert await extractor.extract_claims("source", ["alice"]) == []
    assert await extractor.extract_claims("source", ["alice"]) == []


@pytest.mark.asyncio
async def test_unsupported_claim_adapter_fails_without_provider_call():
    llm = SimpleNamespace(supports_structured_output=False, ainvoke=AsyncMock())
    with pytest.raises(ValueError, match="structured output"):
        await LLMGraphExtractor(llm).extract_claims("source", ["alice"])
    llm.ainvoke.assert_not_called()
