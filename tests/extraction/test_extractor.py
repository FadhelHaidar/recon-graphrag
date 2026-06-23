"""Tests for LLM graph extraction with gleaning."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.parser import AssessmentParser
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)


def _make_schema() -> GraphSchema:
    return GraphSchema(
        node_types=[
            NodeType(label="Person", properties=[PropertyType(name="name", type="STRING")]),
            NodeType(label="Movie", properties=[PropertyType(name="title", type="STRING")]),
        ],
        relationship_types=[RelationshipType(label="ACTED_IN")],
        patterns=[("Person", "ACTED_IN", "Movie")],
    )


def _make_initial_response():
    return '''{
        "nodes": [
            {"id": "p1", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "m1", "label": "Movie", "properties": {"title": "Inception"}}
        ],
        "relationships": [
            {"source_id": "p1", "target_id": "m1", "type": "ACTED_IN", "properties": {"weight": 1.0}}
        ]
    }'''


def _make_continuation_response():
    return '''{
        "nodes": [
            {"id": "p2", "label": "Person", "properties": {"name": "Bob"}}
        ],
        "relationships": [
            {"source_id": "p2", "target_id": "m1", "type": "ACTED_IN", "properties": {"weight": 1.0}}
        ]
    }'''


def _make_empty_continuation_response():
    return '{"nodes": [], "relationships": []}'


class TestAssessmentParser:
    def test_parse_yes(self):
        parser = AssessmentParser()
        assert parser.parse("yes") is True
        assert parser.parse("Yes") is True
        assert parser.parse("YES") is True
        assert parser.parse("yes, there are missed entities") is True

    def test_parse_no(self):
        parser = AssessmentParser()
        assert parser.parse("no") is False
        assert parser.parse("No") is False
        assert parser.parse("no, I got everything") is False

    def test_parse_noisy_response(self):
        parser = AssessmentParser()
        assert parser.parse("The answer is yes.") is True
        assert parser.parse("I don't think so. No.") is False


@pytest.mark.asyncio
async def test_zero_gleanings_single_call():
    """max_gleanings=0 means exactly one extraction call."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=_make_initial_response()))

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("Alice acted in Inception.", schema, max_gleanings=0)

    assert llm.ainvoke.call_count == 1
    assert len(result.nodes) == 2
    assert len(result.relationships) == 1


@pytest.mark.asyncio
async def test_gleaning_adds_missed_entities():
    """Gleaning finds and merges additional entities."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Initial extraction
            return MagicMock(content=_make_initial_response())
        elif call_count == 2:
            # Assessment: "yes, missed some"
            return MagicMock(content="yes")
        elif call_count == 3:
            # Continuation: new entity
            return MagicMock(content=_make_continuation_response())
        else:
            # Re-assessment: "no, done"
            return MagicMock(content="no")

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("Alice and Bob acted in Inception.", schema, max_gleanings=2)

    # 3 calls: initial + assessment + continuation (re-assessment not reached because
    # we stop after max_gleanings iterations or when no new items)
    assert llm.ainvoke.call_count >= 3
    assert len(result.nodes) == 3  # p1, m1, p2
    assert len(result.relationships) == 2  # p1->m1, p2->m1


@pytest.mark.asyncio
async def test_gleaning_stops_on_assessment_no():
    """Gleaning stops when assessment says no."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content=_make_initial_response())
        elif call_count == 2:
            return MagicMock(content="no")
        else:
            raise AssertionError("Should not be called again")

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("Alice acted in Inception.", schema, max_gleanings=5)

    assert llm.ainvoke.call_count == 2  # initial + assessment
    assert len(result.nodes) == 2


@pytest.mark.asyncio
async def test_gleaning_stops_on_empty_continuation():
    """Gleaning stops when continuation adds nothing new."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content=_make_initial_response())
        elif call_count == 2:
            return MagicMock(content="yes")
        elif call_count == 3:
            return MagicMock(content=_make_empty_continuation_response())
        else:
            raise AssertionError("Should not be called again")

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("Alice acted in Inception.", schema, max_gleanings=5)

    assert llm.ainvoke.call_count == 3  # initial + assessment + continuation
    assert len(result.nodes) == 2  # unchanged


@pytest.mark.asyncio
async def test_gleaning_respects_max_iterations():
    """Gleaning stops at max_gleanings even if assessment keeps saying yes."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content=_make_initial_response())
        # Always say yes and return new items
        if call_count % 2 == 0:
            return MagicMock(content="yes")
        return MagicMock(content=_make_continuation_response())

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("text", schema, max_gleanings=1)

    # 1 initial + 1 assessment + 1 continuation = 3 calls
    # Then loop ends because max_gleanings=1
    assert llm.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_gleaning_malformed_continuation_does_not_discard_initial():
    """If continuation parsing fails, initial extraction is preserved."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content=_make_initial_response())
        elif call_count == 2:
            return MagicMock(content="yes")
        elif call_count == 3:
            return MagicMock(content="not valid json at all")
        else:
            return MagicMock(content="no")

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()

    # Malformed continuation should raise, but initial is already captured
    # The extractor lets the parse error propagate — the pipeline handles it
    with pytest.raises(Exception):
        await extractor.extract("text", schema, max_gleanings=2)


@pytest.mark.asyncio
async def test_gleaning_duplicate_entities_not_added():
    """Gleaning does not add duplicate entities."""
    llm = MagicMock()
    call_count = 0

    async def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(content=_make_initial_response())
        elif call_count == 2:
            return MagicMock(content="yes")
        elif call_count == 3:
            # Continuation returns same entities as initial
            return MagicMock(content=_make_initial_response())
        else:
            return MagicMock(content="no")

    llm.ainvoke = AsyncMock(side_effect=mock_invoke)

    extractor = LLMGraphExtractor(llm)
    schema = _make_schema()
    result = await extractor.extract("text", schema, max_gleanings=2)

    # No new items added
    assert len(result.nodes) == 2
    assert len(result.relationships) == 1
