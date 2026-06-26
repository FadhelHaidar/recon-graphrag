"""Tests for the evaluation baseline runner."""

from __future__ import annotations

import json

import pytest

from evaluation.runner import (
    FakeLLM,
    _hash_file,
    _load_jsonl,
    _make_fake_communities,
    run_baseline,
)
from evaluation.schemas import RunManifest, SearchConfigSnapshot


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"document_id": "d1", "text": "hello world", "metadata": {}})
        + "\n"
        + json.dumps({"document_id": "d2", "text": "goodbye world", "metadata": {}})
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def questions_path(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "question": "hello?"}) + "\n"
        + json.dumps({"question_id": "q2", "question": "world?"}) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "runs"


@pytest.mark.asyncio
async def test_run_baseline_fake_creates_artifacts(corpus_path, questions_path, output_dir):
    manifest = await run_baseline(
        corpus_path=corpus_path,
        questions_path=questions_path,
        output_dir=output_dir,
        search_config=SearchConfigSnapshot(top_k=2, level=0),
    )

    assert isinstance(manifest, RunManifest)
    assert manifest.corpus_id == "corpus"
    assert manifest.question_set_id == "questions"
    assert manifest.package_version is not None
    assert manifest.git_sha is not None

    run_dir = output_dir / manifest.run_id
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "results.jsonl").exists()


@pytest.mark.asyncio
async def test_run_baseline_results_one_per_question(corpus_path, questions_path, output_dir):
    manifest = await run_baseline(
        corpus_path=corpus_path,
        questions_path=questions_path,
        output_dir=output_dir,
        search_config=SearchConfigSnapshot(top_k=2, level=0),
    )

    lines = (output_dir / manifest.run_id / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "question_id" in parsed
        assert "answer" in parsed
        assert "elapsed_seconds" in parsed


@pytest.mark.asyncio
async def test_run_baseline_retrieved_contexts_populated(corpus_path, questions_path, output_dir):
    manifest = await run_baseline(
        corpus_path=corpus_path,
        questions_path=questions_path,
        output_dir=output_dir,
        search_config=SearchConfigSnapshot(top_k=2, level=0),
    )

    lines = (output_dir / manifest.run_id / "results.jsonl").read_text().strip().splitlines()
    result = json.loads(lines[0])
    assert len(result["retrieved_contexts"]) == 2
    assert result["retrieved_contexts"][0]["community_id"] == "comm:d1"


@pytest.mark.asyncio
async def test_fake_llm_is_deterministic():
    llm = FakeLLM()
    r1 = await llm.ainvoke("prompt")
    r2 = await llm.ainvoke("prompt")
    assert r1.content == r2.content
    assert r1.usage is not None
    assert r1.usage.total_tokens == 0


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    items = _load_jsonl(path)
    assert items == [{"a": 1}, {"b": 2}]


def test_hash_file_stable(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("hello", encoding="utf-8")
    assert _hash_file(path) == _hash_file(path)


def test_make_fake_communities_from_corpus():
    corpus = [
        {"document_id": "d1", "text": "first document"},
        {"document_id": "d2", "text": "second document"},
    ]
    communities = _make_fake_communities(corpus)
    assert len(communities) == 2
    assert communities[0]["id"] == "comm:d1"
    assert communities[0]["level"] == 0
