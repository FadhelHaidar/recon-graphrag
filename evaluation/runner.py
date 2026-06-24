"""Evaluation runner for capturing baseline GraphRAG runs.

The runner supports two modes:

- ``--fake``: deterministic fake LLM/graph-store run for CI. No external
  services are called.
- real (default): uses configured LLM and a provided graph store. This
  mode is opt-in and documented in ``evaluation/README.md``.

Each run writes a ``manifest.json`` plus ``results.jsonl`` under the output
 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from recon_graphrag._version import __version__, get_git_sha
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM, LLMResponse, LLMUsage
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever

from .schemas import (
    PerQuestionResult,
    PipelineConfigSnapshot,
    RetrievedContext,
    RunManifest,
    SearchConfigSnapshot,
    TokenUsage,
)


class FakeLLM:
    """Deterministic fake LLM for CI baseline runs."""

    def __init__(self, response_template: str = "Answer for prompt {prompt_hash}"):
        self.response_template = response_template
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse:
        self.calls.append(prompt)
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        content = self.response_template.format(prompt_hash=h)
        return LLMResponse(
            content=content,
            usage=LLMUsage(request_tokens=0, response_tokens=0, total_tokens=0),
        )


class FakeGraphStore:
    """Minimal fake graph store that serves deterministic communities."""

    def __init__(self, communities: list[dict] | None = None):
        self._communities = communities or []
        self.calls: list[tuple[str, dict]] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.calls.append(("execute_query", {"query": query.strip(), "params": parameters or {}}))
        params = parameters or {}
        # Global search reads reports at a level
        if "Community" in query and "report_text" in query:
            level = params.get("level", 0)
            return [
                c for c in self._communities if c.get("level") == level
            ]
        return []

    def search_communities(self, index_name, query_vector, graph_name, top_k, level=None):
        self.calls.append(
            (
                "search_communities",
                {
                    "index_name": index_name,
                    "graph_name": graph_name,
                    "top_k": top_k,
                    "level": level,
                },
            )
        )
        results = [
            {
                "id": c["id"],
                "summary": c["summary"],
                "level": c.get("level", 0),
                "score": c.get("score", 0.5),
            }
            for c in self._communities
            if level is None or c.get("level") == level
        ]
        return results[:top_k]


CorpusItem = dict[str, str | dict]
QuestionItem = dict[str, str | list]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _corpus_id_and_hash(path: Path) -> tuple[str, str]:
    stem = path.stem
    return stem, _hash_file(path)


def _question_set_id(path: Path) -> str:
    return path.stem


def _make_fake_communities(corpus: list[CorpusItem]) -> list[dict]:
    communities: list[dict] = []
    for i, item in enumerate(corpus):
        text = item.get("text", "")
        doc_id = item.get("document_id", f"doc:{i}")
        communities.append(
            {
                "id": f"comm:{doc_id}",
                "summary": text[:500],
                "level": 0,
                "score": 0.9 - (i * 0.05),
            }
        )
    return communities


def _extract_token_usage(llm_response: LLMResponse | None) -> TokenUsage | None:
    if llm_response is None or llm_response.usage is None:
        return None
    return TokenUsage(
        request_tokens=llm_response.usage.request_tokens,
        response_tokens=llm_response.usage.response_tokens,
        total_tokens=llm_response.usage.total_tokens,
    )


def _utc_now() -> datetime:
    """Return current UTC datetime; isolated for test patching."""
    return datetime.now(timezone.utc)


async def run_baseline(
    corpus_path: Path,
    questions_path: Path,
    output_dir: Path,
    graph_store: GraphStore | None = None,
    llm: BaseLLM | None = None,
    pipeline_config: PipelineConfigSnapshot | None = None,
    search_config: SearchConfigSnapshot | None = None,
    model_identifiers: dict | None = None,
    prompt_versions: dict | None = None,
) -> RunManifest:
    """Run a baseline evaluation and write artifacts to ``output_dir``.

    If ``graph_store`` and ``llm`` are not provided, a fake deterministic run
    is performed.
    """
    corpus = _load_jsonl(corpus_path)
    questions = _load_jsonl(questions_path)

    corpus_id, corpus_hash = _corpus_id_and_hash(corpus_path)
    question_set_id = _question_set_id(questions_path)

    pipeline_config = pipeline_config or PipelineConfigSnapshot()
    search_config = search_config or SearchConfigSnapshot()
    model_identifiers = model_identifiers or {}
    prompt_versions = prompt_versions or {}

    fake_run = graph_store is None or llm is None
    if fake_run:
        graph_store = FakeGraphStore(_make_fake_communities(corpus))
        llm = FakeLLM()

    retriever = GlobalSearchRetriever(
        graph_store=graph_store,
        llm=llm,
    )

    run_id = uuid.uuid4().hex
    created_at = _utc_now()

    results: list[PerQuestionResult] = []
    for question_item in questions:
        qid = str(question_item.get("question_id", ""))
        question = str(question_item.get("question", ""))
        start = time.perf_counter()
        errors: list[str] = []
        answer = ""
        retrieved: list[RetrievedContext] = []
        token_usage: TokenUsage | None = None
        try:
            search_result = await retriever.search(
                question,
                level=search_config.level,
            )
            answer = search_result.answer
            # Extract retrieved contexts from the search diagnostics
            reports_used = search_result.metadata.get("reports_used", 0)
            if reports_used > 0:
                # Re-read reports for context capture
                raw_communities = graph_store.execute_query(
                    """
                    MATCH (c:Community {graph_name: $graph_name, level: $level})
                    WHERE coalesce(c.report_status, 'success') <> 'failed'
                      AND coalesce(c.report_text, c.summary, '') <> ''
                    RETURN c.id AS id,
                           c.level AS level,
                           coalesce(c.report_text, c.summary) AS summary
                    ORDER BY c.id
                    """,
                    {"graph_name": retriever.graph_name, "level": search_config.level},
                )
                retrieved = [
                    RetrievedContext(
                        community_id=str(c.get("id", "")),
                        level=int(c.get("level", 0)),
                        summary=str(c.get("summary", "")),
                        score=0.0,
                    )
                    for c in raw_communities[:search_config.top_k]
                ]
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
        elapsed = time.perf_counter() - start

        results.append(
            PerQuestionResult(
                question_id=qid,
                question=question,
                answer=answer,
                retrieved_contexts=retrieved,
                token_usage=token_usage,
                elapsed_seconds=elapsed,
                errors=errors,
                retry_count=0,
            )
        )

    manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        git_sha=get_git_sha(),
        package_version=__version__,
        corpus_id=corpus_id,
        corpus_hash=corpus_hash,
        question_set_id=question_set_id,
        pipeline_config=pipeline_config,
        search_config=search_config,
        model_identifiers=model_identifiers,
        prompt_versions=prompt_versions,
    )

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")

    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GraphRAG evaluation baseline.")
    parser.add_argument("--corpus", required=True, type=Path, help="Path to corpus.jsonl")
    parser.add_argument("--questions", required=True, type=Path, help="Path to questions.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use deterministic fake LLM/graph store for CI.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Global search top_k")
    parser.add_argument("--level", type=int, default=None, help="Community level")
    return parser


async def _async_main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    search_config = SearchConfigSnapshot(top_k=args.top_k, level=args.level)

    if args.fake:
        manifest = await run_baseline(
            corpus_path=args.corpus,
            questions_path=args.questions,
            output_dir=args.output,
            search_config=search_config,
        )
    else:
        raise NotImplementedError(
            "Real baseline runs require a configured graph store and LLM. "
            "Use --fake for deterministic CI runs or wire your providers here."
        )

    print(f"Wrote baseline run {manifest.run_id} to {args.output / manifest.run_id}")


def main():
    import asyncio

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
