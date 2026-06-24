# Evaluation Harness

This directory contains a lightweight evaluation harness for Recon-GraphRAG.
It is introduced in Phase 0 of the paper-alignment roadmap to capture baselines
before behavior changes land.

## Layout

```text
evaluation/
  schemas.py          Pydantic models for run artifacts
  runner.py           Baseline runner (fake and real-model paths)
  fixtures/           Small reference corpus and question set
  runs/               Generated run output (only .gitkeep is committed)
```

## Deterministic fake-model baseline (CI)

```bash
python -m evaluation.runner \
  --corpus evaluation/fixtures/corpus.jsonl \
  --questions evaluation/fixtures/questions.jsonl \
  --output evaluation/runs \
  --fake
```

This produces:

```text
evaluation/runs/<run_id>/
  manifest.json
  results.jsonl
```

The manifest records `git_sha`, `package_version`, `corpus_hash`, and
configuration. The JSONL file contains one row per question with stable sorted
keys.

## Real-model baseline (opt-in)

Real baseline runs require a configured graph store and LLM. Wire
your providers into `evaluation.runner.run_baseline(...)` and call it directly;
there is no required real-model command in CI.

## Output format

- `manifest.json`: `RunManifest` serialized with sorted keys and 2-space indent.
- `results.jsonl`: one `PerQuestionResult` per line, serialized with sorted keys.
- Missing provider token usage is recorded as `null`, not `0`.

## Adding fixtures

Add documents to `fixtures/corpus.jsonl` and questions to
`fixtures/questions.jsonl`. Each line must be valid JSON. Re-run the fake
baseline and commit the updated manifest structure if schema versions change.
