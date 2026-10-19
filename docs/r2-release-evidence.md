# R2 release evidence and remaining gates

**Status: draft, not release-ready.** Native schemas and evidence admission are
structural safeguards, not proof that an answer is factually supported.

## Scope and compatibility

R2 preserves directed facts, authorizes references only from packed community
evidence, carries citations only from admitted DRIFT evidence/actions, and checks
complete global reduction prompts against the configured token budget. Global
search still dispatches through the custom `_reduce_phase` hook. Delegating to the
base reducer after changing its admitted inputs fails closed rather than sending
a stale prepared prompt.

DRIFT primer/action/repair calls use native JSON schemas. Custom LLM integrations
must honor these per-call parameters; rejected schema calls are not retried as
prompt-only JSON. Mixed actions retain retrieval-managed citations instead of
requesting redundant generated references. Prose phases are unchanged. Existing
bounded repair and explicit fallback traces remain; fallback is not a successful
DRIFT traversal. See [search documentation](06-search.md).

No re-ingestion or graph migration is required. Previously stored reports are not
automatically repaired; explicit regeneration is separate from upgrading code.

## Candidate validation

- Python 3.11.16: **1,208 passed, 24 integration tests deselected**.
- Python 3.14.7: **1,208 passed, 24 integration tests deselected**.
- `uv lock --check`, source/wheel build, and `twine check`: passed.
- Fresh dispatch review: no findings; override, delegation, mutation rejection,
  stateful-counter, and concurrency regressions are covered.
- Live deterministic local/global/DRIFT workflow: **2 passed** on Neo4j and
  Memgraph in 5.31 seconds at candidate
  `fc01900b3f1750f8355e7a98d2914de5448484dc`. These use fake AI and cannot measure
  answer accuracy. See [test operations](10-testing.md).

These counts describe the focused candidate, which intentionally excludes
unrelated staged cleanup and environment-contract tests in the development tree.
Fresh combined review of the selected candidate and deterministic gate found no
issues. GitHub CI must pass before marking the PR ready; live gates remain open.

## Historical configured-model evidence

An earlier synthetic E2E completed on both Neo4j and Memgraph: **2 passed** in
467.30 seconds. DRIFT recorded respectively 16/14 completed actions including the
primer, **zero failed actions, zero repairs, and no fallback**. Calls were 18/16
within the 20-call limit. Finalizers reported zero remaining nodes and vector
indexes had 768 dimensions. Separately, live directed-fact regressions passed on
both stores. These are historical results, not a fresh run of this exact committed
candidate; historical cleanup does not establish current database contents.

The original temporary log was lost after an interrupted session. Six answers
and report excerpts were recovered from retained transcripts. Their precise
historical model/source identity is not independently bound to an immutable
candidate commit. Do not present them as a controlled release comparison.

## Manual support review

A small review of those six answers against the original three-document corpus
classified deduplicated propositions as supported, unsupported, or explicitly
qualified inference. Unsupported means not established by the corpus, not
necessarily false. Generated reports and citation IDs are not independent proof.

| Backend / mode | Supported | Unsupported | Qualified inference |
|---|---:|---:|---:|
| Neo4j local | 5 | 1 | 0 |
| Neo4j global | 5 | 0 | 0 |
| Neo4j DRIFT | 1 | 4 | 0 |
| Memgraph local | 6 | 1 | 0 |
| Memgraph global | 6 | 1 | 0 |
| Memgraph DRIFT | 3 | 4 | 1 |

For example, the corpus says Payments API depends on Identity Gateway and that
the incident affected Payments API. It does **not** establish that the compromise
originated at or through Identity Gateway, as one DRIFT answer suggested. It also
does not establish Gateway authentication/authorization implementation or that
Bob was the *primary* investigator. Corpus:
`tests/integration/synthetic_e2e_support.py:20-47`.

This is a three-document smoke review, not a statistical accuracy benchmark or a
recall score. No improvement over R1 is claimed.

## Gates still open

The database connection failure was resolved. Fresh preflight verified **both
stores empty**, with only known indexes and 768-dimensional vectors. Live
workflow tests passed, and all six per-test postflights found both stores globally
empty with unchanged vector definitions.

A fresh exact archive of published R1 `v0.10.1`
(`46db7461e23c2d0c72b7441609a499634caa4b9d`) and candidate
`fc01900b3f1750f8355e7a98d2914de5448484dc` each ran both original synthetic E2Es
without modifying tests or assertions. Corpus, test, and factory files were
identical across revisions; both used Python 3.11.16 and the same dependencies and
configured provider profile. Git-normalized source identities were verified
before/after execution, with raw hashes retained for CRLF-normalized files.

| Revision | Real-provider E2E result | Elapsed |
|---|---|---:|
| Published R1 | **2 failed** | 35.28 s |
| Exact candidate | **2 failed** | 34.89 s |

All four tests failed at the initial embedding dimension probe with
`openai.APITimeoutError: Request timed out.`, caused by
`httpcore.ConnectTimeout`. No LLM request, graph build, search, or E2E index
recreation was reached. The runs retained the SDK's default retries; no alternate
endpoint/profile or schema-less retry was used. Answers, citations, DRIFT traces,
and release-over-release support comparisons are **N/A**, not zero or passing.
The earlier interrupted R1 run still has no surviving completion evidence.

Before release:

1. Restore embedding-provider connectivity. Repeat both-store safety preflight
   before fresh writes; the observed empty state is not a future guarantee.
   Never recreate default indexes on populated/shared data.
2. Rerun original published-R1 and exact-candidate E2Es using the same corpus and
   profile. Bind model settings and source revisions to the evidence; review
   factual support and resolve any regression honestly.
3. Keep independent review and PR CI/build green. Do not merge or publish while
   the real-provider/comparison gates remain unresolved. Release through the
   existing Release Please process only after explicit approval; do not start R3.
