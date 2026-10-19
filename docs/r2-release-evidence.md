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

## Historical manual support review (not the paired run)

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

## Fresh published-R1 / candidate comparison

Database connectivity and VPN access to the configured providers were restored.
The earlier embedding connection timeouts were failed attempts, not E2E passes.
A subsequent fresh serial comparison completed both original synthetic E2Es on
both revisions, without modifying tests, factories, assertions, or model profile.

- Published R1 `v0.10.1`: `46db7461e23c2d0c72b7441609a499634caa4b9d`.
- Candidate: `9d3fd26017322dbcf4323bb5480d31a19dd5891e`; only evidence
  documentation changed from the previously tested implementation at `fc01900`.
- Original corpus, synthetic test, factory, and support files were byte-identical.
  Revision-local imports and exact Git-normalized source manifests were verified
  before/after; raw hashes were retained for CRLF-normalized files.
- Same Python 3.11.16/dependencies, configured `qwen3.8-27b` LLM with thinking
  disabled, and `jina-omni-nano-retrieval` embeddings. Successful embedding probes
  returned 768 dimensions. Server-side model alias revisions were not attested.

| Revision | Real-provider E2E result | Pytest elapsed |
|---|---|---:|
| Published R1 | **2 passed** | 469.80 s |
| Exact candidate | **2 passed** | 413.78 s |

| Revision / backend | Completed actions including primer | Failed | Repairs | Calls / limit | Fallback |
|---|---:|---:|---:|---:|---|
| R1 / Neo4j | 16 | 0 | 1 | 19 / 20 | No |
| R1 / Memgraph | 12 | 0 | 0 | 14 / 20 | No |
| Candidate / Neo4j | 10 | 0 | 0 | 12 / 20 | No |
| Candidate / Memgraph | 10 | 0 | 0 | 12 / 20 | No |

All four traces stopped with `all_actions_completed`. Twelve final answer objects
and full exposed citations/context/DRIFT metadata were captured. These are single
stochastic smoke runs, not evidence that fewer actions, fewer repairs, or shorter
elapsed time improve accuracy or reliably improve speed.

Fresh both-store preflight found empty stores and only known 768-dimensional
indexes before allowing the original fixtures to recreate default retrieval
indexes. Four independent postflights checked both stores after each test:
zero nodes/relationships and expected vector definitions. Only exact fixture
scopes were deleted; no global delete or endpoint/profile substitution occurred.

An earlier VPN attempt was invalidated by an external observer's dataclass
serialization error; a subsequent launcher import error occurred before paid
calls. Neither attempt is counted above. After offline import/serialization
self-checks and fresh safety checks, the corrected external recorder captured the
fresh comparison without changing original execution or assertions. The executor
later timed out at its 30-minute agent limit, after saving completed subprocess
outcomes, all answers, integrity records, and postflights; those saved records
were inspected separately rather than rerunning successful tests.

## Independent support review of the fresh paired run

All twelve final answers were independently reviewed against the original corpus,
not generated graph facts, reports, confidence scores, or citation identifiers.
Each answer's deduplicated semantic propositions were classified as supported
(explicit or entailed), unsupported (not established, not necessarily false), or
qualified inference (reasonable possibility, not an established event). Repeated
facts count once per answer; tightly coupled same-judgment details are grouped.

| Backend / mode | R1 supported / unsupported / qualified | Candidate supported / unsupported / qualified |
|---|---:|---:|
| Neo4j local | 4 / 1 / 0 | 3 / 1 / 0 |
| Memgraph local | 4 / 1 / 0 | 3 / 0 / 0 |
| Neo4j global | 4 / 0 / 0 | 10 / 0 / 0 |
| Memgraph global | 4 / 0 / 0 | 5 / 1 / 0 |
| Neo4j DRIFT | 4 / 3 / 0 | 4 / 1 / 1 |
| Memgraph DRIFT | 1 / 2 / 0 | 1 / 2 / 1 |
| **Total** | **21 / 7 / 0** | **26 / 5 / 2** |

These are manual inventories, not accuracy/recall scores. Answer lengths and
stochastically generated graphs differ (Neo4j entities/relationships: R1 9/16,
candidate 10/18; Memgraph: R1 9/18, candidate 9/20). No statistical gain or causal
implementation regression is established by this three-document sample.

A concrete observed deterioration remains: candidate Memgraph global says
**"Identity Gateway: The specific system where the incident originated."** Its
paired R1 answer does not assert origin. The corpus only establishes Payments'
dependency on Gateway and that the incident affected Payments; it does not locate
the incident in Gateway. Other unsupported additions include Gateway
technical responsibilities, guaranteed propagation in R1, and attributing
root-cause discovery specifically to Bob in candidate Neo4j local. All local
answers correctly name Bob; all global answers identify the organization and
three systems. Neo4j DRIFT answers acknowledge unspecified interface details;
Memgraph DRIFT answers do not clearly state that limitation.

The review found coherent execution evidence and no demonstrated blocking
code-level regression, while recording these factual-support limitations. Valid
schemas and citations do not prove an answer's claims. The claim ledger and
reviewed answer excerpts are retained with the sanitized run artifacts.

## Gates still open

The real-provider execution and independent support-review gates are complete.
The quality assessment is mixed, not an accuracy-improvement claim: the observed
unsupported Gateway-origin addition remains unresolved. Keep the PR draft until
the owner decides whether to investigate it further or explicitly accept the
recorded semantic limitations; do not silently waive the quality concern. Keep
PR CI/build green. Release through the existing Release Please process only after
explicit approval; do not merge, publish, or start R3 automatically.
