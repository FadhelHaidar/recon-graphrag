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
requesting redundant generated references. Other phases retain their existing
output formats. Existing
bounded repair and explicit fallback traces remain; fallback is not a successful
DRIFT traversal. See [search documentation](06-search.md).

No re-ingestion or graph migration is required. Previously stored reports are not
automatically repaired; explicit regeneration is separate from upgrading code.

## Candidate validation

- Pre-prompt-mitigation candidate, Python 3.11.16 and Python 3.14.7:
  **1,208 passed, 24 integration tests deselected** on each interpreter.
- Current prompt-mitigation tree, Python 3.14.7:
  **1,213 passed, 24 integration tests deselected**. A fresh code review found no
  issues; prompt-contract tests verify instruction delivery, not factuality.
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

## Targeted prompt-fidelity mitigation and fixed-input comparison

Investigation traced the Gateway claim to an unsupported location in generated
report prose, then map text, then a reducer that strengthened occurrence into
origin. Typed citations authorized evidence identifiers; they did not establish
that every prose claim was supported. The report was generated before search, so
this was not solely a reducer issue.

The default report, global map, and global reduce prompts now explicitly require
supported facts, unchanged relationship/attribution meaning, preserved uncertainty,
and no invented details or padding of sparse evidence. This is a prompt-only
mitigation: no new validator, schema, runtime logic, retry, dependency, or model
call was added. Plain report prefixes inherit the default body; full custom
report/map/reduce templates remain replacements. Explicitly opted-in, labeled
general knowledge remains permitted. Stored reports are **not** repaired; using
new report-generation instructions requires explicit regeneration.

Run `fixed-input-20261020T115748550809Z` compared the old literals at
`0b01095df284edff8c4a5581b604186dd3166340` with the new literals on identical
inputs. Three scenarios exercised report, map, and reduce independently; a fourth
reduce-only case used the actual retained erroneous map partial. No stage output
was fed into another stage. The configured `qwen3.8-27b` profile used thinking off,
temperature 0, seed 20261018, alternating old/new order, and one response per
condition. There were **10 pairs / 20 requests**, with SDK retries disabled:
all 20 responses ended normally, six reports retained nonempty parsed findings,
six maps parsed, and eight reductions were prose. No repair, rerun, database,
embedding, or full E2E call was performed. Source/fixture hashes and privacy checks
passed. These format/integrity results are not semantic validation.

A fresh independent reviewer examined every complete output against the supplied
facts. Coverage below counts required supported propositions, not all output
claims or overall accuracy. Naming a system without its stated relationship does
not preserve that relationship.

| Fixed-input scenario / stage | Old required coverage | New required coverage | Observed change |
|---|---:|---:|---|
| Incident / report | 5/5 | 5/5 | Less embellishment; unsupported criticality remains in the new rating rationale. |
| Incident / map | 1/5 | 1/5 | New explicitly leaves origin unknown, but trades detection for the operator fact; both omit other required relationships. |
| Incident / reduce | 5/5 | 5/5 | New leaves origin unspecified rather than reframing the login description as origin. |
| Sparse roles / report | 3/3 | 3/3 | New removes invented data-integrity and service-detail implications. |
| Sparse roles / map | 3/3 | 3/3 | Both preserve relationships and leave ownership/development/services unspecified. |
| Sparse roles / reduce | 3/3 | 3/3 | New removes explicit owner/operator conflation; its combined heading remains ambiguous and ownership/development unknowns are not explicitly answered. |
| Attribution / report | 3/3 | 3/3 | New removes invented resolution, but still upgrades investigator participation to leadership. |
| Attribution / map | 3/3 | 3/3 | Both preserve unnamed discoverer and unconfirmed causality. |
| Attribution / reduce | 3/3 | 3/3 | Both preserve attribution and uncertainty. |
| Retained erroneous map / reduce | 4/4 | 4/4 | Old strengthens Gateway occurrence into origin; new repeats occurrence without strengthening it. Coverage is relative to the imperfect partial, not original-corpus truth. |

The decisive boundary input says **"Identity Gateway (where the incident
occurred)"**. Old output says **"The system where the incident originated"**;
new says **"The location where the incident occurred"**. The upstream false
location survives in new. This demonstrates less amplification in this sample,
**not** correction of bad reports. Another residual new report says the
investigation was **"led by Omar"**, although only participation is established.
The independent reviewer recommended retaining the mitigation with those limits,
not speculative additional prompt expansion.

Full prompts, raw/parsed outputs, fixture definitions, hashes, execution manifest,
and the independent claim/coverage review are retained with the sanitized run
artifacts. This is one small fixed-input/model-profile check, not a statistical
benchmark, end-to-end rerun, factuality guarantee, or general accuracy claim.
Temperature 0 and a fixed seed do not attest deterministic server behavior.
Longer fixed prompts also leave less room for evidence under existing budgets:
approximate literal token estimates increase report 370→481, map 257→316, and
reduce 110→175; these are not full-request or provider billing counts.

## Gates still open

The real-provider execution and independent support-review gates are complete for
the earlier paired revisions. The prompt-only follow-up has local regression,
code-review, fixed-input provider, and independent semantic-review evidence, but
has not been rerun through regenerated reports and the full E2E pipeline. Its
results favor retaining the mitigation, not declaring the semantic quality issue
solved. Residual unsupported facts, incomplete answers, and surviving upstream
errors remain documented. Keep the PR draft until the owner explicitly accepts
the limitations or chooses further bounded work; do not silently waive them.
Keep PR CI/build green. Release through the existing Release Please process only
after explicit approval; do not merge, publish, or start R3 automatically.
