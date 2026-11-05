# DRIFT fake-only latency attribution

This is measurement infrastructure, **not a production latency benchmark** or an
optimization. No providers, environment files, or databases are used. The harness
reuses `MixedFakeGraphStore` from `tests/retrieval/test_drift.py` and
`ExpansionStore` from `tests/retrieval/test_graph_evidence.py`; it adds no corpus.
LLM and embedding delays are asynchronous sleeps; each synchronous DB method
sleeps on the thread actually invoking it (defaults: 300/100/50 ms respectively).
The default DRIFT configuration here is `max_rounds=2`, `actions_per_round=2`,
`action_concurrency=3`, with HyDE enabled. Bounded-round scheduling selects from a
shared pending pool, so equal numbers do not mean the same work as the historical
`max_depth`/`max_followups` settings recorded below. Token budgets and counters
retain SDK defaults. Tiktoken is warmed before measurement because mixed context
uses it for text truncation.

Run from the repository root:

```sh
uv run --frozen python tests/manual/benchmark_drift.py --runs 2 --json > /tmp/drift-baseline.json
uv run --frozen python tests/manual/benchmark_drift.py --runs 2 --concurrency 2 --workers 2 --shared --repair --graph --json > /tmp/drift-shared-baseline.json
uv run --frozen pytest tests/retrieval/test_drift_benchmark.py -m 'not integration'
```

`runs` repeats batches; `concurrency` is searches per batch; `workers` bounds the
shared default executor. `--shared` reuses one retriever/store across all searches;
without it each search gets its own instance. Action concurrency remains 3.
`--repair` makes the primer's score invalid and repairs that response exactly once;
it is keyed by prompt/phase, never invocation order. All follow-up questions are
prompt-digest-keyed, so concurrent scheduling cannot change the action tree.
`--graph` enables action and reduction graph evidence, including the previously
unmeasured `fetch_community_member_graph` read. `--llm-latency`,
`--embed-latency`, and `--db-latency` accept nonnegative finite seconds.

## Reading the output

- Search-level `phase_ms` is copied from the existing SDK trace, not reconstructed.
  It includes primer/traversal/reduce/total; the SDK uses integer milliseconds.
- `spans` (with `--json`) give start offset, duration, parent ID and thread for
  each search-local span. `counts` and `inclusive_span_ms` summarize by name.
  **These are nested inclusive spans, not additive wall-time components.** Parallel
  siblings overlap too. Do not sum context, DB, executor, citation and token spans
  into a total or subtract their sums to claim exclusive CPU time.
- `invoke.<phase>` covers SDK LLM invocation and token accounting; `llm.<phase>`
  covers only fake provider latency/work. HyDE, primer, action, primer repair and
  reduction are separate. Embeddings are individual `embed` spans.
- `executor.offload` includes queue, execution and async resumption overhead.
  `executor.queue` ends at worker entry and includes dispatch overhead (not just
  contention); `executor.execution` is worker residency, including nested DB,
  context, token and citation work. Direct synchronous work has no executor span.
- `db.<method>` includes the simulated delay and fake implementation. Calls have
  full argument digests, grouped by method within **each search**, with repeated
  calls counted after the first. Positional/keyword representations are preserved;
  this is not a cache recommendation or proof of safely interchangeable reads.
- Context building, primer formatting, JSON parsing/repair, citation helper
  resolution, packing, token count/truncation and graph construction are timed.
  Token-class names distinguish approximate counts from Tiktoken truncation/counts.
- `fingerprint` hashes sorted full prompts, action trace, answer, context, citations
  and graph payload, excluding timings/thread IDs and other runtime metadata.
  Full semantic input is emitted with `--json` for independent inspection. Prompt
  sorting removes call arrival ordering, not content differences. Different
  scenarios (repair or graph) intentionally have different fingerprints.
- `max_stall_ms` is the batch-wide maximum 5 ms heartbeat scheduling gap, including
  the scheduled interval; it is not per-search exclusive blocked time.

Patches use the standard library and restore even on failure. Run this harness in
its **own `asyncio.run` loop/process**: patches are process-global and its bounded
executor becomes that loop's default executor and is shut down at completion.
It is not a telemetry library for embedding in applications or parallel test loops.
Instrumentation overhead is included; no latency thresholds are tested.
On a failed or cancelled batch, sibling searches are cancelled and awaited before
patches are restored, and already-running executor work is drained. Draining can
block until a synchronous read returns; it is not a way to cancel that read.

Validation after review fixes: 11 focused benchmark tests, 243 retrieval and
compatibility tests, and the full non-integration suite (1603 passed, 1 skipped,
134 deselected). Active LSP checks reported no errors in either changed Python
file. These checks do not exercise live providers or databases.

## Historical baseline (old `max_depth=2` / `max_followups=2` scheduler)

Kept for comparison only. This section was recorded before the bounded-round
scheduler landed and must not be compared against new-config runs as a speedup:
the old scheduler executed 7 actions and 9-10 LLM calls, while the new
configuration executes 5 actions and 7-8 LLM calls.

Commands above, SDK HEAD `19fc50c`, local Python 3.12.14, 50/300/100 ms fake delays.
All searches: 7 actions, 8 embedding calls, no failures. All DB calls were observed
on executor threads. Numbers are illustrative host measurements, not promises.

| Scenario | Per-search total ms | primer / traversal / reduce ms | DB calls | LLM calls | repeated DB calls |
| --- | --- | --- | --- | --- | --- |
| isolated, run 1 | 3883.3 | 905 / 2575 / 401 | 58 | 9 | 43 |
| isolated, run 2 | 3882.6 | 907 / 2573 / 401 | 58 | 9 | 43 |
| shared + repair + graph, batch 1 | 4727.4; 5428.9 | 1206 / 3067 / 451; 1207 / 3769 / 451 | 59 each | 10 each | 43 each |
| shared + repair + graph, batch 2 | 4727.6; 5429.1 | 1204 / 3070 / 451; 1205 / 3771 / 452 | 59 each | 10 each | 43 each |

Isolated batch wall: 7767.3 ms, max heartbeat gap 7.1 ms. Shared batch wall:
10859.8 ms, max gap 6.2 ms. Example isolated search inclusive sums: DB 2908.7 ms,
embedding 807.2 ms, context.mixed 1809.7 ms, chunk citation helper 301.1 ms,
reference helper 703.0 ms, executor queue 8.2 ms; LLM HyDE 300.2 ms, primer
300.4 ms, actions 1808.7 ms, reduction 300.3 ms. Shared-search executor queue sums
were 804.9/2562.7 and 805.1/2567.9 ms, demonstrating contention attribution, not a
speedup claim. Repair added one ~300 ms LLM call; graph added one ~50 ms member read.

Stable isolated fingerprint (both runs):
`29adfb5c04a3cb35599a6deacc38fb2dc3b4cb0c79ff30cc4303d810b1ced8e2`.
Stable shared/repair/graph fingerprint (all four):
`4c11761027875c050b944831eb8929209b7f49b4e44c99c56863f1369e38d66d`.

Per isolated search, repeated argument signatures were: vector search 6 calls,
entity context 6, claims 6, three distinct execute-query groups of 6, citation
resolution 13, community-report vector search 2; six keyword queries were unique,
and one final reference execute-query was unique. Graph adds one unique member
read. Constant fake embeddings explain repeated vector arguments: this does not
establish production redundancy or justify caching/concurrent same-store reads.

Independent parent rerun: isolated totals 3874.7/3880.2 ms, with the same
9 LLM calls, 7 actions, 58 DB reads and isolated fingerprint above. A zero-delay
control (`--runs 3 --db-latency 0 --llm-latency 0 --embed-latency 0 --json`)
measured 22.2/14.6/13.1 ms with identical work/fingerprints. Those totals include
instrumentation, scheduling and fake implementations, not just production CPU.
They do not measure realistic large-context tokenization or provider overhead.

Not covered: live latency, backend thread safety, fallback paths, primer folds,
provider rate limits, CPU-heavy real responses, or production semantic quality.
No production optimization should be selected from these fake numbers alone.

## New-configuration control (zero fake latency)

Current SDK, new bounded-round configuration, `--runs 3 --db-latency 0
--llm-latency 0 --embed-latency 0`, workers default, isolated searches. This is a
structural/plumbing control, not a latency result.

| Property | Isolated | Shared + repair + graph |
| --- | --- | --- |
| actions (incl. primer) | 5 | 5 |
| LLM calls | 7 | 8 |
| DB calls | 40 | 41 |
| repeated DB calls | 27 | 27 |
| per-search total ms | 24.6 / 19.0 / 24.5 | 35.4 / 36.1 / 36.4 / 37.7 |
| batch max heartbeat gap ms | 9.5 | not recorded in this control |

Both scenarios keep identical work and fingerprints across repetitions under the
new scheduler: isolated `92989f480bc217103af3e7d012ccfc104c744bf88fdc3661662db6773434c915`,
shared/repair/graph `d6da3399af7d073df05bbc583ec2b5363a6d35da9abe45247c09a907d0bacb56`.
The 5 actions come from two rounds selecting two questions each, plus the primer.
Fingerprints differ from the historical ones because the scheduler selects
different questions; that is expected, not a regression signal.

## Proposed live validation — not authorized or run

The user requested a bounded plan, not permission to spend. Proposed envelope:

- Reuse the existing synthetic incident corpus and the Gateway → Payments query
  on the existing `driftbench` graphs. No extraction, rebuild, deletion, or writes.
  First verify which processes own ports 7688/7689 and match them to the requested
  Podman containers. If ownership or graph provenance is unclear, stop.
- With no production candidate yet, collect three baseline searches per backend
  (six total). Once a candidate passes deterministic equivalence checks, use
  three baseline/candidate pairs per backend (12 searches total), alternating
  order. Do not call baseline-only repetitions evidence of a speedup.
- Freeze models, provider options, prompts, graph contents, queries, configuration,
  citation metadata options and search budgets in a redacted run manifest. Keep
  all stages and stopping rules. Record actual LLM/repair/embedding calls, actions,
  DB calls, queue spans, provider usage, citations and answers alongside timings.
  Changing graph contents invalidates a pair. Generated paths may still differ;
  report those differences rather than treating unequal work as controlled proof.
- Proposed aggregate spend cap: **US$5**, subject to explicit approval. This is a
  cap proposal, not a cost estimate. Before approval, identify the unchanged models,
  verify current prices and their existing input/output/reasoning limits, include
  embeddings and retries, and compute a worst-case reservation for each search.
  If a bound cannot be established without changing model/search settings, do not
  run. Admit a new search only if its full reservation fits the remaining cap.
  Do not truncate an in-flight search to manufacture a cheaper comparable run.
- Request ceiling is the unchanged `max_llm_calls` times search count (at SDK
  default 20: 120 for baseline-only, 240 for paired runs). This excludes transport
  retries, which must be bounded and charged separately; also bound embedding
  requests from the frozen action budget. A cancelled await does not guarantee a
  provider call or worker read stopped. Keep outstanding work in the reservation.
- Report all repetitions, medians and ranges, plus answer/citation checks against
  the synthetic corpus. Three pairs are exploratory validation, not a statistical
  guarantee or a formal semantic-quality regression suite.

The fake harness is not a live runner. Live instrumentation and the final numeric
request/token/cost reservations still need review before requesting spend approval.
