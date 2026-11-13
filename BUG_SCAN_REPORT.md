# Read-only SDK bug scan

## Snapshot and result

- Worktree: `/home/fadhelh/projects/recon-graphrag-bugscan`.
- Base: `660f0ee4e4b05cbf62e05c4d318ce5c731a811fe` (`660f0ee`).
- Scope: `recon_graphrag/` and connected tests, with writer correctness reviewed first.
- Instructions read: `AGENTS.md` and `CONTRIBUTING.md`.
- Result: **seven findings: three High and four Medium**.
- Classification: **all seven findings preexist in HEAD**.
- No verified new bug was found in the uncommitted relationship-read predicate specialization.
- This result does not establish that the snapshot has no other bugs.

High means silent data loss or unintended graph mutation under the stated trigger.
Medium means conditional ingestion failure, invalid provenance, or incorrect retrieval ranking.
Severity depends on the stated prerequisites, not hypothetical remote access.

No SDK code, tests, commits, or existing snapshot changes were modified.
No agents, live databases, or providers were launched.
No credentials or `.env` contents were read.
This report is the only retained worktree output from the scan.

## Findings

| ID | Severity | Finding | Current location | Classification | Evidence |
| --- | --- | --- | --- | --- | --- |
| B01 | High | Record graph scopes can differ from the document scope | `recon_graphrag/pipelines/writer.py:473-474` | Preexisting | Deterministic failure and source |
| B02 | High | Relationship MERGE overwrites parallel matches | `recon_graphrag/pipelines/writer.py:705-716` | Preexisting | Captured query and Cypher semantics; no live execution |
| B03 | High | Replacement deletes a retained claim after relocating its source | `recon_graphrag/pipelines/writer.py:955-967` | Preexisting | Deterministic failure |
| B04 | Medium | Domain-label MERGE cannot reuse an unlabeled stored entity | `recon_graphrag/pipelines/writer.py:624-625` | Preexisting | Captured query and constraint source; no live execution |
| B05 | Medium | Mixed null and string relationship IDs crash reconciliation | `recon_graphrag/pipelines/writer.py:980-996` | Preexisting | Deterministic failure |
| B06 | Medium | Validation accepts chunk references that replacement immediately removes | `recon_graphrag/pipelines/writer.py:266-279` | Preexisting | Deterministic failure and delete-query source |
| B07 | Medium | Replaced chunk text retains its old embedding | `recon_graphrag/pipelines/writer.py:584-589` | Preexisting | Deterministic failure and both backend selection queries |

### B01 — Record graph scopes can differ from the document scope

**Location:** `recon_graphrag/pipelines/writer.py:473-474`.
Related validation: `recon_graphrag/pipelines/writer.py:266-286`.

**Trigger:** Submit a public `GraphDocument` whose document uses `tenant-A`, but whose entity uses `tenant-B`.
This can occur when callers set the document graph but retain child records' default `graph_name`.
Use an entity without evidence links to isolate the mismatch.

**Mechanism:** State reads and reference validation use the document's graph.
Entity rows instead use `entity.graph_name`, without a consistency check.
The MERGE then writes those rows to their individual graph scopes.
If `tenant-B` already contains the entity, its contributions were not read before the update.

**Impact:** A document write can silently mutate another logical graph.
It can also replace entity provenance using state read from the wrong graph.
This is an SDK consistency failure, not a demonstrated database authorization bypass.
Both backend wrappers use this writer.

**Reproduction evidence:** `test_graph_mismatch_is_rejected_before_mutation` used the existing stateful fake store.
The document had graph `tenant-A`; its only entity had graph `tenant-B`.
The writer completed instead of rejecting the mismatch:

```text
Failed: DID NOT RAISE <class 'ValueError'>
```

The emitted entity row uses `tenant-B`, as specified by the cited row construction.
The standard assembler propagates one graph name consistently; custom `GraphDocument` callers expose this failure.

**Classification:** Preexisting; HEAD has the same row construction at `recon_graphrag/pipelines/writer.py:457-458`.

### B02 — Relationship MERGE overwrites parallel matches

**Location:** `recon_graphrag/pipelines/writer.py:705-716`.
Related state selection: `recon_graphrag/pipelines/writer.py:359-399`.

**Trigger:** Two stored relationships have the same type and endpoints, but different IDs or provenance.
Write one relationship for those endpoints.
For example, a new document's incoming ID matches only one stored relationship during the state read.
Legacy data, imports, or external writers can create this state.

**Mechanism:** The emitted write matches neither relationship ID nor relationship graph scope:

```cypher
MERGE (source)-[r:`KNOWS`]->(target)
SET r.id = row.id,
    r.graph_name = row.graph_name,
    r.source_chunk_ids = row.source_chunk_ids,
    r.description_sources = row.description_sources
```

Cypher MERGE passes each existing matching relationship to SET.
The preceding state read can select only one relationship by ID.
Consequently, the generated contribution row need not contain the other relationship's provenance.

**Impact:** All parallel matches receive the same ID and replacement provenance.
Other documents' relationship evidence can be lost.
An edge with a different or missing `r.graph_name` between those endpoints can also be reassigned.
The endpoint graph filters do not filter the relationship itself.
The query is shared by Neo4j and Memgraph.

**Reproduction:** In a disposable graph, create two `KNOWS` edges between graph-scoped entities `a` and `b`.
Assign IDs `r1` and `r2`, with distinct source chunks.
Write a new document containing relationship `r1`, with no previous chunks for that document.
The read can return only `r1`; the write matches both edges and applies the `r1` row to both.

**Evidence:** `test_relationship_merge_query_does_not_filter_relationship_identity` passed and captured the exact unrestricted MERGE shape.
The SDK's stateful fake keys relationships by graph, endpoints, and type.
It cannot represent this parallel-edge state.
This is source-verified Cypher behavior, not a backend execution result.
No live reproduction was attempted.

**Classification:** Preexisting; HEAD uses the same MERGE and SET at `recon_graphrag/pipelines/writer.py:689-700`.

### B03 — Replacement deletes a retained claim after relocating its source

**Location:** `recon_graphrag/pipelines/writer.py:955-967`.
Write order: `recon_graphrag/pipelines/writer.py:214-222`.

**Trigger:** Reingest a document with the same claim ID, move that claim to a new chunk, and remove its old chunk.

**Mechanism:** `_write_claims` updates the retained claim before `_reconcile` runs.
The stale-claim predicate uses the previous source chunk:

```python
if row["id"] not in keep_claim_ids
or row.get("chunk_id") in removed_chunk_id_set
```

The second condition includes a retained claim whose old chunk was removed.
Reconciliation then deletes that claim by ID, including the newly written version.

**Impact:** Successful replacement silently loses a submitted claim.
The write receipt can still report one claim written.
Transactions do not prevent this loss because the queries succeed and commit.
Both backend wrappers use this sequence.

**Reproduction evidence:** `test_retained_claim_id_survives_source_chunk_relocation` performed these writes:

1. Write document `doc:1`, chunk `old`, entity `a`, and claim `claim` sourced from `old`.
2. Replace it with chunk `new` and the same claim ID sourced from `new`.

The second receipt contained `claims == 1`.
The stateful fake then returned `None` for `get_claim('claim')`:

```text
AssertionError: assert None is not None
```

**Scope qualification:** Custom stable claim IDs expose this bug.
The standard assembler includes the chunk ID in generated claim IDs at `recon_graphrag/extraction/assembler.py:378-384`.
Thus, standard assembly ordinarily changes the claim ID when its chunk changes.

**Classification:** Preexisting; HEAD has the same predicate at `recon_graphrag/pipelines/writer.py:939-943`.

### B04 — Domain-label MERGE cannot reuse an unlabeled stored entity

**Location:** `recon_graphrag/pipelines/writer.py:624-625`.

**Trigger:** The graph already contains `(:__Entity__ {id: 'a', graph_name: 'g'})` without the incoming domain label.
Write `EntityRecord(id='a', type='Person', graph_name='g')`.
A changed domain label on an existing ID causes the same problem.

**Mechanism:** Entity state is read by SDK identity, but the MERGE additionally requires the domain label:

```cypher
MERGE (e:__Entity__:`Person` {id: row.id, graph_name: row.graph_name})
SET e.uid = row.uid
```

MERGE does not match the existing node without `Person`.
It attempts to create another node rather than add the missing label.

**Impact:** Without a UID uniqueness constraint, the write creates duplicate SDK identities.
Subsequent identity-only MATCH operations can fan out over those duplicates.
With a backfilled UID and the SDK constraint, setting the new node's UID violates uniqueness and aborts ingestion.
A retry repeats the same label-qualified mismatch.

Constraint evidence:

- Neo4j: `recon_graphrag/graphdb/neo4j/index_manager.py:204-212`.
- Memgraph: `recon_graphrag/graphdb/memgraph/index_manager.py:348-356`.

**Reproduction evidence:** `test_entity_merge_query_requires_domain_label` passed and captured the exact MERGE shown above.
For backend reproduction, seed the unlabeled node, then submit the matching typed entity.
Check duplicate identity count without constraints, or ingestion failure after UID backfill and constraint creation.
No live reproduction was attempted.
The stateful fake ignores label-qualified identity and cannot verify this behavior.

**Scope qualification:** The prerequisite is incomplete, changed, or externally stored labels.
A normal retry against an existing entity with the same domain label does not trigger this issue.

**Classification:** Preexisting; HEAD has the same MERGE at `recon_graphrag/pipelines/writer.py:608-609`.

### B05 — Mixed null and string relationship IDs crash reconciliation

**Location:** `recon_graphrag/pipelines/writer.py:996`.
Related row construction: `recon_graphrag/pipelines/writer.py:849-852`.

**Trigger:** Replacement removes the last contributions from multiple stored relationships.
At least one relationship lacks an ID property, and another has a string ID.
The state query returns the missing ID as `None`.

**Mechanism:** Rebuild rows preserve the stored ID.
The cleanup expression attempts to sort heterogeneous values:

```python
"rel_ids": sorted(set(empty_relationship_ids)),
```

**Impact:** Reconciliation raises `TypeError` instead of completing replacement.
Neo4j and Memgraph transaction wrappers roll back the document write.
A custom store without `_transaction` can retain earlier auto-committed mutations.
The error is not retryable under the writer's retry classifier.

**Reproduction evidence:** `test_mixed_null_and_string_relationship_ids_replacement` used a full writer call sequence.
The first write created `KNOWS` and `WORKS_WITH` edges sourced from chunk `old`.
The probe set one stored edge ID to `None`, then replaced the document with chunk `new` and no relationships.

```text
recon_graphrag/pipelines/writer.py:996: TypeError
TypeError: '<' not supported between instances of 'str' and 'NoneType'
```

**Scope qualification:** Normal `RelationshipRecord` writes supply string IDs.
The failing state requires legacy, imported, or otherwise incomplete stored relationships.

**Classification:** Preexisting; HEAD contains this expression at `recon_graphrag/pipelines/writer.py:980`.

### B06 — Validation accepts chunk references that replacement immediately removes

**Location:** `recon_graphrag/pipelines/writer.py:266-279`.
Deletion: `recon_graphrag/pipelines/writer.py:970-977`.

**Trigger:** A replacement references one of its document's previous chunks but omits that chunk from the new chunk list.
Use a newly submitted claim ID so stale-claim deletion does not mask the separate defect.

**Mechanism:** Validation adds every previous chunk to the valid-ID set:

```python
chunk_ids = {chunk.id for chunk in graph_document.chunks}
chunk_ids.update(state.chunk_ids)
```

The claim write resolves the old chunk successfully.
Reconciliation then detaches and deletes that chunk because it is absent from the new document.
The new claim was absent from `state.claims`, so stale-claim cleanup does not delete it.

**Impact:** A successful receipt can leave a Claim without its `SOURCED_FROM` edge.
Evidence links to removed chunks are also detached after their write.
This breaks the writer's reference-validation guarantee and loses citation provenance.
Both backends execute the same write and cleanup sequence.

**Reproduction evidence:** `test_references_to_chunks_removed_in_same_replacement_are_rejected` performed these writes:

1. Write document `doc:1` with chunk `old` and entity `a`.
2. Replace its chunks with `[new]`, while submitting new claim `new-claim` sourced from `old`.

The writer accepted the invalid replacement:

```text
Failed: DID NOT RAISE <class 'ValueError'>
```

The fake confirms acceptance; the cited `DETACH DELETE` establishes the backend edge-removal effect.
The fake stores claim endpoints as fields, not actual edges, so it does not independently prove the final backend topology.
Custom `GraphDocument` callers expose this state; normal assembly supplies current chunks.

**Classification:** Preexisting; HEAD has the same validation at `recon_graphrag/pipelines/writer.py:266-279`.

### B07 — Replaced chunk text retains its old embedding

**Location:** `recon_graphrag/pipelines/writer.py:584-589`.

**Trigger:** Reingest changed text under the same document identity and embedding model after chunk embedding has completed.
This occurs with an explicit `metadata['document_id']` or a stable `source` or `title`.

**Mechanism:** The chunker assigns IDs by document and position, not content.
The writer updates `c.text` but neither clears the vector nor records changed content for embedding selection.
Both backend selectors consider only a missing vector or a changed model fingerprint.

Connected locations:

- Stable document identity: `recon_graphrag/pipelines/graphrag_pipeline.py:1145-1152`.
- Position-based chunk identity: `recon_graphrag/extraction/chunking.py:85-89`.
- Neo4j selection: `recon_graphrag/graphdb/neo4j/store.py:532-536`.
- Memgraph selection: `recon_graphrag/graphdb/memgraph/store.py:552-556`.
- Embedding assumption: `recon_graphrag/embeddings/chunks.py:50-58`.
- Source ranking: `recon_graphrag/retrieval/_queries.py:25-43`.

**Impact:** Local and DRIFT source ranking can use a vector for previous text while returning the replacement text.
The chunk embedding stage does not select that chunk again while the model fingerprint remains unchanged.
No exception indicates the stale ranking data.

**Reproduction evidence:** `test_changed_chunk_text_invalidates_existing_vector` wrote chunk `doc:1:chunk:0` with old text.
The probe attached vector `[1.0, 0.0]` and model fingerprint `same-model`.
A second write changed the text under the same chunk ID.
The text changed, but the vector remained:

```text
AssertionError: assert [1.0, 0.0] is None
```

The stateful fake preserves untouched properties.
The writer query and both backend selectors independently support the stale-vector mechanism.
No real embedding provider was used.

**Classification:** Preexisting; HEAD updates text without invalidation at `recon_graphrag/pipelines/writer.py:568-573`.
Both backend selectors and the connected chunking code are unchanged from HEAD.

## Verification record

### Existing deterministic suite

The suite ran through `hc` terminal `t2k` from the isolated worktree.
It used the existing Python environment without dependency synchronization.
The configured `pythonpath` selected this worktree's source.
Bytecode and pytest cache writes were disabled.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
PYTHON_DOTENV_DISABLED=1 \
RUN_DATABASE_INTEGRATION_TESTS=0 \
RUN_PROVIDER_INTEGRATION_TESTS=0 \
RUN_WORKFLOW_INTEGRATION_TESTS=0 \
RUN_E2E_INTEGRATION_TESTS=0 \
RUN_ENTITY_RESOLUTION_AI_TESTS=0 \
UV_PROJECT_ENVIRONMENT=/home/fadhelh/projects/recon-graphrag/.venv \
/home/fadhelh/.local/bin/uv run --no-sync pytest -m 'not integration'
```

Result: **1768 passed, 1 skipped, 148 deselected in 20.41 seconds**; exit code 0.
The interpreter was Python 3.12.14; pytest was 9.0.3.
This includes deterministic backend, compatibility, extraction, pipeline, community, embedding, and retrieval tests.
The skip was the Memgraph parameter of a Neo4j-only locking-query test.
Its reason was `Neo4j-specific locking query`, at `tests/graphdb/test_vector_persistence.py:50`.

An initial launch in `hc` terminal `t2j` failed before test execution:

```text
sh: 1: uv: not found
```

Using the absolute `uv` path resolved that launcher issue.

### Focused diagnostic probes

Seven temporary pytest probes ran through `hc` terminal `t2m`.
The temporary test module was outside the worktree and was removed after execution.
No retained regression tests were added, as requested.
The run used the same safety environment and this command suffix:

```bash
/home/fadhelh/.local/bin/uv run --no-sync pytest -m 'not integration' \
  -p no:cacheprovider \
  -o pythonpath=/home/fadhelh/projects/recon-graphrag-bugscan \
  --import-mode=importlib -q /tmp/bugscan-probes-JFgCRa.py
```

Result: **5 failed, 2 passed in 0.43 seconds**; exit code 1.
The five failures are deliberately asserted correctness expectations, not failures in the existing suite.
The two passing probes establish query shapes only.

| Probe | Result | Evidence target |
| --- | --- | --- |
| `test_graph_mismatch_is_rejected_before_mutation` | Failed: no rejection | B01 |
| `test_relationship_merge_query_does_not_filter_relationship_identity` | Passed: query shape confirmed | B02 |
| `test_retained_claim_id_survives_source_chunk_relocation` | Failed: claim missing | B03 |
| `test_entity_merge_query_requires_domain_label` | Passed: query shape confirmed | B04 |
| `test_mixed_null_and_string_relationship_ids_replacement` | Failed: TypeError | B05 |
| `test_references_to_chunks_removed_in_same_replacement_are_rejected` | Failed: no rejection | B06 |
| `test_changed_chunk_text_invalidates_existing_vector` | Failed: old vector retained | B07 |

The probes reused `build_document` and `make_writer` from `tests/pipelines/test_writer_replacement.py:25-100`.
They reused stores from `tests/pipelines/writer_scenarios.py:17-49`.
The input sequences and observed failures are recorded with each finding above.

### Snapshot classification

The only modified SDK source file is `recon_graphrag/pipelines/writer.py`.
Its diff contains 21 insertions and five deletions.
The change specializes relationship selection for ID-only, chunk-only, and combined inputs.
The graph envelope and projected state fields remain unchanged.

Each finding's defective expression exists in HEAD, outside those changed predicates.
Classification used local source and `git show HEAD:recon_graphrag/pipelines/writer.py`.
No checkout or source rewrite was used.
The focused probes were run on the current snapshot, not separately on an executed HEAD checkout.

## Scope limits and skipped checks

### Source review coverage

The deepest review covered writer validation, state reads, contribution ownership, writes, replacement, receipts, transactions, and retry classification.
Connected source checks covered both backend wrappers, UID constraints, provenance helpers, assembly, record types, and existing writer tests.

The later-stage review sampled these paths:

- Graph-building orchestration, row assembly, and finalization in `recon_graphrag/pipelines/graphrag_pipeline.py`.
- Chunk identity and schema validation in `recon_graphrag/extraction/`.
- Entity merge code in both backend entity resolvers.
- Community orchestration and Neo4j detection in `recon_graphrag/communities/`.
- Entity, chunk, and community-report embedding loops in `recon_graphrag/embeddings/`.
- Local retrieval, shared query templates, graph evidence, and citation handling in `recon_graphrag/retrieval/`.

This was a bounded scan, not an exhaustive audit of every SDK function or branch.
Global and DRIFT internals, provider adapters, and Memgraph community detection did not receive equivalent line-by-line source review.
Their deterministic tests ran, but passing tests do not establish source completeness.

### Graph evidence limitation

MCP `list_projects` returned six projects, none rooted at or above this isolated worktree.
The original repository's index was not treated as evidence for this snapshot.

`check_index_coverage` was requested for this worktree, the writer path, and the `recon_graphrag` and `tests` scopes.
It returned:

```text
project not found or not indexed
```

Therefore, this report uses local source fallback rather than graph traces or index completeness claims.
No new index was created.
The supplied original-project Scout evidence was used only as a lead.

### Backend and runtime limits

No live Neo4j, Memgraph, GDS, APOC, MAGE, LLM, or embedder checks ran.
All integration and E2E checks were excluded.
No database concurrency or real rollback experiment ran.
The deterministic suite includes mocked transaction tests, but those do not establish backend isolation under concurrency.

The stateful writer fake is not a Cypher interpreter.
It cannot represent parallel relationships or domain-label-qualified duplicate identities.
It also stores claim endpoints as fields rather than separate relationships.
B02 and B04 consequently retain an explicit backend-execution verification limit.

No package build, dependency synchronization, lint, or type-check command ran.
There were no packaging or dependency changes, and the repository defines no mandatory lint or type-check command.
No claim of end-to-end validation is made.
