# GraphBuilderPipeline data-quality fixes

This report records seven fixes and their regression checks.
The fixes preserve add-only gleaning, document-owned evidence, identity safeguards, and strict embedding completion.
Relationship embeddings remain outside the current pipeline.

## Reproductions and changes

Each focused regression test failed before its corresponding fix.
The affected deterministic tests passed before work advanced to the next finding.
The final mandatory suite includes all seven regression groups.

| Finding | Before-fix evidence | Implemented change | Focused regression coverage |
| --- | --- | --- | --- |
| F1: malformed claim descriptions and unvalidated output | 17 failed; 4 passed | Request strict structured output. Validate the envelope locally. Reject unsupported adapters before pipeline writes. Preserve standalone parser arrays and defaults. | `tests/extraction/test_claim_quality.py`; pipeline preflight coverage in `tests/pipelines/test_graph_builder.py` |
| F2: distinct assertion collisions | Five assembler failures; one writer failure; both live backends collapsed distinct assertions | Hash mapped endpoints, object, time interval, and assertion content. Reject conflicting assertions under one submitted ID before database access. | `tests/extraction/test_claim_identity.py`; `tests/pipelines/test_claim_identity.py` |
| F3: stale summaries after property changes | Four failed; one passed | Compare previous and effective prompt inputs. Preserve omitted properties. Handle explicit null removals. Invalidate provisional summaries after resolution. | `tests/pipelines/test_summary_freshness.py`; live summary and resolution cases |
| F4: normalized dispatcher drops conflict rules | Two failed | Forward `conflict_properties` through generic normalized dispatch. Keep direct pipeline strategy calls unchanged. | `tests/graphdb/test_entity_resolution_contract.py` |
| F5: later relationship descriptions disappear | Four failed | Retain unique nonempty descriptions within each document contribution. Preserve counts, source chunks, frequency weights, and first-observation strength. | `tests/extraction/test_relationship_descriptions.py`; live replacement case |
| F6: entity preview truncation loses raw evidence | Five failed | Persist complete owned observations. Partition oversized summary inputs. Require every evidence batch before success. Preserve raw evidence after failures. | `tests/extraction/test_description_coverage.py`; live complete-evidence case |
| F7: failed first embedding batch blocks later nodes | Two deterministic failures; both live backends reproduced starvation | Exclude attempted backend-native IDs before limiting. Keep failures pending and preserve termination protection. | `tests/embeddings/test_embedding_scheduling.py`; live backlog and recovery case |

F6 also tests Unicode boundaries, invalid budgets, oversized output, incomplete output, nonshrinking consolidation, and failed-batch recovery.
The summary policy fingerprint includes budget, output reserve, and supported counter configuration.

## Intentional contract changes

### Claim extraction

Claim extraction now requires strict structured output.
The prompt requires the `claims` object envelope and nullable optional fields.
Unsupported adapters fail explicitly; extraction has no prompt-only fallback.
Standalone `ClaimParser` still accepts legacy arrays and applies its existing defaults.

Generated claim IDs use the `claim:v2:` prefix and a full SHA256 digest.
Status and source metadata remain mutable; they do not define assertion identity.
Custom `ClaimRecord.id` values remain unchanged.
Re-ingestion replaces legacy generated claims through document reconciliation.
Refresh external references that retain old generated IDs.

### Entity embedding

`GraphStore.get_unembedded_entities` adds keyword-only `exclude_ids`.
Both built-in backends apply exclusions before the batch limit.
Neo4j uses string element IDs; Memgraph uses integer node IDs.
Custom adapters must implement this argument before running the new embedding pass.
Existing positional arguments and result shapes remain unchanged.

The public API fixture changes only the three affected method signatures.
Other compatibility snapshots remain unchanged.

## Final validation

| Check | Result |
| --- | --- |
| `uv run pytest -m 'not integration' -q` | 1,872 passed; 1 skipped; 288 deselected |
| Focused live quality checks and writer invariants | 47 passed; 1 skipped |
| Historical live writer regressions | 128 passed |
| Deterministic database workflows with locally pinned dimensions | 2 passed |
| Live stage inspection with configured LLM and embedder | 10 passed across Neo4j and Memgraph |
| Active LSP error probe across 23 implementation and regression files | No reported errors; two existing policy-suppressed findings were excluded |
| `git diff --check` | Passed |

Live tests used disposable services from `docker-compose.yml`, started with Podman Compose.
Temporary overrides isolated container names and networks.
The original stopped containers remained untouched.
The disposable test containers were stopped after validation, without deleting their data.

The first broad database run failed with 34 failures, 128 setup errors, 106 passes, and three skips.
It combined incompatible 1,536-dimension and 768-dimension index requirements.
Separate disposable database instances resolved the dimension conflicts.
The new embedding test also needed vectors matching the default 1,536-dimension index.

Thirty-one failures reproduced on an untouched archive of HEAD `6243363`.
They affect bounded retrieval, community-member ordering, one resolution fixture, and Neo4j performance assertions.
Those baseline failures remain unresolved and outside these seven fixes.
The broad database suite must not be described as passing.

The existing deterministic workflow fixture probes the configured embedder for its dimension.
The initial broad run reached that probe.
The successful rerun pinned its cached dimension to 768 through a temporary pytest plugin.
That rerun used fake LLM and embedding responses against real databases.
A subsequent authorized live stage-inspection run passed all ten tests in 413.97 seconds.
That run used the configured LLM and embedder against both disposable databases.
The first attempt reached databases before startup completed and failed during fixture setup.
The rerun started after both services became healthy.

The live inspection covered extraction, deterministic resolution, vector search, community reports, and retrieval citations.
Memgraph stored vectors for all nine inspected entities and both chunks.
Neo4j stored vectors for all ten inspected entities and both chunks.
The community stages produced and embedded seven Memgraph reports and four Neo4j reports.
Local and global retrieval returned source citations on both backends.
These counts come from separate stochastic builds and are not cross-backend parity requirements.
This scenario disables gleaning and claim extraction; it does not validate their live-provider paths.

## Operational limits

Complete evidence submission does not guarantee complete LLM summaries.
The default token counter estimates tokens; provider overhead and billing can differ.
Local validation rejects oversized or incomplete summaries after bounded retries.
Configure provider-side output limits separately when required.
Fixed context that exceeds the budget fails without discarding evidence.

Raw text discarded by earlier writes cannot be recovered by summary refresh alone.
Re-ingest or re-extract affected documents from their source data.
Each embedding pass retains attempted IDs in memory and sends exclusions to the backend.
Strict completion remains graph-wide and point-in-time, not atomic under concurrent writes.
