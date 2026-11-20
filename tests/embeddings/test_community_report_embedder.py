import asyncio
import threading
from copy import deepcopy
from typing import Any, cast

import pytest

from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder
from recon_graphrag.entity_text import embedder_fingerprint
from recon_graphrag.graphdb.store_base import _GENERATION_UNGUARDED
from recon_graphrag.report_text import report_embedding_fingerprint
from tests.pipelines.test_async_offload import BlockingCalls, with_heartbeat


class FakeStore:
    """In-memory community report store.

    Honours failure marks and vector writes so ``embed_reports`` converges the
    way the real shared store does.
    """

    # A fix that never converges should fail fast instead of hanging the suite.
    max_fetches = 10

    def __init__(self, reports=None):
        self.reports = reports if reports is not None else [
            {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
            {"id": "c1", "level": 1, "title": "Bad", "report_text": "bad"},
        ]
        self.embedded = set()
        self.failed = set()
        self.upserts = []
        self.fetches = 0
        self.fail_writes = False

    def get_unembedded_community_reports(self, graph_name, limit=500):
        self.fetches += 1
        assert self.fetches <= self.max_fetches, (
            "embed_reports did not converge; it refetched the same reports"
        )
        return [
            report for report in self.reports
            if (report["id"], report["level"]) not in self.embedded | self.failed
        ][:limit]

    def upsert_community_report_vectors(
        self, node_ids, vectors, graph_name, levels
    ):
        self.upserts.append((node_ids, vectors, graph_name, levels))
        self.embedded.update(zip(node_ids, levels, strict=True))

    def execute_query(self, query, parameters=None):
        if self.fail_writes:
            raise RuntimeError("store rejected the failure mark")
        params = parameters or {}
        self.failed.add((params["id"], params["level"]))
        return []


class NoopMarkStore(FakeStore):
    """Store that acknowledges nothing: marks and writes are silently dropped."""

    def execute_query(self, query, parameters=None):
        return []

    def upsert_community_report_vectors(
        self, node_ids, vectors, graph_name, levels
    ):
        self.upserts.append((node_ids, vectors, graph_name, levels))


class PartialEmbedder:
    async def async_embed_query(self, text):
        if "bad" in text.lower():
            raise RuntimeError("provider failure")
        return [0.1, 0.2]


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    async def async_embed_query(self, text):
        self.calls.append(text)
        return [0.1, 0.2]


def _blank_report(report_id="c0", level=0):
    return {"id": report_id, "level": level, "title": "", "report_text": "   \n\t  "}


def _embedder(store, embedder, graph_name="graph-a") -> CommunityReportEmbedder:
    """The fakes implement only the members this test exercises."""
    return CommunityReportEmbedder(cast(Any, store), cast(Any, embedder), graph_name)


@pytest.mark.asyncio
async def test_embed_reports_scopes_keys_and_terminates_after_failure():
    store = FakeStore()
    count = await _embedder(store, PartialEmbedder()).embed_reports()

    assert count == 1
    assert store.upserts == [(["c1"], [[0.1, 0.2]], "graph-a", [0])]
    assert store.failed == {("c1", 1)}


@pytest.mark.asyncio
async def test_blank_report_is_marked_failed_without_calling_the_provider():
    store = FakeStore([_blank_report()])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 0
    assert store.failed == {("c0", 0)}
    assert embedder.calls == []
    assert store.fetches == 2  # processed once, then drained


@pytest.mark.asyncio
async def test_blank_report_does_not_discard_embeddable_reports():
    store = FakeStore([
        {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
        _blank_report("c2"),
    ])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 1
    assert "Good - good" in embedder.calls
    assert store.failed == {("c2", 0)}


@pytest.mark.asyncio
async def test_title_only_report_is_still_embeddable():
    store = FakeStore([
        {"id": "c1", "level": 0, "title": "Only Title", "report_text": "   "},
    ])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 1
    assert store.failed == set()


@pytest.mark.asyncio
async def test_unchanged_batch_stops_when_the_store_ignores_marks_and_writes():
    store = NoopMarkStore([_blank_report()])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 0
    assert store.fetches == 2  # second identical batch stops the loop
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_failure_mark_error_keeps_the_running_count():
    store = FakeStore([_blank_report()])
    store.fail_writes = True

    count = await _embedder(store, CountingEmbedder()).embed_reports()

    assert count == 0
    assert store.failed == set()


@pytest.mark.asyncio
async def test_corrected_report_is_embedded_after_the_error_is_cleared():
    store = FakeStore([_blank_report()])
    embedder = CountingEmbedder()
    project = _embedder(store, embedder)

    assert await project.embed_reports() == 0
    assert store.failed == {("c0", 0)}

    # An operator fixes the externally written row and clears the mark.
    store.reports[0]["report_text"] = "recovered"
    store.failed.clear()

    assert await project.embed_reports() == 1
    assert store.embedded == {("c0", 0)}


# --- Guarded store contract -------------------------------------------------

class GuardedFakeStore:
    """Stateful fake of the private guarded report operations.

    The fake applies the same snapshot checks as the Cypher store so the
    embedder tests can prove stale acknowledgments do not count or overwrite.
    """

    def __init__(self, reports=None):
        self.rows = {}
        for report in reports or []:
            self.add(**report) if isinstance(report, dict) else self.add(*report)
        self.select_calls = []
        self.upsert_calls = []
        self.mark_calls = []
        self.fail_marks = False
        self.execute_query_calls = []

    def add(
        self,
        report_id="c1",
        level=0,
        title="Good",
        report_text="good",
        generation="gen-a",
        embedding_revision=0,
        vector=None,
        input_fingerprint=None,
        content_fingerprint=None,
        model_fingerprint=None,
        error=None,
    ):
        row = {
            "id": report_id,
            "level": level,
            "title": title,
            "report_text": report_text,
            "generation": generation,
            "embedding_revision": embedding_revision,
            "vector": vector,
            "input_fingerprint": input_fingerprint,
            "content_fingerprint": content_fingerprint,
            "model_fingerprint": model_fingerprint,
            "error": error,
        }
        self.rows[(level, report_id)] = row
        return row

    def rewrite(self, report_id, level, title, report_text):
        """Apply the report-write field changes, including vector invalidation."""
        row = self.rows[(level, report_id)]
        old_fingerprint = report_embedding_fingerprint(
            row["title"], row["report_text"]
        )
        new_fingerprint = report_embedding_fingerprint(title, report_text)
        row["title"] = title
        row["report_text"] = report_text
        row["input_fingerprint"] = new_fingerprint
        row["error"] = None
        if old_fingerprint != new_fingerprint:
            row["vector"] = None
            row["content_fingerprint"] = None
            row["model_fingerprint"] = None

    @staticmethod
    def _row_view(row):
        return {
            "id": row["id"],
            "level": row["level"],
            "title": row["title"],
            "report_text": row["report_text"],
            "generation": row["generation"],
            "embedding_revision": row["embedding_revision"],
        }

    def _is_stale(self, row, model_fingerprint):
        return (
            row["vector"] is None
            or row["input_fingerprint"] is None
            or row["content_fingerprint"] is None
            or row["content_fingerprint"] != row["input_fingerprint"]
            or (
                model_fingerprint is not None
                and row["model_fingerprint"] != model_fingerprint
            )
        )

    def _generation_matches(self, row, generation):
        if generation is _GENERATION_UNGUARDED:
            return True
        return row["generation"] == generation

    def _get_unembedded_community_reports_for_embedding(
        self,
        graph_name,
        limit=500,
        *,
        model_fingerprint,
        exclude_keys,
        expected_generation=_GENERATION_UNGUARDED,
    ):
        self.select_calls.append(
            (graph_name, limit, model_fingerprint, expected_generation)
        )
        excluded = {(key["level"], key["id"]) for key in (exclude_keys or [])}
        rows = [
            row
            for (level, report_id), row in sorted(self.rows.items())
            if (level, report_id) not in excluded
            and self._generation_matches(row, expected_generation)
            and self._is_stale(row, model_fingerprint)
        ]
        return [self._row_view(row) for row in rows[:limit]]

    def _upsert_community_report_vectors_guarded(
        self, graph_name, pairs, *, model_fingerprint
    ):
        self.upsert_calls.append((deepcopy(pairs), model_fingerprint))
        confirmed = 0
        for pair in pairs:
            expected = report_embedding_fingerprint(
                pair["title"], pair["report_text"]
            )
            assert pair["content_fingerprint"] == expected
            row = self.rows.get((pair["level"], pair["id"]))
            if row is None:
                continue
            if not self._generation_matches(row, pair["generation"]):
                continue
            if row["embedding_revision"] != pair["embedding_revision"]:
                continue
            if row["title"] != pair["title"]:
                continue
            if row["report_text"] != pair["report_text"]:
                continue
            if (
                row["input_fingerprint"] is not None
                and row["input_fingerprint"] != pair["content_fingerprint"]
            ):
                continue
            if pair.get("vector") is None or pair.get("content_fingerprint") is None:
                continue
            new_revision = pair.get("new_embedding_revision")
            if new_revision is None or new_revision == pair["embedding_revision"]:
                continue
            row.update(
                vector=pair["vector"],
                content_fingerprint=pair["content_fingerprint"],
                input_fingerprint=(
                    row["input_fingerprint"] or pair["content_fingerprint"]
                ),
                model_fingerprint=model_fingerprint,
                embedding_revision=new_revision,
                error=None,
            )
            confirmed += 1
        return confirmed

    def _mark_community_report_embedding_failed_guarded(
        self, graph_name, pair, error
    ):
        self.mark_calls.append((graph_name, deepcopy(pair), error))
        if self.fail_marks:
            raise RuntimeError("store rejected the diagnostic write")
        row = self.rows.get((pair["level"], pair["id"]))
        if row is None:
            return False
        if not self._generation_matches(row, pair["generation"]):
            return False
        if row["embedding_revision"] != pair["embedding_revision"]:
            return False
        if row["title"] != pair["title"]:
            return False
        if row["report_text"] != pair["report_text"]:
            return False
        if (
            row["input_fingerprint"] is not None
            and row["input_fingerprint"] != pair["content_fingerprint"]
        ):
            return False
        row["error"] = error
        return True

    def execute_query(self, query, parameters=None):
        self.execute_query_calls.append((query, parameters))
        return []


class FlakyEmbedder:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def async_embed_query(self, text):
        self.calls.append(text)
        if self.fail and "bad" in text.lower():
            raise RuntimeError("provider failure")
        return [0.1, 0.2]


class ModelEmbedder:
    def __init__(self, model, vector=None):
        self.model = model
        self.vector = [0.1, 0.2] if vector is None else vector
        self.calls = []

    async def async_embed_query(self, text):
        self.calls.append(text)
        return self.vector


class HookEmbedder:
    def __init__(self, hook=None, model="model-a"):
        self.model = model
        self.hook = hook
        self.calls = []

    async def async_embed_query(self, text):
        self.calls.append(text)
        if self.hook is not None:
            self.hook(text)
        return [0.1, 0.2]


def _guarded_embedder(store, embedder, graph_name="graph-a") -> CommunityReportEmbedder:
    return CommunityReportEmbedder(cast(Any, store), cast(Any, embedder), graph_name)


@pytest.mark.asyncio
async def test_guarded_private_pass_binds_the_exact_generation():
    store = GuardedFakeStore([
        {"report_id": "c1", "generation": "gen-a"},
        {"report_id": "c2", "generation": "gen-a"},
    ])
    embedder = CountingEmbedder()

    count = await _guarded_embedder(store, embedder)._embed_reports_for_generation(
        "graph-a", "gen-a"
    )

    assert count == 2
    assert store.select_calls[0][3] == "gen-a"
    pairs, model_fingerprint = store.upsert_calls[0]
    assert {pair["generation"] for pair in pairs} == {"gen-a"}
    assert model_fingerprint == embedder_fingerprint(embedder)
    assert store.execute_query_calls == []


@pytest.mark.asyncio
async def test_guarded_public_pass_samples_each_rows_generation():
    store = GuardedFakeStore([
        {"report_id": "c1", "generation": "gen-a"},
        {"report_id": "c2", "generation": "gen-b"},
    ])

    count = await _guarded_embedder(store, CountingEmbedder()).embed_reports()

    assert count == 2
    assert store.select_calls[0][3] is _GENERATION_UNGUARDED
    pairs, _ = store.upsert_calls[0]
    assert {pair["generation"] for pair in pairs} == {"gen-a", "gen-b"}


@pytest.mark.asyncio
async def test_private_none_selects_only_legacy_null_generation():
    store = GuardedFakeStore([
        {"report_id": "legacy", "generation": None},
        {"report_id": "current", "generation": "gen-a"},
    ])

    count = await _guarded_embedder(store, CountingEmbedder())._embed_reports_for_generation(
        "graph-a", None
    )

    assert count == 1
    pairs, _ = store.upsert_calls[0]
    assert [pair["generation"] for pair in pairs] == [None]


@pytest.mark.asyncio
async def test_permanent_failure_does_not_starve_later_reports_and_retries_later():
    store = GuardedFakeStore([
        {"report_id": "c0", "title": "Bad", "report_text": "bad", "generation": "g1"},
        {"report_id": "c1", "title": "Good", "report_text": "good", "generation": "g1"},
    ])
    embedder = FlakyEmbedder()
    embedder.fail = True
    worker = _guarded_embedder(store, embedder)

    # One batch slot. The failed first key must not hide the second key.
    count = await worker._embed_reports_for_generation("graph-a", "g1", batch_size=1)

    assert count == 1
    assert embedder.calls == ["Bad - bad", "Good - good"]
    assert store.rows[(0, "c0")]["error"] == "provider failure"
    assert store.rows[(0, "c1")]["vector"] == [0.1, 0.2]

    # The next pass retries the permanent failure exactly once.
    embedder.fail = False
    assert await worker._embed_reports_for_generation("graph-a", "g1", batch_size=1) == 1
    assert embedder.calls == ["Bad - bad", "Good - good", "Bad - bad"]
    assert store.rows[(0, "c0")]["vector"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_failure_mark_storage_error_keeps_batch_successes():
    store = GuardedFakeStore([
        {"report_id": "bad", "title": "Bad", "report_text": "bad"},
        {"report_id": "good", "title": "Good", "report_text": "good"},
    ])
    store.fail_marks = True
    embedder = PartialEmbedder()

    count = await _guarded_embedder(store, embedder).embed_reports()

    assert count == 1
    assert store.rows[(0, "good")]["vector"] == [0.1, 0.2]
    assert store.rows[(0, "bad")]["vector"] is None


@pytest.mark.asyncio
async def test_blank_text_calls_no_provider_and_marks_failure():
    store = GuardedFakeStore([
        {"report_id": "blank", "title": "", "report_text": " "},
        {"report_id": "good", "title": "Good", "report_text": "good"},
    ])
    embedder = CountingEmbedder()

    count = await _guarded_embedder(store, embedder)._embed_reports_for_generation(
        "graph-a", "gen-a", batch_size=1
    )

    assert count == 1
    assert embedder.calls == ["Good - good"]
    assert store.rows[(0, "blank")]["error"] == (
        "community report has no embeddable text"
    )


@pytest.mark.asyncio
async def test_legacy_vector_is_refreshed_once_and_certified():
    store = GuardedFakeStore([
        {
            "report_id": "c1",
            "vector": [0.9],
            "input_fingerprint": None,
            "content_fingerprint": None,
            "model_fingerprint": None,
        }
    ])
    embedder = CountingEmbedder()
    worker = _guarded_embedder(store, embedder)

    assert await worker.embed_reports() == 1
    row = store.rows[(0, "c1")]
    assert row["vector"] == [0.1, 0.2]
    assert row["input_fingerprint"] == report_embedding_fingerprint("Good", "good")
    assert row["content_fingerprint"] == row["input_fingerprint"]
    assert row["model_fingerprint"] == worker.model_fingerprint
    assert row["embedding_revision"] == 1

    assert await worker.embed_reports() == 0
    assert embedder.calls == ["Good - good"]


@pytest.mark.asyncio
async def test_model_change_refreshes_and_the_fingerprint_is_stable():
    row_fingerprint = report_embedding_fingerprint("Good", "good")
    store = GuardedFakeStore([
        {
            "report_id": "c1",
            "vector": [0.9],
            "input_fingerprint": row_fingerprint,
            "content_fingerprint": row_fingerprint,
            "model_fingerprint": "old-model",
            "embedding_revision": 2,
        }
    ])
    embedder = ModelEmbedder("new-model")
    worker = _guarded_embedder(store, embedder)
    before = worker.model_fingerprint

    assert await worker.embed_reports() == 1
    assert store.rows[(0, "c1")]["model_fingerprint"] == before
    assert store.rows[(0, "c1")]["embedding_revision"] == 3

    # Changing the provider attribute after construction must not recertify.
    embedder.model = "different-model"
    assert worker.model_fingerprint == before
    assert await worker.embed_reports() == 0


@pytest.mark.asyncio
async def test_text_rewrite_during_provider_call_rejects_the_old_vector():
    store = GuardedFakeStore([{"report_id": "c1", "generation": "g1"}])
    rewrote = {"done": False}

    def rewrite(text):
        if not rewrote["done"]:
            rewrote["done"] = True
            store.rewrite("c1", 0, "New", "new text")

    embedder = HookEmbedder(rewrite)
    worker = _guarded_embedder(store, embedder)

    assert await worker._embed_reports_for_generation("graph-a", "g1") == 0
    assert store.rows[(0, "c1")]["vector"] is None
    assert store.mark_calls == []

    assert await worker._embed_reports_for_generation("graph-a", "g1") == 1
    assert store.rows[(0, "c1")]["vector"] == [0.1, 0.2]
    assert len(embedder.calls) == 2


@pytest.mark.asyncio
async def test_generation_replacement_during_provider_call_rejects_the_old_vector():
    store = GuardedFakeStore([{"report_id": "c1", "generation": "g1"}])

    def replace_generation(text):
        store.rows[(0, "c1")]["generation"] = "g2"

    embedder = HookEmbedder(replace_generation)
    worker = _guarded_embedder(store, embedder)

    assert await worker._embed_reports_for_generation("graph-a", "g1") == 0
    pairs, _ = store.upsert_calls[0]
    assert pairs[0]["generation"] == "g1"
    assert store.rows[(0, "c1")]["vector"] is None

    # A new pass samples the replacement generation without adopting it late.
    assert await worker.embed_reports() == 1
    assert store.rows[(0, "c1")]["vector"] == [0.1, 0.2]
    assert len(embedder.calls) == 2


@pytest.mark.asyncio
async def test_older_success_cannot_overwrite_a_newer_success():
    store = GuardedFakeStore([{"report_id": "c1", "generation": "g1"}])

    def newer_success(text):
        row = store.rows[(0, "c1")]
        pair = {
            "id": "c1",
            "level": 0,
            "title": row["title"],
            "report_text": row["report_text"],
            "generation": row["generation"],
            "embedding_revision": row["embedding_revision"],
            "content_fingerprint": report_embedding_fingerprint(
                row["title"], row["report_text"]
            ),
            "vector": [9.0],
            "new_embedding_revision": 1,
        }
        store._upsert_community_report_vectors_guarded(
            "graph-a", [pair], model_fingerprint="newer-model"
        )

    embedder = HookEmbedder(newer_success)
    worker = _guarded_embedder(store, embedder)

    assert await worker._embed_reports_for_generation("graph-a", "g1") == 0
    row = store.rows[(0, "c1")]
    assert row["vector"] == [9.0]
    assert row["embedding_revision"] == 1


@pytest.mark.asyncio
async def test_older_failure_cannot_overwrite_a_newer_success():
    store = GuardedFakeStore([{"report_id": "c1", "generation": "g1"}])

    def newer_success_then_fail(text):
        row = store.rows[(0, "c1")]
        pair = {
            "id": "c1",
            "level": 0,
            "title": row["title"],
            "report_text": row["report_text"],
            "generation": row["generation"],
            "embedding_revision": row["embedding_revision"],
            "content_fingerprint": report_embedding_fingerprint(
                row["title"], row["report_text"]
            ),
            "vector": [9.0],
            "new_embedding_revision": 1,
        }
        store._upsert_community_report_vectors_guarded(
            "graph-a", [pair], model_fingerprint="newer-model"
        )
        raise RuntimeError("late provider failure")

    embedder = HookEmbedder(newer_success_then_fail)
    worker = _guarded_embedder(store, embedder)

    assert await worker._embed_reports_for_generation("graph-a", "g1") == 0
    row = store.rows[(0, "c1")]
    assert row["vector"] == [9.0]
    assert row["embedding_revision"] == 1
    assert row["error"] is None


@pytest.mark.asyncio
async def test_zero_acknowledgment_is_not_counted_or_marked_failed():
    store = GuardedFakeStore([{"report_id": "c1"}])
    store._upsert_community_report_vectors_guarded = lambda *args, **kwargs: 0

    count = await _guarded_embedder(store, CountingEmbedder()).embed_reports()

    assert count == 0
    assert store.rows[(0, "c1")]["vector"] is None
    assert store.mark_calls == []


@pytest.mark.asyncio
async def test_deleted_community_is_not_counted_and_has_no_failure_mark():
    store = GuardedFakeStore([{"report_id": "c1"}])

    def delete(text):
        store.rows.clear()

    worker = _guarded_embedder(store, HookEmbedder(delete))

    assert await worker.embed_reports() == 0
    assert store.mark_calls == []


@pytest.mark.asyncio
async def test_one_attempt_per_key_when_the_store_repeats_the_row():
    store = GuardedFakeStore([{"report_id": "c1"}])
    original = store._get_unembedded_community_reports_for_embedding

    def duplicate(*args, **kwargs):
        rows = original(*args, **kwargs)
        return rows + rows

    store._get_unembedded_community_reports_for_embedding = duplicate
    embedder = CountingEmbedder()

    count = await _guarded_embedder(store, embedder).embed_reports()

    assert count == 1
    assert embedder.calls == ["Good - good"]


@pytest.mark.asyncio
async def test_custom_store_without_guards_keeps_the_legacy_signatures(caplog):
    store = FakeStore([
        {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
    ])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports(batch_size=1)

    assert count == 1
    assert store.upserts == [(["c1"], [[0.1, 0.2]], "graph-a", [0])]
    assert "does not support guarded embedding" in caplog.text


class BlockingGuardedEmbeddingStore(GuardedFakeStore, BlockingCalls):
    """Guarded report store whose reads and writes block their callers."""

    def __init__(self, reports=None):
        GuardedFakeStore.__init__(self, reports)
        BlockingCalls.__init__(self)

    def _get_unembedded_community_reports_for_embedding(self, *args, **kwargs):
        self.block("embedding_select")
        return super()._get_unembedded_community_reports_for_embedding(
            *args, **kwargs
        )

    def _upsert_community_report_vectors_guarded(self, *args, **kwargs):
        self.block("embedding_upsert")
        return super()._upsert_community_report_vectors_guarded(*args, **kwargs)

    def _mark_community_report_embedding_failed_guarded(self, *args, **kwargs):
        self.block("embedding_failure")
        return super()._mark_community_report_embedding_failed_guarded(
            *args, **kwargs
        )


@pytest.mark.asyncio
async def test_guarded_embedding_select_upsert_and_failure_run_off_loop():
    store = BlockingGuardedEmbeddingStore([
        {"report_id": "c1", "generation": "gen-a"},
        {
            "report_id": "c2",
            "title": "Bad",
            "report_text": "bad",
            "generation": "gen-a",
        },
    ])
    embedder = FlakyEmbedder()
    embedder.fail = True

    count = await with_heartbeat(
        _guarded_embedder(store, embedder)._embed_reports_for_generation(
            "graph-a", "gen-a"
        )
    )

    assert count == 1
    store.assert_offloaded("embedding_select", "embedding_upsert", "embedding_failure")


class GatedVectorStore(GuardedFakeStore):
    """Guarded vector store whose upsert waits for a test gate."""

    def __init__(self):
        super().__init__([{"report_id": "c1", "generation": "gen-a"}])
        self.upsert_started = threading.Event()
        self.release_upsert = threading.Event()
        self.upsert_finished = threading.Event()
        self.upsert_thread: int | None = None

    def _upsert_community_report_vectors_guarded(self, *args, **kwargs):
        self.upsert_thread = threading.get_ident()
        self.upsert_started.set()
        self.release_upsert.wait(5)
        confirmed = super()._upsert_community_report_vectors_guarded(*args, **kwargs)
        self.upsert_finished.set()
        return confirmed


@pytest.mark.asyncio
async def test_cancelled_guarded_vector_write_leaves_a_replaced_generation_untouched():
    store = GatedVectorStore()
    worker = _guarded_embedder(store, CountingEmbedder())
    task = asyncio.create_task(
        worker._embed_reports_for_generation("graph-a", "gen-a")
    )
    assert await asyncio.to_thread(store.upsert_started.wait, 2)

    task.cancel()
    await asyncio.sleep(0.01)
    row = store.rows[(0, "c1")]
    row["generation"] = "gen-b"
    store.release_upsert.set()
    assert await asyncio.to_thread(store.upsert_finished.wait, 2)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert store.upsert_thread != threading.get_ident()
    assert store.upsert_calls[0][0][0]["generation"] == "gen-a"
    assert row["vector"] is None
    assert row["embedding_revision"] == 0
    assert row["error"] is None
    assert store.mark_calls == []
