"""Guarded community report writes and freshness selection.

These tests pin the Cypher shape and the Python return conversion. They do not
prove backend locking or isolation; those checks need a live database.
"""

from recon_graphrag.graphdb.store_base import (
    _GENERATION_UNGUARDED,
    BaseGraphStore,
)
from recon_graphrag.models.artifacts import CommunityReport, report_to_text
from recon_graphrag.report_text import report_embedding_fingerprint

LOCK_PROPERTY = "`__recon_community_write_lock`"
VECTOR_PAIR = {
    "id": "community:1",
    "level": 0,
    "title": "Title",
    "report_text": "Title\nCommunity community:1 (level 0)",
    "generation": "generation-a",
    "embedding_revision": 4,
    "content_fingerprint": report_embedding_fingerprint(
        "Title", "Title\nCommunity community:1 (level 0)"
    ),
    "vector": [0.1, 0.2],
    "new_embedding_revision": 5,
}


class RecordingStore(BaseGraphStore):
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def execute_query(self, query: str, parameters: dict | None = None):
        self.calls.append((query.strip(), parameters or {}))
        return self.results.pop(0) if self.results else []


def _assert_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _report(
    *,
    title: str = "Title",
    summary: str = "Summary",
    community_id: str = "community:1",
    level: int = 0,
) -> CommunityReport:
    return CommunityReport(
        id=f"report:{community_id}:{level}",
        community_id=community_id,
        level=level,
        title=title,
        summary=summary,
    )


def test_public_report_write_locks_and_preserves_exact_text_vector():
    store = RecordingStore()
    report = _report()

    store.store_community_report(report, "graph-a")

    query, params = store.calls[-1]
    lock_set = f"SET c.{LOCK_PROPERTY} = $lock_token"
    lock_cleared = f"SET c.{LOCK_PROPERTY} = NULL"
    assert lock_set in query
    assert lock_cleared in query
    # The lock must be held before the guard reads generation and text.
    assert query.index(lock_set) < query.index("AS same_text")
    assert query.index("AS same_text") < query.index(lock_cleared)
    assert "c.report_embedding_input_fingerprint = $report_text_fingerprint" in query
    assert "WHEN same_text THEN previous_vector ELSE NULL" in query
    assert "WHEN same_text THEN previous_content ELSE NULL" in query
    assert "WHEN same_text THEN previous_model ELSE NULL" in query
    assert "WHEN same_text THEN previous_error ELSE NULL" in query
    assert "c.report_text = $report_text" in query
    assert "coalesce(c.title, '') = coalesce($title, '')" in query
    assert "RETURN sum(CASE WHEN accepted THEN 1 ELSE 0 END) AS written" in query
    assert "c.report_embedding_revision =" not in query
    assert "CommunityGeneration" not in query
    assert params["guard_generation"] is False
    assert params["generation"] is None
    assert params["lock_token"]
    stored_text = report_to_text(report)
    assert params["report_text"] == stored_text
    assert params["report_text_fingerprint"] == report_embedding_fingerprint(
        report.title, stored_text
    )


def test_guarded_report_write_uses_explicit_generation_and_returns_rejection():
    store = RecordingStore([[{"written": 1}]])
    report = _report()

    assert (
        store._store_community_report_guarded(
            report, "graph-a", expected_generation="generation-a"
        )
        is True
    )

    query, params = store.calls[-1]
    assert params["guard_generation"] is True
    assert params["generation"] == "generation-a"
    assert "c.generation = $generation" in query
    assert "$generation IS NULL AND c.generation IS NULL" in query

    rejected = RecordingStore()
    assert (
        rejected._store_community_report_guarded(
            report, "graph-a", expected_generation=None
        )
        is False
    )
    _, params = rejected.calls[-1]
    assert params["guard_generation"] is True
    assert params["generation"] is None

    _assert_value_error(
        lambda: rejected._store_community_report_guarded(
            report,
            "graph-a",
            expected_generation=_GENERATION_UNGUARDED,  # type: ignore[arg-type]
        )
    )


def test_guarded_report_failure_mark_uses_explicit_generation():
    store = RecordingStore([[{"written": 1}]])

    assert (
        store._mark_community_report_failed_guarded(
            "graph-a",
            "community:1",
            0,
            "bad json",
            expected_generation="generation-a",
        )
        is True
    )

    query, params = store.calls[-1]
    assert LOCK_PROPERTY in query
    assert "c.generation = $generation" in query
    assert "$generation IS NULL AND c.generation IS NULL" in query
    assert "SET c.report_status = 'failed'" in query
    assert params == {
        "graph_name": "graph-a",
        "cid": "community:1",
        "level": 0,
        "error": "bad json",
        "generation": "generation-a",
        "lock_token": params["lock_token"],
    }

    rejected = RecordingStore()
    assert (
        rejected._mark_community_report_failed_guarded(
            "graph-a",
            "community:1",
            0,
            "bad json",
            expected_generation=None,
        )
        is False
    )

    _assert_value_error(
        lambda: rejected._mark_community_report_failed_guarded(
            "graph-a",
            "community:1",
            0,
            "bad json",
            expected_generation=_GENERATION_UNGUARDED,  # type: ignore[arg-type]
        )
    )


def test_get_community_generations_returns_observed_values():
    store = RecordingStore([[{"generation": "generation-a"}, {"generation": None}]])

    assert store._get_community_generations("graph-a", level=2) == [
        "generation-a",
        None,
    ]

    query, params = store.calls[-1]
    assert "RETURN DISTINCT c.generation AS generation" in query
    assert "ORDER BY generation" in query
    assert params == {"graph_name": "graph-a", "level": 2}


def test_private_selection_excludes_before_limit_and_returns_snapshot():
    row = {
        "id": "community:1",
        "level": 0,
        "report_text": "Title\nCommunity community:1 (level 0)",
        "title": "Title",
        "generation": "generation-a",
        "embedding_revision": 4,
    }
    store = RecordingStore([[row]])

    result = store._get_unembedded_community_reports_for_embedding(
        "graph-a",
        limit=7,
        model_fingerprint="model-a",
        exclude_keys=[
            {"id": "community:1", "level": 0},
            {"id": "community:9", "level": 2},
        ],
    )

    assert result == [row]
    query, params = store.calls[-1]
    assert params == {
        "graph_name": "graph-a",
        "limit": 7,
        "model_fingerprint": "model-a",
        "exclude_key_strings": ["0:community:1", "2:community:9"],
        "filter_generation": False,
        "generation": None,
    }
    assert (
        "NOT ((toString(c.level) + ':' + c.id) IN $exclude_key_strings)" in query
    )
    assert query.index("$exclude_key_strings") < query.index("LIMIT $limit")
    assert "ORDER BY c.level, c.id" in query
    assert "c.report_embedding_error" not in query
    assert "c.generation AS generation" in query
    assert "c.report_embedding_revision AS embedding_revision" in query
    assert "coalesce(c.report_embedding_model_fingerprint, '')" in query
    assert "<> $model_fingerprint" in query


def test_private_selection_can_require_an_explicit_generation():
    for expected_generation, filter_generation, generation in (
        (None, True, None),
        ("generation-a", True, "generation-a"),
    ):
        store = RecordingStore()

        store._get_unembedded_community_reports_for_embedding(
            "graph-a",
            model_fingerprint="model-a",
            exclude_keys=[],
            expected_generation=expected_generation,
        )

        _, params = store.calls[-1]
        assert params["filter_generation"] is filter_generation
        assert params["generation"] == generation


def test_public_selection_keeps_the_missing_vector_and_failure_filters():
    row = {
        "id": "community:1",
        "level": 0,
        "report_text": "Title",
        "title": "Title",
    }
    store = RecordingStore([[row]])

    assert store.get_unembedded_community_reports("graph-a", limit=3) == [row]

    query, params = store.calls[-1]
    assert params == {"graph_name": "graph-a", "limit": 3}
    assert "c.report_embedding IS NULL" in query
    assert "c.report_embedding_error IS NULL" in query
    assert "c.report_embedding_input_fingerprint" not in query
    assert "c.report_embedding_content_fingerprint" not in query
    assert "c.report_embedding_model_fingerprint" not in query
    assert "c.generation AS generation" not in query
    assert "c.report_embedding_revision" not in query
    assert "ORDER BY c.level, c.id" in query
    assert "LIMIT $limit" in query


def test_guarded_vector_upsert_checks_every_snapshot_field():
    store = RecordingStore([[{"updated_count": 1}]])

    assert (
        store._upsert_community_report_vectors_guarded(
            "graph-a", [VECTOR_PAIR], model_fingerprint="model-a"
        )
        == 1
    )

    query, params = store.calls[-1]
    assert LOCK_PROPERTY in query
    assert query.index(f"SET c.{LOCK_PROPERTY} = NULL") < query.index("FOREACH")
    assert "WHERE accepted" not in query
    assert "pair.generation IS NULL AND c.generation IS NULL" in query
    assert "pair.generation IS NOT NULL" in query
    assert "pair.embedding_revision IS NULL" in query
    assert "pair.embedding_revision IS NOT NULL" in query
    assert "coalesce(c.title, '') = coalesce(pair.title, '')" in query
    assert "coalesce(c.report_text, '') = coalesce(pair.report_text, '')" in query
    assert "c.report_embedding_input_fingerprint = pair.content_fingerprint" in query
    assert "pair.vector IS NOT NULL AS vector_present" in query
    assert "pair.content_fingerprint IS NOT NULL AS content_certified" in query
    assert "pair.new_embedding_revision IS NOT NULL" in query
    assert "pair.new_embedding_revision" in query
    assert "<> pair.embedding_revision" in query
    assert "c.report_embedding_revision = pair.new_embedding_revision" in query
    assert "coalesce(" in query
    assert "c.report_embedding_input_fingerprint, pair.content_fingerprint" in query
    assert "c.report_embedding_model_fingerprint = $model_fingerprint" in query
    assert "c.report_embedding_error = NULL" in query
    assert "RETURN sum(" in query
    assert "CommunityGeneration" not in query
    assert params["pairs"] == [VECTOR_PAIR]
    assert params["model_fingerprint"] == "model-a"
    assert params["lock_token"]


def test_guarded_vector_upsert_rejects_a_mismatched_content_certificate():
    store = RecordingStore([[{"updated_count": 1}]])
    pair = dict(VECTOR_PAIR, content_fingerprint="not-the-text-fingerprint")

    _assert_value_error(
        lambda: store._upsert_community_report_vectors_guarded(
            "graph-a", [pair], model_fingerprint="model-a"
        )
    )
    assert store.calls == []


def test_guarded_vector_upsert_counts_only_stored_rows():
    missing = RecordingStore()
    assert (
        missing._upsert_community_report_vectors_guarded(
            "graph-a", [VECTOR_PAIR], model_fingerprint="model-a"
        )
        == 0
    )

    empty = RecordingStore()
    assert (
        empty._upsert_community_report_vectors_guarded(
            "graph-a", [], model_fingerprint="model-a"
        )
        == 0
    )
    assert empty.calls == []


def test_guarded_embedding_failure_mark_is_scoped_to_snapshot():
    pair = {
        key: value
        for key, value in VECTOR_PAIR.items()
        if key not in {"vector", "new_embedding_revision"}
    }
    store = RecordingStore([[{"written": 1}]])

    assert (
        store._mark_community_report_embedding_failed_guarded(
            "graph-a", pair, "provider timeout"
        )
        is True
    )

    query, params = store.calls[-1]
    assert LOCK_PROPERTY in query
    assert query.index(f"SET c.{LOCK_PROPERTY} = NULL") < query.index("FOREACH")
    assert "WHERE accepted" not in query
    assert "SET c.report_embedding_error = $error" in query
    assert "c.report_embedding = " not in query
    assert "pair.embedding_revision IS NULL" in query
    assert "c.report_embedding_input_fingerprint = pair.content_fingerprint" in query
    assert "coalesce(c.title, '') = coalesce(pair.title, '')" in query
    assert params == {
        "graph_name": "graph-a",
        "pairs": [pair],
        "error": "provider timeout",
        "lock_token": params["lock_token"],
    }

    missing = RecordingStore()
    assert (
        missing._mark_community_report_embedding_failed_guarded(
            "graph-a", pair, "provider timeout"
        )
        is False
    )


def test_public_vector_upsert_clears_the_failure_marker_and_certification():
    store = RecordingStore()

    result = store.upsert_community_report_vectors(
        ["community:1"], [[0.1, 0.2]], "graph-a", [2]
    )

    assert result is None
    query, params = store.calls[-1]
    assert "c.report_embedding = pair.vector" in query
    assert "c.report_embedding_content_fingerprint = NULL" in query
    assert "c.report_embedding_model_fingerprint = NULL" in query
    assert "coalesce(c.report_embedding_revision, 0) + 1" in query
    assert "c.report_embedding_error = NULL" in query
    assert "c.report_embedding_input_fingerprint" not in query
    assert "$model_fingerprint" not in query
    assert params == {
        "graph_name": "graph-a",
        "pairs": [
            {
                "node_id": "community:1",
                "level": 2,
                "vector": [0.1, 0.2],
            }
        ],
    }
