"""Disposable-database regressions for community report vector freshness.

These tests use deterministic fake embedders. They require a disposable Neo4j
or Memgraph database and never call a live provider.
"""

from __future__ import annotations

from functools import partial
from uuid import uuid4

import pytest

from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder
from recon_graphrag.models.artifacts import CommunityReport
from recon_graphrag.report_text import report_embedding_fingerprint
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env

pytestmark = [pytest.mark.integration, pytest.mark.database]

REPORT_ID = "community:1"
REPORT_LEVEL = 0
REPORT_TITLE = "Title"
REPORT_TEXT = "Body"
REPORT_INDEX_NAME = "community-report-embeddings"


def fake_vector(dimension: int, value: float = 0.25) -> list[float]:
    """Return a deterministic vector for the database's actual index dimension."""
    return [value] * dimension


def read_report_index_dimension(backend: str, store) -> int:
    """Read the configured report vector dimension without changing indexes."""
    if backend == "neo4j":
        rows = store.execute_query("SHOW INDEXES YIELD name, options")
        for row in rows:
            if row.get("name") != REPORT_INDEX_NAME:
                continue
            config = (row.get("options") or {}).get("indexConfig") or {}
            dimension = config.get("vector.dimensions")
            if dimension is not None:
                return int(dimension)
    else:
        for row in store.execute_query("SHOW VECTOR INDEX INFO"):
            if row.get("index_name") != REPORT_INDEX_NAME:
                continue
            dimension = row.get("dimension")
            if dimension is not None:
                return int(dimension)
    pytest.fail(
        f"{REPORT_INDEX_NAME!r} vector index is missing for {backend}. "
        "Run store.create_indexes(embedding_dim=...) before this test; "
        "this test does not create or delete indexes."
    )


class FreshnessEmbedder:
    """Deterministic fake provider with an optional mid-call hook."""

    def __init__(
        self,
        dimension: int,
        model="freshness-fake-v1",
        hook=None,
        vector_value: float = 0.25,
    ):
        self.dimension = dimension
        self.model = model
        self.hook = hook
        self.vector_value = vector_value
        self.calls: list[str] = []
        self.fail_texts: tuple[str, ...] = ()

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return fake_vector(self.dimension, self.vector_value)

    async def async_embed_query(self, text: str, **kwargs) -> list[float]:
        self.calls.append(text)
        if self.hook is not None:
            self.hook(text)
        if any(marker in text for marker in self.fail_texts):
            raise RuntimeError("fake provider failure")
        return fake_vector(self.dimension, self.vector_value)


@pytest.fixture(params=["neo4j", "memgraph"])
def report_store(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "community report freshness",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    graph = f"stage3-report-freshness-{uuid4().hex}"
    try:
        dimension = read_report_index_dimension(request.param, store)
        yield store, graph, dimension
    finally:
        cleanup_graph(store, graph)
        store.driver.close()


def seed_report(
    store,
    graph_name: str,
    *,
    report_id: str = REPORT_ID,
    level: int = REPORT_LEVEL,
    title: str = REPORT_TITLE,
    report_text: str = REPORT_TEXT,
    generation: str | None = "generation-a",
    vector: list[float] | None = None,
    input_fingerprint: str | None = None,
    content_fingerprint: str | None = None,
    model_fingerprint: str | None = None,
    revision: int | None = None,
    error: str | None = None,
) -> None:
    store.execute_query(
        """
        CREATE (c:Community {
            graph_name: $graph_name,
            id: $id,
            level: $level,
            title: $title,
            report_text: $report_text,
            generation: $generation,
            report_embedding: $vector,
            report_embedding_input_fingerprint: $input_fingerprint,
            report_embedding_content_fingerprint: $content_fingerprint,
            report_embedding_model_fingerprint: $model_fingerprint,
            report_embedding_revision: $revision,
            report_embedding_error: $error
        })
        """,
        {
            "graph_name": graph_name,
            "id": report_id,
            "level": level,
            "title": title,
            "report_text": report_text,
            "generation": generation,
            "vector": vector,
            "input_fingerprint": input_fingerprint,
            "content_fingerprint": content_fingerprint,
            "model_fingerprint": model_fingerprint,
            "revision": revision,
            "error": error,
        },
    )


def read_report(
    store,
    graph_name: str,
    *,
    report_id: str = REPORT_ID,
    level: int = REPORT_LEVEL,
) -> dict:
    rows = store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name, id: $id, level: $level})
        RETURN c.title AS title,
               c.report_text AS report_text,
               c.generation AS generation,
               c.report_embedding AS vector,
               c.report_embedding_input_fingerprint AS input_fingerprint,
               c.report_embedding_content_fingerprint AS content_fingerprint,
               c.report_embedding_model_fingerprint AS model_fingerprint,
               c.report_embedding_revision AS revision,
               c.report_embedding_error AS error
        """,
        {"graph_name": graph_name, "id": report_id, "level": level},
    )
    assert len(rows) == 1, f"missing community report: {rows}"
    return rows[0]


def rewrite_report(store, graph_name: str, title: str, summary: str) -> None:
    store.store_community_report(
        CommunityReport(
            id=f"report:{REPORT_ID}:{REPORT_LEVEL}",
            community_id=REPORT_ID,
            level=REPORT_LEVEL,
            title=title,
            summary=summary,
        ),
        graph_name,
    )


def store_newer_vector(store, graph_name: str, dimension: int) -> None:
    """Write a newer certified vector for the selected snapshot."""
    fingerprint = report_embedding_fingerprint(REPORT_TITLE, REPORT_TEXT)
    written = store._upsert_community_report_vectors_guarded(
        graph_name,
        [
            {
                "id": REPORT_ID,
                "level": REPORT_LEVEL,
                "title": REPORT_TITLE,
                "report_text": REPORT_TEXT,
                "generation": "generation-a",
                "embedding_revision": None,
                "content_fingerprint": fingerprint,
                "vector": fake_vector(dimension, 0.9),
                "new_embedding_revision": 1,
            }
        ],
        model_fingerprint="newer-model",
    )
    assert written == 1


@pytest.mark.asyncio
async def test_legacy_vector_is_refreshed_once(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(store, graph, vector=vector(0.5))
    embedder = FreshnessEmbedder(dimension)
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker.embed_reports() == 1

    fingerprint = report_embedding_fingerprint(REPORT_TITLE, REPORT_TEXT)
    row = read_report(store, graph)
    assert row["vector"] == vector()
    assert row["input_fingerprint"] == fingerprint
    assert row["content_fingerprint"] == fingerprint
    assert row["model_fingerprint"] == worker.model_fingerprint
    assert row["revision"] == 1

    assert await worker.embed_reports() == 0
    assert embedder.calls == [f"{REPORT_TITLE} - {REPORT_TEXT}"]


def test_public_select_upsert_loop_ends_after_each_page(report_store):
    store, graph, dimension = report_store
    vector = fake_vector(dimension)
    seed_report(
        store, graph, report_id="community:1", title="First", report_text="Body one"
    )
    seed_report(
        store, graph, report_id="community:2", title="Second", report_text="Body two"
    )
    seed_report(
        store,
        graph,
        report_id="community:3",
        title="Failed",
        report_text="Body three",
        error="previous provider failure",
    )

    first_page = store.get_unembedded_community_reports(graph, limit=1)
    assert [row["id"] for row in first_page] == ["community:1"]
    store.upsert_community_report_vectors(["community:1"], [vector], graph, [0])

    second_page = store.get_unembedded_community_reports(graph, limit=1)
    assert [row["id"] for row in second_page] == ["community:2"]
    store.upsert_community_report_vectors(["community:2"], [vector], graph, [0])

    # The legacy selector excludes a failure marker, so the loop ends here.
    assert store.get_unembedded_community_reports(graph, limit=1) == []
    failed = read_report(store, graph, report_id="community:3")
    assert failed["vector"] is None
    assert failed["error"] == "previous provider failure"


@pytest.mark.asyncio
async def test_text_rewrite_during_embedding_rejects_the_old_vector(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(store, graph)
    rewrote = {"done": False}

    def rewrite(text: str) -> None:
        if not rewrote["done"]:
            rewrote["done"] = True
            rewrite_report(store, graph, "New title", "new body")

    embedder = FreshnessEmbedder(dimension, hook=rewrite)
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert "new body" in row["report_text"]
    assert row["vector"] is None
    assert row["content_fingerprint"] is None

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 1
    row = read_report(store, graph)
    assert row["vector"] == vector()
    assert row["content_fingerprint"] == row["input_fingerprint"]
    assert len(embedder.calls) == 2


@pytest.mark.asyncio
async def test_replacement_during_embedding_rejects_the_old_generation(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(store, graph, generation="generation-a")

    def replace_generation(text: str) -> None:
        store.execute_query(
            """
            MATCH (c:Community {graph_name: $graph_name, id: $id, level: $level})
            SET c.generation = 'generation-b'
            """,
            {"graph_name": graph, "id": REPORT_ID, "level": REPORT_LEVEL},
        )

    worker = CommunityReportEmbedder(
        store,
        FreshnessEmbedder(dimension, hook=replace_generation),
        graph_name=graph,
    )

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert row["generation"] == "generation-b"
    assert row["vector"] is None

    assert await worker._embed_reports_for_generation(graph, "generation-b") == 1
    assert read_report(store, graph)["vector"] == vector()


@pytest.mark.asyncio
async def test_permanent_failure_does_not_starve_later_reports(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(
        store,
        graph,
        report_id="a-bad",
        title="Bad",
        report_text="Body FAIL",
    )
    seed_report(
        store,
        graph,
        report_id="b-good",
        title="Good",
        report_text="Body",
    )
    embedder = FreshnessEmbedder(dimension)
    embedder.fail_texts = ("FAIL",)
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    # One batch slot. The permanently failing first key must not hide the
    # second report from the same pass.
    assert await worker.embed_reports(batch_size=1) == 1
    assert embedder.calls == ["Bad - Body FAIL", "Good - Body"]
    assert read_report(store, graph, report_id="a-bad")["error"] == (
        "fake provider failure"
    )
    assert read_report(store, graph, report_id="b-good")["vector"] == vector()

    # The next pass retries the failure once and heals it.
    embedder.fail_texts = ()
    assert await worker.embed_reports(batch_size=1) == 1
    assert len(embedder.calls) == 3
    assert read_report(store, graph, report_id="a-bad")["vector"] == vector()
    assert read_report(store, graph, report_id="a-bad")["error"] is None


@pytest.mark.asyncio
async def test_stale_failure_mark_does_not_hide_rewritten_report(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(store, graph)

    def rewrite_then_fail(text: str) -> None:
        rewrite_report(store, graph, "New title", "new body")
        raise RuntimeError("late provider failure")

    embedder = FreshnessEmbedder(dimension, hook=rewrite_then_fail)
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert "new body" in row["report_text"]
    assert row["error"] is None

    embedder.hook = None
    assert await worker._embed_reports_for_generation(graph, "generation-a") == 1
    assert read_report(store, graph)["vector"] == vector()


@pytest.mark.asyncio
async def test_deleted_community_acknowledgment_is_not_counted(report_store):
    store, graph, dimension = report_store
    seed_report(store, graph)

    def delete_community(text: str) -> None:
        cleanup_graph(store, graph)

    worker = CommunityReportEmbedder(
        store,
        FreshnessEmbedder(dimension, hook=delete_community),
        graph_name=graph,
    )

    assert await worker.embed_reports() == 0


@pytest.mark.asyncio
async def test_model_change_refreshes_the_report_vector(report_store):
    store, graph, dimension = report_store
    vector = partial(fake_vector, dimension)
    seed_report(store, graph, vector=vector(0.5))

    old_embedder = FreshnessEmbedder(dimension, model="model-a")
    old_worker = CommunityReportEmbedder(store, old_embedder, graph_name=graph)
    assert await old_worker.embed_reports() == 1
    first = read_report(store, graph)
    assert first["vector"] == vector()
    assert first["model_fingerprint"] == old_worker.model_fingerprint
    assert first["revision"] == 1

    new_embedder = FreshnessEmbedder(
        dimension, model="model-b", vector_value=0.75
    )
    new_worker = CommunityReportEmbedder(store, new_embedder, graph_name=graph)
    assert await new_worker.embed_reports() == 1

    second = read_report(store, graph)
    assert second["vector"] == vector(0.75)
    assert second["model_fingerprint"] == new_worker.model_fingerprint
    assert second["model_fingerprint"] != old_worker.model_fingerprint
    assert second["revision"] == 2
    assert new_embedder.calls == [f"{REPORT_TITLE} - {REPORT_TEXT}"]


@pytest.mark.asyncio
async def test_older_success_cannot_replace_a_newer_vector(report_store):
    store, graph, dimension = report_store
    seed_report(store, graph)

    def write_newer_then_succeed(text: str) -> None:
        store_newer_vector(store, graph, dimension)

    embedder = FreshnessEmbedder(
        dimension, hook=write_newer_then_succeed, vector_value=0.1
    )
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert row["vector"] == pytest.approx(fake_vector(dimension, 0.9))
    assert row["model_fingerprint"] == "newer-model"
    assert row["revision"] == 1
    assert row["error"] is None


@pytest.mark.asyncio
async def test_older_failure_cannot_replace_a_newer_vector(report_store):
    store, graph, dimension = report_store
    seed_report(store, graph)

    def write_newer_then_fail(text: str) -> None:
        store_newer_vector(store, graph, dimension)
        raise RuntimeError("late provider failure")

    embedder = FreshnessEmbedder(dimension, hook=write_newer_then_fail)
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert row["vector"] == pytest.approx(fake_vector(dimension, 0.9))
    assert row["error"] is None
    assert row["revision"] == 1


@pytest.mark.asyncio
async def test_replacement_during_embedding_rejects_the_old_generation_failure(
    report_store,
):
    store, graph, dimension = report_store
    seed_report(store, graph, generation="generation-a")

    def replace_generation_then_fail(text: str) -> None:
        store.execute_query(
            """
            MATCH (c:Community {graph_name: $graph_name, id: $id, level: $level})
            SET c.generation = 'generation-b'
            """,
            {"graph_name": graph, "id": REPORT_ID, "level": REPORT_LEVEL},
        )
        raise RuntimeError("late provider failure")

    embedder = FreshnessEmbedder(
        dimension, hook=replace_generation_then_fail
    )
    worker = CommunityReportEmbedder(store, embedder, graph_name=graph)

    assert await worker._embed_reports_for_generation(graph, "generation-a") == 0
    row = read_report(store, graph)
    assert row["generation"] == "generation-b"
    assert row["vector"] is None
    assert row["error"] is None
