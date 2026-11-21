"""Vector embedding generation for community reports.

Generates embeddings for community report text, then upserts them into
the graph for semantic retrieval by DRIFT search.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, cast

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.entity_text import embedder_fingerprint
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.store_base import _GENERATION_UNGUARDED
from recon_graphrag.report_text import (
    report_embedding_fingerprint,
    report_embedding_text,
)

logger = logging.getLogger(__name__)

_GUARDED_STORE_OPERATIONS = (
    "_get_unembedded_community_reports_for_embedding",
    "_upsert_community_report_vectors_guarded",
    "_mark_community_report_embedding_failed_guarded",
)


@dataclass(frozen=True)
class _ReportSnapshot:
    """Immutable report state read before the provider call.

    Generation, revision, title, and report text must come from the same
    selection row. Re-reading any of them after the provider call would let a
    late result adopt a newer generation or newer text.
    """

    id: str
    level: int
    title: str
    report_text: str
    generation: str | None
    embedding_revision: int | None
    content_fingerprint: str
    text: str

    @classmethod
    def from_row(cls, row: dict) -> _ReportSnapshot:
        title = row.get("title") or ""
        report_text = row.get("report_text") or ""
        revision = row.get("embedding_revision")
        return cls(
            id=row["id"],
            level=int(row["level"]),
            title=title,
            report_text=report_text,
            generation=row.get("generation"),
            embedding_revision=None if revision is None else int(revision),
            content_fingerprint=report_embedding_fingerprint(title, report_text),
            text=report_embedding_text(title, report_text),
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.id, self.level)

    @property
    def success_revision(self) -> int:
        """Return a revision that is newer than the selected snapshot."""
        return (self.embedding_revision or 0) + 1

    def failure_pair(self) -> dict:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "report_text": self.report_text,
            "generation": self.generation,
            "embedding_revision": self.embedding_revision,
            "content_fingerprint": self.content_fingerprint,
        }

    def success_pair(self, vector: list[float]) -> dict:
        pair = self.failure_pair()
        pair["vector"] = vector
        pair["new_embedding_revision"] = self.success_revision
        return pair

    def as_row(self) -> dict:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "report_text": self.report_text,
        }


@dataclass(frozen=True)
class _EmbeddedReport:
    snapshot: _ReportSnapshot
    vector: list[float]


@dataclass(frozen=True)
class _FailedReport:
    snapshot: _ReportSnapshot
    error: Exception


class CommunityReportEmbedder:
    """Generate and store vector embeddings for community reports."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: BaseEmbedder,
        graph_name: str = "entity-graph",
        concurrency: int = 5,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.graph_name = graph_name
        self.concurrency = concurrency
        # Read once: a late provider result must not certify a different model.
        self.model_fingerprint = embedder_fingerprint(embedder)
        self._unguarded_warning_logged = False

    async def embed_reports(self, batch_size: int = 500) -> int:
        """Generate missing or stale community report embeddings.

        Loops until all pending reports are processed. Embeds reports
        concurrently up to ``self.concurrency`` at a time, then batch-upserts
        all embeddings at once. Returns the number of confirmed vector writes.

        A store without the guarded private operations uses the legacy public
        operations. That path does not detect a model change and counts
        attempted rows because the store cannot confirm each write.
        """
        # The stand-alone caller has no detection token. Sample every observed
        # generation, then write each result against the generation in its row.
        return await self._embed_reports_for_generation(
            self.graph_name,
            _GENERATION_UNGUARDED,
            batch_size=batch_size,
        )

    async def _embed_reports_for_generation(
        self,
        graph_name: str,
        expected_generation: str | None | object,
        *,
        batch_size: int = 500,
    ) -> int:
        """Embed reports for the pipeline's detection generation.

        ``None`` selects only communities whose generation is NULL. A token
        selects one generation exactly. The stand-alone public pass passes the
        store sentinel, which samples all generations; that wildcard never
        reaches a guarded write because every write uses the row snapshot.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        attempted: set[tuple[str, int]] = set()
        last_batch_keys: frozenset | None = None
        guarded = self._supports_guarded_operations()
        if not guarded:
            self._warn_unguarded_once()

        while True:
            reports = await asyncio.to_thread(
                self._select_reports,
                graph_name,
                expected_generation,
                batch_size,
                attempted,
                guarded=guarded,
            )
            batch: list[_ReportSnapshot] = []
            batch_keys: set[tuple[str, int]] = set()
            for report in reports:
                snapshot = _ReportSnapshot.from_row(report)
                if snapshot.key in attempted or snapshot.key in batch_keys:
                    continue
                batch.append(snapshot)
                batch_keys.add(snapshot.key)

            if not batch:
                if reports:
                    logger.warning(
                        "stopping: %s community reports could not be embedded "
                        "(no progress); leaving them unembedded",
                        len(reports),
                    )
                break

            # The store sees every attempted key before LIMIT. The client
            # filter above also protects stores that ignore exclusions.
            batch_key_set = frozenset(batch_keys)
            if batch_key_set == last_batch_keys:
                logger.warning(
                    "stopping: %s community reports could not be embedded "
                    "(no progress); leaving them unembedded",
                    len(batch),
                )
                break
            last_batch_keys = batch_key_set
            attempted.update(batch_keys)

            results = await tqdm_asyncio.gather(
                *[self._embed_snapshot(snapshot, semaphore) for snapshot in batch],
                desc="Embedding reports",
                disable=None,
            )

            embedded: list[_EmbeddedReport] = []
            failed: list[_FailedReport] = []
            for result in results:
                if isinstance(result, _EmbeddedReport):
                    embedded.append(result)
                else:
                    failed.append(result)

            # Save confirmed successes before diagnostics. A diagnostic write
            # must never discard vectors that the provider already produced.
            total += await asyncio.to_thread(
                self._store_successes, graph_name, embedded, guarded=guarded
            )
            await asyncio.to_thread(
                self._store_failures, graph_name, failed, guarded=guarded
            )

        if total == 0:
            logger.info("all community reports already have embeddings")
        else:
            logger.info("embedded %s community reports", total)

        return total

    def _supports_guarded_operations(self) -> bool:
        """Return True when the store exposes the guarded report operations."""
        return all(
            callable(getattr(self.graph_store, name, None))
            for name in _GUARDED_STORE_OPERATIONS
        )

    def _warn_unguarded_once(self) -> None:
        """State the weaker custom-store guarantee without warning every batch."""
        if self._unguarded_warning_logged:
            return
        logger.warning(
            "community report store does not support guarded embedding; "
            "stale vectors and model changes are not detected"
        )
        self._unguarded_warning_logged = True

    def _select_reports(
        self,
        graph_name: str,
        expected_generation: str | None | object,
        batch_size: int,
        attempted: set[tuple[str, int]],
        *,
        guarded: bool,
    ) -> list[dict]:
        if not guarded:
            return self.graph_store.get_unembedded_community_reports(
                graph_name=graph_name,
                limit=batch_size,
            )
        store = cast(Any, self.graph_store)
        return store._get_unembedded_community_reports_for_embedding(
            graph_name,
            batch_size,
            model_fingerprint=self.model_fingerprint,
            exclude_keys=[
                {"id": report_id, "level": level}
                for report_id, level in sorted(attempted)
            ],
            expected_generation=expected_generation,
        )

    async def _embed_snapshot(
        self,
        snapshot: _ReportSnapshot,
        semaphore: asyncio.Semaphore,
    ) -> _EmbeddedReport | _FailedReport:
        if not snapshot.text:
            # Do not spend a provider call on blank text. Record the failure so
            # the report stays visible and can be fixed and retried later.
            return _FailedReport(
                snapshot,
                ValueError("community report has no embeddable text"),
            )
        async with semaphore:
            try:
                embedding = await self.embedder.async_embed_query(snapshot.text)
                if not embedding:
                    raise ValueError("embedder returned an empty vector")
                return _EmbeddedReport(snapshot, list(embedding))
            except Exception as error:
                return _FailedReport(snapshot, error)

    def _store_successes(
        self,
        graph_name: str,
        embedded: list[_EmbeddedReport],
        *,
        guarded: bool,
    ) -> int:
        if not embedded:
            return 0
        if not guarded:
            ids = [result.snapshot.id for result in embedded]
            levels = [result.snapshot.level for result in embedded]
            vectors = [result.vector for result in embedded]
            self.graph_store.upsert_community_report_vectors(
                ids,
                vectors,
                graph_name,
                levels,
            )
            return len(ids)

        store = cast(Any, self.graph_store)
        pairs = [
            result.snapshot.success_pair(result.vector) for result in embedded
        ]
        confirmed = store._upsert_community_report_vectors_guarded(
            graph_name,
            pairs,
            model_fingerprint=self.model_fingerprint,
        )
        if confirmed < len(pairs):
            logger.debug(
                "stored %s of %s community report vectors; stale snapshots "
                "remain for a later pass",
                confirmed,
                len(pairs),
            )
        return int(confirmed)

    def _store_failures(
        self,
        graph_name: str,
        failed: list[_FailedReport],
        *,
        guarded: bool,
    ) -> None:
        store = cast(Any, self.graph_store)
        for failure in failed:
            snapshot = failure.snapshot
            error = str(failure.error)
            logger.warning(
                "error embedding community report '%s': %s",
                snapshot.id,
                error,
            )
            try:
                if guarded:
                    accepted = store._mark_community_report_embedding_failed_guarded(
                        graph_name,
                        snapshot.failure_pair(),
                        error[:1000],
                    )
                    if not accepted:
                        logger.debug(
                            "stale embedding failure marker rejected for '%s'",
                            snapshot.id,
                        )
                else:
                    self._mark_failed(
                        snapshot.as_row(),
                        error,
                        graph_name=graph_name,
                    )
            except Exception as mark_error:
                # Diagnostics are best effort. Keep the batch's confirmed
                # successes and let a later pass retry the report.
                logger.warning(
                    "error recording embedding failure for '%s': %s",
                    snapshot.id,
                    mark_error,
                )

    def _mark_failed(
        self,
        report: dict,
        error: str,
        *,
        graph_name: str | None = None,
    ) -> None:
        """Exclude a failed report until its fingerprint changes."""
        self.graph_store.execute_query(
            """
            MATCH (c:Community {
                graph_name: $graph_name,
                id: $id,
                level: $level
            })
            SET c.report_embedding_error = $error
            """,
            {
                "graph_name": graph_name or self.graph_name,
                "id": report.get("id"),
                "level": report.get("level"),
                "error": error[:1000],
            },
        )

    @staticmethod
    def _report_to_text(report: dict) -> str:
        return report_embedding_text(
            report.get("title"),
            report.get("report_text"),
        )
