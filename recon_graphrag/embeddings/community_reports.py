"""Vector embedding generation for community reports.

Generates embeddings for community report text, then upserts them into
the graph for semantic retrieval by DRIFT search.
"""

from __future__ import annotations

import asyncio
import logging

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore

logger = logging.getLogger(__name__)


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

    async def embed_reports(self, batch_size: int = 500) -> int:
        """Generate embeddings for community reports without embeddings.

        Loops until all unembedded reports are processed.
        Embeds reports concurrently up to ``self.concurrency`` at a time,
        then batch-upserts all embeddings at once.
        Returns total number of reports embedded.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        last_batch_keys: frozenset | None = None

        while True:
            reports = self.graph_store.get_unembedded_community_reports(
                graph_name=self.graph_name,
                limit=batch_size,
            )
            if not reports:
                break

            # No-forward-progress guard: if the store hands back the same reports
            # it just did, processing them again cannot change the result, so a
            # store that ignores failure marks or vector writes would loop forever.
            batch_keys = frozenset(
                (report.get("id"), report.get("level")) for report in reports
            )
            if batch_keys == last_batch_keys:
                logger.warning(
                    "stopping: %s community reports could not be embedded "
                    "(no progress); leaving them unembedded",
                    len(reports),
                )
                break
            last_batch_keys = batch_keys

            # Returns the exception on failure so tqdm_asyncio.gather (no
            # return_exceptions=) still lets the caller sort ok/failed results.
            async def _embed(report: dict):
                text = self._report_to_text(report)
                if not text:
                    # Returned as a failure, not skipped: an unembeddable report
                    # left unmarked stays eligible for refetch forever.
                    return ValueError("community report has no embeddable text")
                async with semaphore:
                    try:
                        embedding = await self.embedder.async_embed_query(text)
                        if not embedding:
                            raise ValueError("embedder returned an empty vector")
                        return report["id"], int(report["level"]), embedding
                    except Exception as e:
                        return e

            results = await tqdm_asyncio.gather(
                *[_embed(r) for r in reports],
                desc="Embedding reports",
                disable=None,
            )

            ids: list[str] = []
            levels: list[int] = []
            embeddings: list[list[float]] = []

            for report, result in zip(reports, results):
                if isinstance(result, Exception):
                    rid = report.get("id", "?")
                    logger.warning(
                        "error embedding community report '%s': %s", rid, result
                    )
                    try:
                        self._mark_failed(report, str(result))
                    except Exception as mark_error:
                        logger.warning(
                            "error recording embedding failure for '%s': %s",
                            rid, mark_error,
                        )
                        return total
                else:
                    ids.append(result[0])
                    levels.append(result[1])
                    embeddings.append(result[2])

            if ids:
                self.graph_store.upsert_community_report_vectors(
                    ids,
                    embeddings,
                    graph_name=self.graph_name,
                    levels=levels,
                )
                total += len(ids)

        if total == 0:
            logger.info("all community reports already have embeddings")
        else:
            logger.info("embedded %s community reports", total)

        return total

    def _mark_failed(self, report: dict, error: str) -> None:
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
                "graph_name": self.graph_name,
                "id": report.get("id"),
                "level": report.get("level"),
                "error": error[:1000],
            },
        )

    @staticmethod
    def _report_to_text(report: dict) -> str:
        title = report.get("title", "").strip()
        text = report.get("report_text", "").strip()
        parts = []
        if title:
            parts.append(title)
        if text:
            parts.append(text)
        return " - ".join(parts) if parts else ""
