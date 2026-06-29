"""Vector embedding generation for community reports.

Generates embeddings for community report text, then upserts them into
the graph for semantic retrieval by DRIFT search.
"""

from __future__ import annotations

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore


class CommunityReportEmbedder:
    """Generate and store vector embeddings for community reports."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: BaseEmbedder,
        graph_name: str = "entity-graph",
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.graph_name = graph_name

    async def embed_reports(self, batch_size: int = 500) -> int:
        """Generate embeddings for community reports without embeddings.

        Loops until all unembedded reports are processed.
        Embeds report text and title.
        Returns total number of reports embedded.
        """
        total = 0

        while True:
            reports = self.graph_store.get_unembedded_community_reports(
                graph_name=self.graph_name,
                limit=batch_size,
            )
            if not reports:
                break

            ids: list[str] = []
            levels: list[int] = []
            embeddings: list[list[float]] = []

            for report in reports:
                try:
                    text = self._report_to_text(report)
                    if not text:
                        continue
                    embedding = await self.embedder.async_embed_query(text)
                    if not embedding:
                        raise ValueError("embedder returned an empty vector")
                    ids.append(report["id"])
                    levels.append(int(report["level"]))
                    embeddings.append(embedding)
                except Exception as e:
                    rid = report.get("id", "?")
                    print(f"  Error embedding community report '{rid}': {e}")
                    try:
                        self._mark_failed(report, str(e))
                    except Exception as mark_error:
                        print(
                            f"  Error recording embedding failure for '{rid}': "
                            f"{mark_error}"
                        )
                        return total

            if ids:
                self.graph_store.upsert_community_report_vectors(
                    ids,
                    embeddings,
                    graph_name=self.graph_name,
                    levels=levels,
                )
                total += len(ids)

        if total == 0:
            print("  All community reports already have embeddings.")
        else:
            print(f"  Embedded {total} community reports.")

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
