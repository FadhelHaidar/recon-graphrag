"""Canonical text embedded for a community report, plus its fingerprint.

The store records the fingerprint of the report text it expects to be embedded
and the embedder records the fingerprint of the text it actually embedded.
Both sides must build the text identically, so the formatting lives here.
"""

from __future__ import annotations

from recon_graphrag.entity_text import text_fingerprint


def report_embedding_text(title: str | None, report_text: str | None) -> str:
    """Build the exact string embedded for one community report.

    Keep the previous formatting: strip both parts and join the non-empty
    parts with ``" - "``. The stored report text already starts with the
    title, so this text contains the title twice. A format change mixes vector
    spaces, so do not change it without re-embedding every report.
    """
    title = (title or "").strip()
    text = (report_text or "").strip()
    return " - ".join(part for part in (title, text) if part)


def report_embedding_fingerprint(title: str | None, report_text: str | None) -> str:
    """Hash the exact report text that an embedding vector represents."""
    return text_fingerprint(report_embedding_text(title, report_text))


__all__ = ["report_embedding_fingerprint", "report_embedding_text"]
