"""The exact community report text used for embedding and its fingerprint."""

from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder
from recon_graphrag.entity_text import text_fingerprint
from recon_graphrag.report_text import (
    report_embedding_fingerprint,
    report_embedding_text,
)


def test_report_embedding_text_preserves_previous_formatting():
    cases = [
        ("Title", "Body", "Title - Body"),
        ("  Title  ", "  Body  ", "Title - Body"),
        ("", "Body", "Body"),
        (None, "Body", "Body"),
        ("Title", "", "Title"),
        ("Title", None, "Title"),
        ("", "", ""),
        (None, None, ""),
        (
            "Title",
            "Title\nCommunity community:1 (level 0)",
            "Title - Title\nCommunity community:1 (level 0)",
        ),
    ]

    for title, report_text, expected in cases:
        assert report_embedding_text(title, report_text) == expected


def test_report_embedding_fingerprint_hashes_the_exact_text():
    assert report_embedding_fingerprint("Title", "Body") == text_fingerprint(
        "Title - Body"
    )
    assert report_embedding_fingerprint(" Title ", " Body ") == (
        report_embedding_fingerprint("Title", "Body")
    )
    assert report_embedding_fingerprint("Other", "Body") != (
        report_embedding_fingerprint("Title", "Body")
    )


def test_helper_matches_the_existing_embedder_formatting():
    report = {
        "title": "  Title  ",
        "report_text": "  Title\nCommunity community:1 (level 0)  ",
    }

    assert report_embedding_text(report["title"], report["report_text"]) == (
        CommunityReportEmbedder._report_to_text(report)
    )
