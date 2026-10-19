"""Fail-fast unsupported operations for scenario-specific protocol fakes."""

from typing import Never


class UnsupportedGraphStore:
    """Scenarios override only operations they support; other calls fail loudly."""

    def _unsupported(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("Operation is not supported by this test fake")

    async def _unsupported_async(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("Operation is not supported by this test fake")

    execute_query = _unsupported
    write_graph_document = _unsupported
    create_indexes = _unsupported
    drop_indexes = _unsupported
    resolve_entities_exact = _unsupported_async
    resolve_entities_normalized = _unsupported_async
    resolve_entities_fuzzy = _unsupported_async
    resolve_entities_hybrid = _unsupported_async
    create_vector_index = _unsupported
    create_fulltext_index = _unsupported
    upsert_vectors = _unsupported
    get_entities_needing_summary = _unsupported
    get_relationships_needing_summary = _unsupported
    persist_entity_summaries = _unsupported
    persist_relationship_summaries = _unsupported
    vector_search = _unsupported
    keyword_search = _unsupported
    fetch_entity_context = _unsupported
    detect_communities = _unsupported
    get_communities = _unsupported
    get_community_stats = _unsupported
    store_community_report = _unsupported
    mark_community_report_failed = _unsupported
    get_unembedded_entities = _unsupported
    get_community_ranked_context = _unsupported
    get_child_community_reports = _unsupported
    get_claims_for_entities = _unsupported
    resolve_chunk_citations = _unsupported
    get_community_reports_by_keys = _unsupported
    get_unembedded_community_reports = _unsupported
    upsert_community_report_vectors = _unsupported
    vector_search_community_reports = _unsupported
    get_entity_count = _unsupported
    get_chunk_count = _unsupported
    get_evidence_link_count = _unsupported
    get_relationship_count = _unsupported
    backfill_descriptions = _unsupported
    validate_graph_build = _unsupported


class AsyncOnlyLLM:
    """Synchronous generation is unsupported in async-only scenarios."""

    def invoke(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("Synchronous generation is not used by this scenario")


class AsyncOnlyEmbedder:
    """Synchronous embedding is unsupported in async-only scenarios."""

    def embed_query(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("Synchronous embedding is not used by this scenario")
