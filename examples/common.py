"""Shared helpers for the movie example scripts.

Provides backend selection, extraction, ingestion, and search utilities
used by extract.py, ingest.py, communities.py, search.py, and compare_backends.py.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from recon_graphrag import IndexManager as Neo4jIndexManager
from recon_graphrag.embeddings import EntityEmbedder
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import PageWindowBuilder
from recon_graphrag.extraction.extractor import LLMGraphExtractor
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.extraction.validator import SchemaValidator
from recon_graphrag.graphdb.memgraph.index_manager import IndexManager as MemgraphIndexManager
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever

try:
    from .config import EMBEDDING_DIM, get_memgraph_store, get_neo4j_store
    from .prompts import (
        DRIFT_ANSWER_PROMPT,
        GLOBAL_MAP_PROMPT,
        GLOBAL_REDUCE_PROMPT,
        LOCAL_ANSWER_PROMPT,
    )
except ImportError:
    from config import EMBEDDING_DIM, get_memgraph_store, get_neo4j_store
    from prompts import (
        DRIFT_ANSWER_PROMPT,
        GLOBAL_MAP_PROMPT,
        GLOBAL_REDUCE_PROMPT,
        LOCAL_ANSWER_PROMPT,
    )


DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "movie_graph.json"

ALL_BACKENDS = "all"

SEARCH_OPTIONS = {
    "local": {},
    "global": {
        "community_level": "coarsest",
    },
    "drift": {
        "top_k": 10,
        "community_level": "finest",
    },
}

BACKEND_REGISTRY = {
    "neo4j": (get_neo4j_store, Neo4jIndexManager),
    "memgraph": (get_memgraph_store, MemgraphIndexManager),
}
BACKEND_CHOICES = (*BACKEND_REGISTRY, ALL_BACKENDS)


def get_backend_targets(backend: str) -> list[tuple[str, Any, type]]:
    """Return (name, store, index_manager_cls) for the selected backend(s)."""
    if backend == ALL_BACKENDS:
        selected = BACKEND_REGISTRY.items()
    elif backend in BACKEND_REGISTRY:
        selected = [(backend, BACKEND_REGISTRY[backend])]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return [
        (name, store_factory(), index_manager_cls)
        for name, (store_factory, index_manager_cls) in selected
    ]


def configure_movie_rag(
    graph_store, llm, embedder, *, graph_name: str = "entity-graph"
) -> dict[str, Any]:
    """Create and configure local, global, and drift search instances."""
    local = LocalSearchRetriever(graph_store, llm, embedder, graph_name=graph_name)
    local.answer_prompt = LOCAL_ANSWER_PROMPT

    global_search = GlobalSearchRetriever(graph_store, llm, graph_name=graph_name)
    global_search.map_prompt = GLOBAL_MAP_PROMPT
    global_search.reduce_prompt = GLOBAL_REDUCE_PROMPT

    drift = DriftSearchRetriever(graph_store, llm, embedder, graph_name=graph_name)
    drift.reduce_prompt = DRIFT_ANSWER_PROMPT

    return {"local": local, "global": global_search, "drift": drift}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_document_id(text: str, metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or metadata.get("title")
    if source:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(source).lower()).strip("-")
        return f"doc:{slug}"
    return f"doc:{_hash_text(text)[:16]}"


async def extract_graph_document_from_pages(
    pages: list[str],
    llm,
    schema,
    metadata: dict[str, Any] | None = None,
    graph_name: str = "entity-graph",
    window_size: int = 2,
    window_overlap: int = 1,
) -> GraphDocument:
    """Run LLM extraction over pages and return a neutral GraphDocument."""
    metadata = metadata or {}
    text = "\n\n".join(pages)
    document_id = _make_document_id(text, metadata)
    text_hash = _hash_text(text)
    chunks = PageWindowBuilder(
        window_size=window_size,
        window_overlap=window_overlap,
    ).build_windows(pages=pages, document_id=document_id, metadata=metadata)

    extractor = LLMGraphExtractor(llm)
    validator = SchemaValidator()
    chunk_extractions = {}
    extraction_errors = {}
    total = len(chunks)

    print(f"Starting extraction artifact build: document_id={document_id} chunks={total}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{total}] Extracting chunk {chunk.id} ...")
        try:
            raw_extraction = await extractor.extract(text=chunk.text, schema=schema)
            validated = validator.validate(raw_extraction, schema)
            chunk_extractions[chunk.id] = validated
            print(
                f"  [{i}/{total}] extracted "
                f"({len(validated.nodes)} nodes, {len(validated.relationships)} rels)"
            )
        except Exception as exc:
            extraction_errors[chunk.id] = exc
            print(f"  [{i}/{total}] failed: {exc}")

    if chunks and not chunk_extractions:
        first_chunk_id, first_error = next(iter(extraction_errors.items()))
        raise RuntimeError(
            f"Extraction failed for all {len(chunks)} chunk(s). "
            f"First failure was for {first_chunk_id}: {first_error}"
        ) from first_error

    return GraphDocumentAssembler().assemble(
        document_id=document_id,
        text_hash=text_hash,
        chunks=chunks,
        chunk_extractions=chunk_extractions,
        metadata=metadata,
        graph_name=graph_name,
    )


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def write_graph_document_for_ingest(
    store,
    graph_document: GraphDocument,
    index_manager_cls,
    create_indexes: bool = True,
) -> dict:
    """Write graph data for one backend without running post-write steps."""
    if create_indexes:
        index_manager_cls(store, embedding_dim=EMBEDDING_DIM).create_indexes()
    write_stats = store.write_graph_document(graph_document)
    print(f"Write complete: {write_stats}")
    return write_stats


async def finalize_graph_ingest(
    store,
    graph_document: GraphDocument,
    embedder,
    llm=None,
    entity_resolution_strategy: str = "normalized",
    run_entity_resolution: bool = True,
    embed_entities: bool = True,
    allow_ai_auto_merge: bool = False,
) -> dict:
    """Run post-write maintenance: backfill, resolve, embed, validate.

    Entity resolution is delegated to the graph store's strategy-specific
    methods (resolve_entities_exact, resolve_entities_normalized,
    resolve_entities_fuzzy, resolve_entities_hybrid) based on
    `entity_resolution_strategy`.
    """
    store.backfill_descriptions()
    if run_entity_resolution:
        print("Resolving duplicate entities after all selected graph ingests")
        if entity_resolution_strategy == "exact":
            resolution = await store.resolve_entities_exact(
                graph_name=graph_document.document.graph_name,
            )
        elif entity_resolution_strategy == "normalized":
            resolution = await store.resolve_entities_normalized(
                graph_name=graph_document.document.graph_name,
            )
        elif entity_resolution_strategy == "fuzzy":
            resolution = await store.resolve_entities_fuzzy(
                graph_name=graph_document.document.graph_name,
            )
        elif entity_resolution_strategy == "hybrid":
            resolution = await store.resolve_entities_hybrid(
                graph_name=graph_document.document.graph_name,
                embedder=embedder,
                llm=llm,
                allow_ai_auto_merge=allow_ai_auto_merge,
            )
        else:
            raise ValueError(f"Unknown entity resolution strategy: {entity_resolution_strategy}")
        print(f"Entity resolution result: {resolution}")
    else:
        resolution = {"skipped": True, "reason": "disabled"}
        print("Entity resolution skipped")

    if embed_entities:
        await EntityEmbedder(store, embedder).embed_entities()

    validation = store.validate_graph_build()
    print(f"Validation: {validation}")
    return {
        "entity_resolution": resolution,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


async def run_movie_search_suite(
    search_instances: dict[str, Any],
    test_queries: list[dict[str, Any]],
    modes_filter: list[str] | None = None,
) -> None:
    """Run the shared movie query suite against configured search instances."""
    for item in test_queries:
        modes = item.get("modes", ["local", "global", "drift"])
        if modes_filter:
            modes = [mode for mode in modes if mode in modes_filter]

        print("\n" + "=" * 60)
        print(f"QUERY: {item['query']}")
        print(f"MODES: {', '.join(modes)}")
        print(f"OBJECTIVE: {item['test_objective']}")
        print("=" * 60)

        for mode in modes:
            if mode not in SEARCH_OPTIONS:
                print(f"\n>>> [{mode.upper()} ERROR]: Unknown mode.")
                continue
            search = search_instances.get(mode)
            if search is None:
                print(f"\n>>> [{mode.upper()} ERROR]: No search instance for mode.")
                continue
            try:
                result = await search.search(
                    item["query"],
                    **SEARCH_OPTIONS[mode],
                )
                print(f"\n>>> [{mode.upper()} ANSWER]:\n{result.answer}")
            except Exception as exc:
                print(f"\n>>> [{mode.upper()} ERROR]: {exc}")
