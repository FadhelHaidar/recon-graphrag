"""Internal hybrid entity retrieval for local and DRIFT search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore


HybridRanker = Literal["naive", "linear"]


@dataclass
class RetrievalItem:
    """One retrieved context payload."""

    content: dict | str


@dataclass
class RetrievalResult:
    """Retriever result compatible with the previous local formatting code."""

    items: list[RetrievalItem]
    metadata: dict | None = None


class HybridEntityRetriever:
    """Hybrid vector + keyword retrieval over entity nodes.

    Ranking mirrors neo4j-graphrag hybrid search:
    source scores are normalized independently. The "naive" ranker uses the
    max normalized score per node, and "linear" uses an alpha-weighted sum.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: BaseEmbedder,
        retrieval_query: str,
        vector_index_name: str,
        fulltext_index_name: str,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.retrieval_query = retrieval_query
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name

    async def search(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        top_k: int = 10,
        effective_search_ratio: int = 1,
        query_params: dict | None = None,
        ranker: HybridRanker | str = "naive",
        alpha: float | None = None,
    ) -> RetrievalResult:
        ranker = validate_hybrid_ranker(ranker, alpha)
        validate_positive_int("top_k", top_k)
        validate_positive_int("effective_search_ratio", effective_search_ratio)

        if query_vector is None:
            query_vector = await self.embedder.async_embed_query(query_text)

        candidate_k = top_k * effective_search_ratio
        vector_rows = self._vector_search(query_vector, candidate_k, top_k)
        keyword_rows = self._keyword_search(query_text, candidate_k, top_k)
        matches = merge_hybrid_scores(
            vector_rows,
            keyword_rows,
            top_k,
            ranker=ranker,
            alpha=alpha,
        )
        if not matches:
            return RetrievalResult(items=[], metadata={"query_vector": query_vector})

        context_rows = self._fetch_context(matches, query_params=query_params)
        return RetrievalResult(
            items=[RetrievalItem(content=row) for row in context_rows],
            metadata={"query_vector": query_vector},
        )

    def _vector_search(
        self,
        query_vector: list[float],
        candidate_k: int,
        top_k: int,
    ) -> list[dict]:
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
        YIELD node, score
        WHERE node:__Entity__
        RETURN elementId(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.graph_store.execute_query(
            query,
            {
                "index_name": self.vector_index_name,
                "k": candidate_k,
                "top_k": top_k,
                "query_vector": query_vector,
            },
        )

    def _keyword_search(
        self,
        query_text: str,
        candidate_k: int,
        top_k: int,
    ) -> list[dict]:
        query = """
        CALL db.index.fulltext.queryNodes(
            $index_name,
            $query_text,
            {limit: $k}
        )
        YIELD node, score
        WHERE node:__Entity__
        RETURN elementId(node) AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.graph_store.execute_query(
            query,
            {
                "index_name": self.fulltext_index_name,
                "query_text": query_text,
                "k": candidate_k,
                "top_k": top_k,
            },
        )

    def _fetch_context(
        self,
        matches: list[dict],
        query_params: dict | None = None,
    ) -> list[dict]:
        query = f"""
        UNWIND $matches AS match
        MATCH (node)
        WHERE elementId(node) = match.id
        WITH node, match.score AS score
        {self.retrieval_query}
        """
        parameters = {"matches": matches}
        if query_params:
            for key, value in query_params.items():
                parameters.setdefault(key, value)
        return self.graph_store.execute_query(query, parameters)


def merge_hybrid_scores(
    vector_rows: list[dict],
    keyword_rows: list[dict],
    top_k: int,
    ranker: HybridRanker | str = "naive",
    alpha: float | None = None,
) -> list[dict]:
    """Merge vector and keyword scores with neo4j-graphrag-compatible ranking."""
    ranker = validate_hybrid_ranker(ranker, alpha)
    validate_positive_int("top_k", top_k)

    vector_scores = _normalized_scores(vector_rows)
    keyword_scores = _normalized_scores(keyword_rows)
    ids = set(vector_scores) | set(keyword_scores)
    scores: dict[str, float] = {}
    for entity_id in ids:
        vector_score = vector_scores.get(entity_id, 0.0)
        keyword_score = keyword_scores.get(entity_id, 0.0)
        if ranker == "linear":
            scores[entity_id] = (alpha or 0.0) * vector_score + (
                1 - (alpha or 0.0)
            ) * keyword_score
        else:
            scores[entity_id] = max(vector_score, keyword_score)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [{"id": entity_id, "score": score} for entity_id, score in ranked[:top_k]]


def _normalized_scores(rows: list[dict]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        entity_id = row.get("id")
        if entity_id is not None:
            value = float(row.get("score", 0.0))
            values[str(entity_id)] = max(values.get(str(entity_id), 0.0), value)

    max_score = max(values.values(), default=0.0)
    if max_score <= 0.0:
        return {entity_id: 0.0 for entity_id in values}
    return {entity_id: score / max_score for entity_id, score in values.items()}


def validate_hybrid_ranker(ranker: HybridRanker | str, alpha: float | None) -> HybridRanker:
    normalized = str(ranker).lower()
    if normalized not in {"naive", "linear"}:
        raise ValueError("Invalid ranker value. Allowed values are: naive, linear.")

    if normalized == "linear":
        if alpha is None:
            raise ValueError("alpha must be provided when using the linear ranker")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

    return normalized  # type: ignore[return-value]


def validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
