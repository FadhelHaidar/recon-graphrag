"""Tests for internal hybrid entity retrieval."""

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.hybrid import HybridEntityRetriever, merge_hybrid_scores
from recon_graphrag.retrieval.local import LocalSearchRetriever


class FakeGraphStore:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append((query.strip(), params))
        if "db.index.vector.queryNodes" in query:
            return [
                {"id": "a", "score": 0.7},
                {"id": "b", "score": 0.5},
            ]
        if "db.index.fulltext.queryNodes" in query:
            return [
                {"id": "a", "score": 1.0},
                {"id": "c", "score": 0.4},
            ]
        if "UNWIND $matches AS match" in query:
            return [
                {
                    "title": "Alice (Person)",
                    "relationships": ["Person: Alice -[DIRECTED]-> Movie: Inception"],
                    "source_text": ["Alice directed Inception."],
                    "score": params["matches"][0]["score"],
                    "custom": params.get("custom"),
                }
            ]
        return []

    @property
    def driver(self):
        return None


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    async def async_embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(content="Alice directed Inception.")


def test_merge_hybrid_scores_naive_matches_neo4j_graphrag_ranker():
    merged = merge_hybrid_scores(
        vector_rows=[{"id": "a", "score": 0.7}, {"id": "b", "score": 0.5}],
        keyword_rows=[{"id": "a", "score": 1.0}, {"id": "c", "score": 0.4}],
        top_k=10,
    )

    assert merged == [
        {"id": "a", "score": 1.0},
        {"id": "b", "score": 0.7142857142857143},
        {"id": "c", "score": 0.4},
    ]


def test_merge_hybrid_scores_linear_matches_neo4j_graphrag_ranker():
    merged = merge_hybrid_scores(
        vector_rows=[{"id": "a", "score": 0.7}, {"id": "b", "score": 0.5}],
        keyword_rows=[{"id": "a", "score": 1.0}, {"id": "c", "score": 0.4}],
        top_k=10,
        ranker="linear",
        alpha=0.25,
    )

    assert merged == [
        {"id": "a", "score": 1.0},
        {"id": "c", "score": 0.30000000000000004},
        {"id": "b", "score": 0.17857142857142858},
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ranker": "bogus"}, "Invalid ranker value"),
        ({"ranker": "linear"}, "alpha must be provided"),
        ({"ranker": "linear", "alpha": 1.5}, "alpha must be between 0 and 1"),
        ({"top_k": 0}, "top_k must be a positive integer"),
    ],
)
def test_merge_hybrid_scores_validates_ranker_inputs(kwargs, message):
    params = {"top_k": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        merge_hybrid_scores(
            vector_rows=[{"id": "a", "score": 1.0}],
            keyword_rows=[],
            **params,
        )


@pytest.mark.asyncio
async def test_hybrid_entity_retriever_embeds_queries_and_fetches_context():
    store = FakeGraphStore()
    embedder = FakeEmbedder()
    retriever = HybridEntityRetriever(
        graph_store=store,
        embedder=embedder,
        retrieval_query="RETURN node.name AS title, score",
        vector_index_name="entity-embeddings",
        fulltext_index_name="entity-names",
    )

    result = await retriever.search(
        "Who directed Inception?",
        top_k=2,
        effective_search_ratio=3,
        query_params={"custom": "value"},
    )

    assert embedder.queries == ["Who directed Inception?"]
    assert result.items[0].content["title"] == "Alice (Person)"
    assert result.items[0].content["custom"] == "value"
    vector_params = [
        params
        for query, params in store.calls
        if "db.index.vector.queryNodes" in query
    ][0]
    keyword_params = [
        params
        for query, params in store.calls
        if "db.index.fulltext.queryNodes" in query
    ][0]
    assert vector_params["k"] == 6
    assert keyword_params["k"] == 6
    context_params = [
        params for query, params in store.calls if "UNWIND $matches AS match" in query
    ][0]
    assert context_params["matches"] == [
        {"id": "a", "score": 1.0},
        {"id": "b", "score": 0.7142857142857143},
    ]
    assert result.metadata == {"query_vector": [0.1, 0.2, 0.3]}


@pytest.mark.asyncio
async def test_hybrid_entity_retriever_accepts_precomputed_query_vector():
    store = FakeGraphStore()
    embedder = FakeEmbedder()
    retriever = HybridEntityRetriever(
        graph_store=store,
        embedder=embedder,
        retrieval_query="RETURN node.name AS title, score",
        vector_index_name="entity-embeddings",
        fulltext_index_name="entity-names",
    )

    result = await retriever.search(
        "Who directed Inception?",
        query_vector=[9.0, 8.0, 7.0],
        top_k=1,
    )

    assert embedder.queries == []
    assert result.metadata == {"query_vector": [9.0, 8.0, 7.0]}


@pytest.mark.asyncio
async def test_local_search_uses_internal_retriever_and_llm():
    store = FakeGraphStore()
    embedder = FakeEmbedder()
    llm = FakeLLM()
    retriever = LocalSearchRetriever(store, llm, embedder)

    result = await retriever.search("Who directed Inception?", top_k=2)

    assert result.mode == "local"
    assert result.answer == "Alice directed Inception."
    assert "Finding: Alice (Person)" in result.context
    assert "Alice directed Inception." in llm.prompts[0]
