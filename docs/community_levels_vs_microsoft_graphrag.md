# Community Levels vs Microsoft GraphRAG

## Summary

Recon-GraphRAG follows the same broad architecture as Microsoft GraphRAG's "From Local to Global" pattern:

```text
source documents
-> text chunks
-> entity and relationship extraction
-> entity graph
-> hierarchical community detection
-> community summaries at each level
-> local / global / DRIFT-style retrieval
```

The important difference is the current **community level numbering convention**.

## Microsoft GraphRAG Pattern

The Microsoft GraphRAG paper describes a graph index built in two major stages:

1. Extract an entity knowledge graph from source documents.
2. Detect hierarchical communities and pre-generate summaries for those communities.

At query time:

- **Local search** answers specific questions by retrieving relevant entities, their neighbors, and source text.
- **Global search** answers broad sensemaking questions by using precomputed community summaries.

This codebase follows that same pattern at the architecture level.

## Recon-GraphRAG Flow

Current Recon-GraphRAG flow:

```text
GraphBuilderPipeline
-> entity graph
-> CommunityDetector using Leiden/GDS
-> CommunitySummarizer
-> CommunityEmbedder
-> LocalSearchRetriever / GlobalSearchRetriever / DriftSearchRetriever
```

This aligns with the "local to global" design:

- Entity graph supports local retrieval.
- Community hierarchy supports global retrieval.
- Community summaries provide higher-level context.
- DRIFT combines entity-level and community-level context.

## Level Numbering Difference

In many Microsoft GraphRAG descriptions, level notation is commonly interpreted as:

```text
C0 = root / coarsest / most global level
C1 = less coarse
C2 = finer
C3 = finest / most local level
```

In the current Recon-GraphRAG implementation, the convention is the opposite:

```text
level 0 = finest / most local communities
level 1 = parent / broader communities
level 2 = even broader communities
highest level = coarsest / most global communities
```

This comes from `CommunityDetector._normalize_community_path()` and `_write_community_hierarchy()`, where the community path is treated as finest-to-coarsest and then enumerated:

```python
for level, community_id in enumerate(path):
    ...
```

## Practical Mapping

Use this mapping when comparing to Microsoft GraphRAG terminology:

```text
Recon level 0       ~= Microsoft finest / deepest community level
Recon highest level ~= Microsoft C0 / root / coarsest community level
```

So:

- Use `level=0` for detailed, fine-grained community summaries.
- Use a higher level for broader, more global summaries.
- Use `level=None` when searching across all available community levels.

## Retrieval Implication

`GlobalSearchRetriever` currently defaults to:

```python
level=None
```

That means it searches across all community levels, which is a pragmatic default.

However, explicitly passing:

```python
level=0
```

does **not** mean "most global" in this codebase. It means "finest / most local community summaries."

For Microsoft-style global summaries, prefer the highest available community level.

## Implemented Selection API

Keep the current stored implementation because the codebase already depends on `level 0 = finest`.

Global and DRIFT retrieval support semantic level selectors:

```python
community_level="all"
community_level="finest"
community_level="coarsest"
community_level=0
community_level=1
```

Global search also accepts the existing `level=` argument for backward compatibility:

```python
await graphrag.search("What are the major themes?", mode="global", level=0)
await graphrag.search("What are the major themes?", mode="global", community_level="coarsest")
```

DRIFT search accepts `community_level=`:

```python
await graphrag.search(
    "Explain Inception using detailed and broader context",
    mode="drift",
    community_level="all",
)
```

Selector behavior:

```text
"all"      -> no level filter
"finest"   -> level 0
"coarsest" -> highest available level for the graph
int        -> exact stored level
None       -> default behavior
```

For DRIFT, the constructor default remains `0` for backward compatibility. Callers can pass `"all"` or `"coarsest"` when they want broader community context.

## Documentation Recommendation

Document the convention clearly in:

- README
- community pipeline docs
- query examples
- global search examples

This avoids breaking existing graph data while making query intent clearer.

## See also

- [README](../README.md) — project overview and quick start
- [Search](search.md) — how to use community levels in local, global, and DRIFT search
- [Pipelines](pipelines.md) — how communities are built
