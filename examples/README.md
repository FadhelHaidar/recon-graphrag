# Example

This directory contains an end-to-end movie-domain GraphRAG workflow:

```bash
python extract.py
python ingest.py --backend neo4j
python communities.py --backend neo4j
python search.py --backend neo4j
```
