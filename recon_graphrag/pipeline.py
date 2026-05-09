"""Knowledge graph construction pipeline.

Ingests text via SimpleKGPipeline (neo4j-graphrag), then automatically
runs entity resolution and embedding (steps 1-3 of the full pipeline).

Steps 4-6 (community detection, summarization, community embedding)
are handled separately by the CommunityPipeline — typically on a schedule.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.schema import GraphSchema

from recon_graphrag.graph_store import GraphStore
from recon_graphrag.indexes import IndexManager
from recon_graphrag.communities.embeddings import CommunityEmbedder


class GraphBuilderPipeline:
    """Build a knowledge graph from text using LLM entity extraction."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        embedder: Embedder,
        schema: GraphSchema,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.schema = schema
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def build_from_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> dict:
        """Build knowledge graph from raw text.

        Uses SimpleKGPipeline's character-level chunking, then automatically:
          - Step 2: Entity resolution (merge duplicates)
          - Step 3: Entity embedding (for local/DRIFT search)

        Steps 4-6 must be run separately via CommunityPipeline.
        """
        pipeline = self._build_pipeline()
        result = await pipeline.run_async(text=text, document_metadata=metadata)

        await self._resolve_entities()
        await self._embed_entities()

        return result

    def _build_pipeline(self) -> SimpleKGPipeline:
        neo4j_database = getattr(self.graph_store, "_database", None)
        return SimpleKGPipeline(
            llm=self.llm,
            driver=self.graph_store.driver,
            embedder=self.embedder,
            schema=self.schema,
            from_file=False,
            perform_entity_resolution=False,
            neo4j_database=neo4j_database,
        )

    async def _resolve_entities(self):
        """Step 2: Merge duplicate entities via SinglePropertyExactMatchResolver."""
        mgr = IndexManager(self.graph_store)
        try:
            await mgr.resolve_entities()
        except Exception as e:
            print(f"  Warning: entity resolution failed (APOC plugin required): {e}")

    async def _embed_entities(self):
        """Step 3: Generate vector embeddings for entity nodes."""
        embedder = CommunityEmbedder(self.graph_store, self.embedder)
        try:
            await embedder.embed_entities()
        except Exception as e:
            print(f"  Warning: entity embedding failed: {e}")
