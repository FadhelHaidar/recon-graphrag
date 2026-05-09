"""Graph building pipeline: ingestion + community detection.

Run this to populate the graph and build communities.

Usage:
  python build_graph.py              # Full pipeline
  python build_graph.py --indexes    # Drop and recreate indexes only
"""

import asyncio
import sys

from recon_graphrag import GraphBuilderPipeline, IndexManager, CommunityPipeline

from config import get_neo4j_store, get_llm, get_embedder
from schema import MOVIE_SCHEMA, COMMUNITY_RELATIONSHIP_TYPES
from prompts import COMMUNITY_SUMMARY_PROMPT


async def rebuild_indexes():
    store = get_neo4j_store()
    IndexManager(store, embedding_dim=1536).create_indexes()
    print("Indexes recreated.")


async def build():
    store = get_neo4j_store()
    llm = get_llm()
    embedder = get_embedder()

    # Create indexes (run once)
    IndexManager(store, embedding_dim=1536).create_indexes()

    # Ingest
    pipeline = GraphBuilderPipeline(store, llm, embedder, schema=MOVIE_SCHEMA)
    text = (
        # --- Expanding the Nolan Universe (Interconnectivity) ---
        "Christopher Nolan also directed Interstellar (2010), featuring Anne Hathaway "
        "and Matthew McConaughey. Anne Hathaway previously collaborated with Nolan "
        "in The Dark Knight Rises (2012), where she played Catwoman. Warner Bros "
        "distributed both Interstellar and the Dark Knight trilogy. "

        # --- Introducing Theme-based Links (Semantic Traversal) ---
        "Interstellar and Inception are both noted for their soundtracks composed by Hans Zimmer. "
        "Zimmer’s work often utilizes ‘Shepard tones’ to create tension, a technique "
        "also found in Dunkirk (2017), another Nolan film produced by Emma Thomas. "

        # --- Cross-Director & Industry Links (Testing Pathfinding) ---
        "Leonardo DiCaprio, who starred in Inception, also led Killers of the Flower Moon (2023), "
        "directed by Martin Scorsese. Scorsese is a known admirer of Bong Joon-ho, "
        "comparing the tension in Parasite to the works of Alfred Hitchcock. "
        "Parasite made history by being the first non-English language film to win "
        "the Oscar for Best Picture, a category where it beat Sam Mendes’s 1917. "

        # --- Technical & Award Context (Attribute nodes) ---
        "Hoyte van Hoytema served as the cinematographer for Interstellar and Oppenheimer (2023). "
        "Oppenheimer won the Oscar for Best Picture in 2024, mirroring the critical success "
        "of Parasite. Both films deal with high-stakes systemic conflicts, though "
        "in vastly different historical contexts. "

        # --- The "Actor Bridge" (Connecting Nolan, Villeneuve, and Iñárritu) ---
        "Tom Hardy, who played Eames in Inception, starred alongside Leonardo DiCaprio "
        "again in The Revenant (2015). The Revenant was directed by Alejandro G. Iñárritu, "
        "whose film Birdman (2014) won the Oscar for Best Picture. Tom Hardy also "
        "portrayed Bane in The Dark Knight Rises, linking him back to Nolan’s Batman trilogy. "

        # --- The "Sci-Fi Pedigree" (Connecting Interstellar to Dune) ---
        "Denis Villeneuve directed Dune (2021), which features a musical score by Hans Zimmer, "
        "further cementing Zimmer’s influence on modern sci-fi alongside his work on Interstellar. "
        "Timothée Chalamet, the lead in Dune, played the younger version of Casey Affleck’s "
        "character in Interstellar, creating a direct cast link between the two space epics. "

        # --- The "Frequent Collaborator" Hub (Cillian Murphy) ---
        "Cillian Murphy is a central node in the Nolan multiverse, appearing in Inception as "
        "Robert Fischer and starring as J. Robert Oppenheimer in Oppenheimer (2023). "
        "Outside of Nolan’s films, Murphy starred in the series Peaky Blinders, which "
        "also featured Tom Hardy, creating a television-to-film relationship bridge. "

        # --- The "Studio Rivalry & Success" (A24 vs. Major Studios) ---
        "While Warner Bros produced Inception, the independent studio A24 gained prominence "
        "with Everything Everywhere All At Once (2022). That film swept the Oscars in 2023, "
        "much like Parasite did in 2020. Michelle Yeoh, the star of Everything Everywhere, "
        "previously appeared in Danny Boyle’s Sunshine (2007) alongside Cillian Murphy. "

        # --- Technical Aesthetics (IMAX and Practical Effects) ---
        "Nolan and Villeneuve are both proponents of IMAX cinematography. While "
        "Hoyte van Hoytema shot Oppenheimer on IMAX film, Greig Fraser used IMAX digital "
        "for Dune. Both cinematographers focus on ‘tactile’ filmmaking, a style "
        "often contrasted with the heavy CGI usage in Disney’s Marvel Cinematic Universe."
    )
    result = await pipeline.build_from_text(text, metadata={"source": "example"})
    print(f"Ingestion result: {result}")

    # Build communities
    community = CommunityPipeline(
        store, llm, embedder,
        relationship_types=COMMUNITY_RELATIONSHIP_TYPES,
        summary_prompt=COMMUNITY_SUMMARY_PROMPT,
    )
    comm_result = await community.build()
    print(f"Community result: {comm_result}")


if __name__ == "__main__":
    if "--indexes" in sys.argv:
        asyncio.run(rebuild_indexes())
    else:
        asyncio.run(build())
