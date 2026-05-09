"""Movie industry domain schema for film analysis and recommendations."""

from recon_graphrag.schema import GraphSchema, NodeType, PropertyType, RelationshipType

MOVIE_SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Movie",
            description=(
                "A film or motion picture. "
                "E.g. The Shawshank Redemption, Inception, Parasite."
            ),
            properties=[
                PropertyType(name="title", type="STRING"),
                PropertyType(name="year", type="STRING"),
                PropertyType(name="genre", type="STRING"),
            ],
        ),
        NodeType(
            label="Person",
            description=(
                "An individual involved in the film industry. "
                "Can be a director, actor, writer, or producer."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="role", type="STRING"),
            ],
        ),
        NodeType(
            label="Studio",
            description="A production studio or distribution company. E.g. Warner Bros, A24, Neon.",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Award",
            description="An award or nomination. E.g. Oscar, Palme d'Or, Golden Globe.",
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="category", type="STRING"),
                PropertyType(name="year", type="STRING"),
            ],
        ),
        NodeType(
            label="Theme",
            description="A narrative theme or motif. E.g. redemption, time manipulation, social inequality.",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Location",
            description="A filming location or story setting. E.g. Los Angeles, Tokyo, outer space.",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Franchise",
            description="A film series or shared universe. E.g. Marvel Cinematic Universe, Dune series.",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Review",
            description="A critical or audience review with rating and sentiment.",
            properties=[
                PropertyType(name="source", type="STRING"),
                PropertyType(name="score", type="STRING"),
                PropertyType(name="verdict", type="STRING"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(label="DIRECTED", description="Person directed a movie"),
        RelationshipType(label="ACTED_IN", description="Person acted in a movie"),
        RelationshipType(label="PRODUCED", description="Studio produced a movie"),
        RelationshipType(label="WON_AWARD", description="Movie won or was nominated for an award"),
        RelationshipType(label="EXPLORES", description="Movie explores a theme"),
        RelationshipType(label="SET_IN", description="Movie is set in or filmed at a location"),
        RelationshipType(label="BELONGS_TO", description="Movie belongs to a franchise"),
        RelationshipType(label="REVIEWED", description="Movie received a review"),
        RelationshipType(label="COLLABORATED", description="Person collaborated with another person"),
        RelationshipType(label="SIMILAR_TO", description="Movie is similar to another movie"),
    ],
    patterns=[
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
        ("Studio", "PRODUCED", "Movie"),
        ("Movie", "WON_AWARD", "Award"),
        ("Movie", "EXPLORES", "Theme"),
        ("Movie", "SET_IN", "Location"),
        ("Movie", "BELONGS_TO", "Franchise"),
        ("Movie", "REVIEWED", "Review"),
        ("Person", "COLLABORATED", "Person"),
        ("Movie", "SIMILAR_TO", "Movie"),
    ],
)

# Relationship types for community detection projection
COMMUNITY_RELATIONSHIP_TYPES = [
    "DIRECTED",
    "ACTED_IN",
    "PRODUCED",
    "WON_AWARD",
    "EXPLORES",
    "SET_IN",
    "BELONGS_TO",
    "COLLABORATED",
]
