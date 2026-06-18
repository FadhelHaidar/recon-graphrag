"""Movie industry domain schema for film analysis and recommendations."""

from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType, RelationshipType

MOVIE_SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Movie",
            description=(
                "A film or motion picture. "
                "E.g. Inception, Interstellar, Parasite, Oppenheimer, Dune."
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
                "Can be a director, actor, writer, producer, composer, or cinematographer."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="role", type="STRING"),
            ],
        ),
        NodeType(
            label="Studio",
            description=(
                "A production studio or distribution company. "
                "E.g. Warner Bros, A24, Universal Pictures, Disney."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="type", type="STRING"),  # production, distribution, major studio, independent studio
            ],
        ),
        NodeType(
            label="Award",
            description=(
                "An award, award category, or nomination. "
                "E.g. Oscar for Best Picture, Best Director, Palme d'Or."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="category", type="STRING"),
                PropertyType(name="year", type="STRING"),
            ],
        ),
        NodeType(
            label="Theme",
            description=(
                "A narrative theme or motif. "
                "E.g. time dilation, moral responsibility, class inequality, survival, empire."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
            ],
        ),
        NodeType(
            label="Technique",
            description=(
                "A cinematic, musical, visual, or storytelling technique. "
                "E.g. IMAX cinematography, Shepard tones, non-linear editing, practical effects."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="category", type="STRING"),  # music, cinematography, editing, visual effects
            ],
        ),
        NodeType(
            label="Character",
            description=(
                "A fictional character portrayed by an actor. "
                "E.g. Catwoman, Bane, Robert Fischer, J. Robert Oppenheimer."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="alias", type="STRING"),
            ],
        ),
        NodeType(
            label="Location",
            description=(
                "A filming location or story setting. "
                "E.g. outer space, Dunkirk, Los Angeles, Arrakis."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
            ],
        ),
        NodeType(
            label="Franchise",
            description=(
                "A film series, shared universe, or long-running media property. "
                "E.g. The Dark Knight trilogy, Marvel Cinematic Universe, Dune series."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
            ],
        ),
        NodeType(
            label="TVSeries",
            description=(
                "A television series connected to film actors, directors, or franchises. "
                "E.g. Peaky Blinders."
            ),
            properties=[
                PropertyType(name="title", type="STRING"),
                PropertyType(name="year", type="STRING"),
            ],
        ),
        NodeType(
            label="HistoricalEvent",
            description=(
                "A real historical event represented or referenced by a film. "
                "E.g. World War II, Dunkirk evacuation, Manhattan Project."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="period", type="STRING"),
            ],
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
        NodeType(
            label="Genre",
            description="Film categories like Sci-Fi, War, Drama, Thriller, Comedy, or Historical Drama.",
            properties=[
                PropertyType(name="name", type="STRING"),
            ],
        ),
        NodeType(
            label="Occupation",
            description=(
                "Professional roles in the film industry. "
                "E.g. Director, Actor, Composer, Cinematographer, Producer."
            ),
            properties=[
                PropertyType(name="name", type="STRING"),
            ],
        ),
    ],

    relationship_types=[
        RelationshipType(
            label="DIRECTED",
            description="Person directed a movie or television series",
        ),
        RelationshipType(
            label="ACTED_IN",
            description="Person acted in a movie or television series",
        ),
        RelationshipType(
            label="PLAYED_CHARACTER",
            description="Person portrayed a specific character",
        ),
        RelationshipType(
            label="PRODUCED",
            description="Person or studio produced a movie",
        ),
        RelationshipType(
            label="DISTRIBUTED",
            description="Studio distributed a movie",
        ),
        RelationshipType(
            label="WON_AWARD",
            description="Movie or person won an award",
        ),
        RelationshipType(
            label="NOMINATED_FOR",
            description="Movie or person was nominated for an award",
        ),
        RelationshipType(
            label="EXPLORES",
            description="Movie explores a theme",
        ),
        RelationshipType(
            label="USES_TECHNIQUE",
            description="Movie uses a cinematic, musical, or storytelling technique",
        ),
        RelationshipType(
            label="SET_IN",
            description="Movie is set in or filmed at a location",
        ),
        RelationshipType(
            label="DEPICTS",
            description="Movie depicts or references a historical event",
        ),
        RelationshipType(
            label="BELONGS_TO",
            description="Movie belongs to a franchise or film series",
        ),
        RelationshipType(
            label="REVIEWED",
            description="Movie received a review",
        ),
        RelationshipType(
            label="COLLABORATED",
            description="Person collaborated with another person",
        ),
        RelationshipType(
            label="SIMILAR_TO",
            description="Movie is similar to another movie",
        ),
        RelationshipType(
            label="COMPOSED_MUSIC",
            description="Person composed the musical score for a movie",
        ),
        RelationshipType(
            label="SHOT_BY",
            description="Person served as cinematographer for a movie",
        ),
        RelationshipType(
            label="HAS_GENRE",
            description="Movie belongs to a specific genre",
        ),
        RelationshipType(
            label="HAS_OCCUPATION",
            description="Person has a professional occupation",
        ),
        RelationshipType(
            label="CONNECTED_TO",
            description="Generic connection between entities",
        ),
        RelationshipType(
            label="BASED_ON",
            description="Movie is based on a book, novel, historical event, or source material",
        ),
        RelationshipType(
            label="INFLUENCED_BY",
            description="Person, movie, or style was influenced by another person, movie, or style",
        ),
        RelationshipType(
            label="COMPARED_TO",
            description="Person, movie, or style is compared to another person, movie, or style",
        )
    ],

    patterns=[
        # --- Core film industry patterns ---
        ("Person", "DIRECTED", "Movie"),
        ("Person", "ACTED_IN", "Movie"),
        ("Person", "PLAYED_CHARACTER", "Character"),
        ("Character", "CONNECTED_TO", "Movie"),

        # --- Studio and production patterns ---
        ("Studio", "PRODUCED", "Movie"),
        ("Studio", "DISTRIBUTED", "Movie"),
        ("Person", "PRODUCED", "Movie"),

        # --- Award patterns ---
        ("Movie", "WON_AWARD", "Award"),
        ("Person", "WON_AWARD", "Award"),
        ("Movie", "NOMINATED_FOR", "Award"),
        ("Person", "NOMINATED_FOR", "Award"),

        # --- Semantic and thematic retrieval patterns ---
        ("Movie", "EXPLORES", "Theme"),
        ("Movie", "USES_TECHNIQUE", "Technique"),
        ("Movie", "HAS_GENRE", "Genre"),
        ("Movie", "SIMILAR_TO", "Movie"),
        ("Movie", "CONNECTED_TO", "Movie"),

        # --- Location, setting, and history patterns ---
        ("Movie", "SET_IN", "Location"),
        ("Movie", "DEPICTS", "HistoricalEvent"),
        ("Movie", "BASED_ON", "HistoricalEvent"),

        # --- Franchise and source-material patterns ---
        ("Movie", "BELONGS_TO", "Franchise"),
        ("Movie", "BASED_ON", "Franchise"),

        # --- Review pattern ---
        ("Movie", "REVIEWED", "Review"),

        # --- Technical and creative role patterns ---
        ("Person", "COMPOSED_MUSIC", "Movie"),
        ("Person", "SHOT_BY", "Movie"),
        ("Person", "HAS_OCCUPATION", "Occupation"),

        # --- Collaboration patterns ---
        ("Person", "COLLABORATED", "Person"),
        ("Person", "CONNECTED_TO", "Person"),

        # --- TV-to-film bridge patterns ---
        ("Person", "ACTED_IN", "TVSeries"),
        ("Person", "DIRECTED", "TVSeries"),
        ("TVSeries", "CONNECTED_TO", "Movie"),

        # --- Influence and comparison patterns ---
        ("Person", "INFLUENCED_BY", "Person"),
        ("Movie", "INFLUENCED_BY", "Movie"),
        ("Movie", "COMPARED_TO", "Movie"),
        ("Person", "COMPARED_TO", "Person"),
    ],
)

# Relationship types for community detection projection
COMMUNITY_RELATIONSHIP_TYPES = [
    # Strong person-movie links
    "DIRECTED",
    "ACTED_IN",
    "PLAYED_CHARACTER",

    # Creative/technical collaborator links
    "COMPOSED_MUSIC",
    "SHOT_BY",
    "PRODUCED",

    # Studio/franchise glue
    "DISTRIBUTED",
    "BELONGS_TO",

    # Semantic clustering
    "HAS_GENRE",
    "EXPLORES",
    "USES_TECHNIQUE",
    "SIMILAR_TO",

    # Award-based clustering
    "WON_AWARD",
    "NOMINATED_FOR",

    # Cross-domain bridge links
    "COLLABORATED",
    "CONNECTED_TO",
    "INFLUENCED_BY",
    "COMPARED_TO",
]
