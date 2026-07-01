"""Movie industry example corpus with per-document metadata."""

MOVIE_EXAMPLE_PAGES = [
    {
        "text": """
        Christopher Nolan directed Interstellar (2014), a science-fiction film starring
        Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Casey Affleck.
        Anne Hathaway previously collaborated with Nolan in The Dark Knight Rises (2012),
        where she played Selina Kyle, also known as Catwoman. The Dark Knight Rises also
        featured Christian Bale as Batman and Tom Hardy as Bane.

        Warner Bros distributed The Dark Knight trilogy and also distributed Interstellar
        internationally. Nolan frequently collaborated with producer Emma Thomas, who
        produced Interstellar, Inception, Dunkirk, The Dark Knight Rises, and Oppenheimer.
        """,
        "metadata": {
            "source": "nolan-universe",
            "topic": "director-collaborators",
            "page_index": 0,
        },
    },
    {
        "text": """
        Interstellar and Inception are both known for their philosophical science-fiction
        themes and their musical scores composed by Hans Zimmer. Zimmer's score for
        Inception became especially famous for its use of deep brass sounds and auditory
        tension. His work often uses Shepard tones, a musical illusion associated with
        rising tension.

        Shepard tones are also strongly associated with Dunkirk (2017), another Nolan film.
        Dunkirk was directed by Christopher Nolan and produced by Emma Thomas. The film
        uses music, sound design, and non-linear editing to create suspense during the
        evacuation of Allied soldiers from Dunkirk during World War II.
        """,
        "metadata": {
            "source": "theme-music",
            "topic": "music-themes",
            "page_index": 1,
        },
    },
    {
        "text": """
        Leonardo DiCaprio starred in Inception (2010), where he played Dom Cobb, a thief
        who extracts secrets through dream-sharing technology. DiCaprio later starred in
        Killers of the Flower Moon (2023), directed by Martin Scorsese. Scorsese has worked
        with DiCaprio on several films, including The Departed, Shutter Island, The Wolf of
        Wall Street, and Killers of the Flower Moon.

        Martin Scorsese has publicly praised Bong Joon-ho, the director of Parasite (2019).
        Parasite made history by becoming the first non-English-language film to win the
        Oscar for Best Picture. At the 2020 Academy Awards, Parasite won Best Picture,
        Best Director, Best Original Screenplay, and Best International Feature Film.
        Parasite beat other Best Picture nominees including 1917, directed by Sam Mendes.
        """,
        "metadata": {
            "source": "cross-director",
            "topic": "industry-links",
            "page_index": 2,
        },
    },
    {
        "text": """
        Hoyte van Hoytema served as the cinematographer for Interstellar and later worked
        with Christopher Nolan again on Dunkirk, Tenet, and Oppenheimer. Oppenheimer (2023)
        was directed by Christopher Nolan and starred Cillian Murphy as J. Robert Oppenheimer,
        the theoretical physicist associated with the Manhattan Project.

        Oppenheimer won the Oscar for Best Picture at the 2024 Academy Awards. The film also
        won major awards for directing, acting, cinematography, editing, and score. Like
        Parasite, Oppenheimer became a major critical success, although the two films explore
        very different kinds of social and historical conflict. Parasite focuses on class
        inequality, while Oppenheimer focuses on science, war, politics, and moral responsibility.
        """,
        "metadata": {
            "source": "technical-awards",
            "topic": "award-context",
            "page_index": 3,
        },
    },
    {
        "text": """
        Tom Hardy played Eames in Inception, a member of Dom Cobb's dream-heist team.
        Hardy later starred alongside Leonardo DiCaprio in The Revenant (2015). The Revenant
        was directed by Alejandro G. Inarritu and earned DiCaprio the Oscar for Best Actor.

        Alejandro G. Inarritu also directed Birdman (2014), which won the Oscar for Best
        Picture. Tom Hardy also portrayed Bane in The Dark Knight Rises, connecting him back
        to Christopher Nolan's Batman trilogy. Through Hardy and DiCaprio, Inception connects
        to The Revenant, Birdman, and the Academy Awards network.
        """,
        "metadata": {
            "source": "actor-bridge",
            "topic": "nolan-scorsese-inarritu",
            "page_index": 4,
        },
    },
    {
        "text": """
        Denis Villeneuve directed Dune (2021), a science-fiction epic based on the novel by
        Frank Herbert. Dune stars Timothee Chalamet as Paul Atreides, Zendaya as Chani,
        Rebecca Ferguson as Lady Jessica, and Oscar Isaac as Duke Leto Atreides.

        Hans Zimmer composed the score for Dune, further connecting the film to modern
        science-fiction cinema through his previous work on Interstellar and Inception.
        Timothee Chalamet also appeared in Interstellar as the young version of Tom Cooper,
        the son of Matthew McConaughey's character Joseph Cooper. This creates a cast bridge
        between Interstellar and Dune.
        """,
        "metadata": {
            "source": "sci-fi-pedigree",
            "topic": "interstellar-dune-villeneuve",
            "page_index": 5,
        },
    },
    {
        "text": """
        Cillian Murphy is a frequent collaborator in Christopher Nolan's films. He appeared
        as Dr. Jonathan Crane, also known as Scarecrow, in Batman Begins, The Dark Knight,
        and The Dark Knight Rises. He also appeared in Inception as Robert Fischer, the heir
        to a business empire targeted by Dom Cobb's team.

        Murphy later starred as J. Robert Oppenheimer in Oppenheimer (2023), becoming the
        central actor in one of Nolan's most acclaimed films. Outside of Nolan's filmography,
        Murphy starred as Thomas Shelby in the television series Peaky Blinders. Tom Hardy
        also appeared in Peaky Blinders as Alfie Solomons, creating a television-to-film
        relationship bridge between Murphy, Hardy, Nolan, and Inception.
        """,
        "metadata": {
            "source": "collaborator-hub",
            "topic": "cillian-murphy",
            "page_index": 6,
        },
    },
    {
        "text": """
        Warner Bros produced and distributed many major Christopher Nolan films, including
        The Dark Knight trilogy and Inception. However, Nolan later worked with Universal
        Pictures for Oppenheimer after leaving Warner Bros.

        The independent studio A24 gained major prominence with Everything Everywhere All
        At Once (2022), directed by Daniel Kwan and Daniel Scheinert, collectively known as
        Daniels. The film starred Michelle Yeoh, Ke Huy Quan, Stephanie Hsu, and Jamie Lee
        Curtis. Everything Everywhere All At Once won Best Picture at the 2023 Academy Awards,
        similar to how Parasite became a major Oscar success in 2020.

        Michelle Yeoh previously appeared in Sunshine (2007), a science-fiction film directed
        by Danny Boyle. Sunshine also starred Cillian Murphy, creating a link between Michelle
        Yeoh, Cillian Murphy, A24's Oscar success, and Nolan's actor network.
        """,
        "metadata": {
            "source": "studio-oscars",
            "topic": "studio-rivalry-oscar-success",
            "page_index": 7,
        },
    },
    {
        "text": """
        Christopher Nolan and Denis Villeneuve are both known as proponents of large-format
        cinema and immersive theatrical presentation. Nolan frequently uses IMAX film cameras,
        especially in films such as The Dark Knight, Dunkirk, Tenet, and Oppenheimer.

        Hoyte van Hoytema shot Oppenheimer using IMAX film photography, including black-and-white
        IMAX film stock. Greig Fraser served as the cinematographer for Dune and used large-format
        digital cinematography, including IMAX presentation formats. Both Hoyte van Hoytema and
        Greig Fraser are associated with tactile, atmospheric visual styles.

        Nolan's preference for practical effects is often contrasted with the heavy use of CGI
        in large franchise filmmaking, including parts of Disney's Marvel Cinematic Universe.
        Villeneuve's Dune also combines practical sets, location photography, visual effects,
        and large-scale production design to create a grounded science-fiction aesthetic.
        """,
        "metadata": {
            "source": "technical-aesthetics",
            "topic": "imax-practical-effects",
            "page_index": 8,
        },
    },
    {
        "text": """
        Alfred Hitchcock is frequently associated with suspense cinema and psychological tension.
        Bong Joon-ho's Parasite has often been discussed in relation to suspense, class anxiety,
        and tonal shifts. Martin Scorsese has praised Bong Joon-ho's filmmaking, helping connect
        Parasite to a wider network of international cinema and auteur directors.

        Sam Mendes's 1917 also uses tension as a central cinematic device, but its style differs
        from Parasite. 1917 is designed to appear like one continuous shot, while Parasite relies
        on changes in space, social hierarchy, and genre. Both films competed for Best Picture
        at the 2020 Academy Awards.
        """,
        "metadata": {
            "source": "suspense-influence",
            "topic": "hitchcock-suspense",
            "page_index": 9,
        },
    },
    {
        "text": """
        Birdman, Parasite, Everything Everywhere All At Once, and Oppenheimer all won the Oscar
        for Best Picture. Birdman was directed by Alejandro G. Inarritu, Parasite was directed
        by Bong Joon-ho, Everything Everywhere All At Once was directed by Daniels, and
        Oppenheimer was directed by Christopher Nolan.

        These Best Picture winners form an award-based graph that connects directors, studios,
        actors, and themes. Birdman connects to The Revenant through Inarritu. The Revenant
        connects to Inception through Leonardo DiCaprio and Tom Hardy. Parasite connects to
        international cinema and class conflict. Everything Everywhere All At Once connects
        to A24 and Michelle Yeoh. Oppenheimer connects to Nolan, Cillian Murphy, Hoyte van
        Hoytema, and historical drama.
        """,
        "metadata": {
            "source": "award-graph",
            "topic": "best-picture-winners",
            "page_index": 10,
        },
    },
    {
        "text": """
        Leonardo DiCaprio, Tom Hardy, Cillian Murphy, Anne Hathaway, Timothee Chalamet,
        Matthew McConaughey, Michelle Yeoh, and Ke Huy Quan each connect different parts
        of the modern film graph.

        DiCaprio links Inception, The Revenant, Killers of the Flower Moon, and Scorsese's
        filmography. Hardy links Inception, The Dark Knight Rises, The Revenant, and Peaky
        Blinders. Murphy links Batman Begins, Inception, Peaky Blinders, Sunshine, and
        Oppenheimer. Hathaway links Interstellar and The Dark Knight Rises. Chalamet links
        Interstellar and Dune. Yeoh links Sunshine and Everything Everywhere All At Once.
        """,
        "metadata": {
            "source": "actor-web",
            "topic": "actor-collaboration",
            "page_index": 11,
        },
    },
    {
        "text": """
        Several films in this corpus explore time, memory, identity, survival, and moral
        responsibility. Inception explores dreams, memory, guilt, and constructed realities.
        Interstellar explores time dilation, parenthood, survival, and the future of humanity.
        Dunkirk explores survival, fear, and collective rescue during war. Oppenheimer explores
        scientific discovery, political pressure, and ethical responsibility.

        Dune explores empire, ecology, prophecy, and political power. Parasite explores class
        inequality, social mobility, and domestic space. Everything Everywhere All At Once
        explores identity, family, nihilism, and multiverse possibility. These themes allow
        semantic retrieval to connect films even when they do not share the same actors,
        directors, studios, or awards.
        """,
        "metadata": {
            "source": "thematic-expansion",
            "topic": "themes",
            "page_index": 12,
        },
    },
]
