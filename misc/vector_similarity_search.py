# ---
# deploy: false
# lambda-test: false
# ---

# # Vector similarity search with pgvector and sentence-transformers
#
# This example shows how to build a semantic search pipeline on Modal using
# [sentence-transformers](https://www.sbert.net/) for GPU-accelerated embedding
# and [pgvector](https://github.com/pgvector/pgvector) on
# [Neon](https://neon.tech/) for storing and querying vectors.
#
# By the end you will have an end-to-end system that embeds a small text corpus,
# stores those vectors in a Postgres-compatible database, and retrieves the most
# semantically similar documents for any free-text query.

# ## Set up the container Image
#
# All Modal Functions run inside a container, so we need to declare our Python
# dependencies in a Modal [Image](https://modal.com/docs/guide/images).
# `sentence-transformers` gives us the embedding model, and `psycopg2-binary`
# is the Postgres driver we'll use to talk to Neon.

import modal

app = modal.App("example-vector-similarity-search")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "sentence-transformers==3.3.1",
    "psycopg2-binary==2.9.10",
)

# ## Embed text on a GPU
#
# `all-MiniLM-L6-v2` is a compact sentence-transformer model that encodes text
# into 384-dimensional vectors capturing semantic meaning.  Attaching a T4 GPU
# with `gpu="T4"` makes batch encoding significantly faster than CPU — important
# as your corpus grows.
#
# We import `sentence_transformers` inside the Function body rather than at the
# top of the file.  This pattern lets you `import` the module on your local
# machine (where the package may not be installed) without raising an error.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # pin to avoid surprises!
EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2


@app.function(gpu="T4", image=image)
def embed(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    # normalize_embeddings=True makes cosine similarity equivalent to dot product,
    # which pgvector can compute very efficiently.
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ## Provide database credentials via a Secret
#
# We never hard-code credentials.  Instead we read them from a Modal
# [Secret](https://modal.com/docs/guide/secrets) called `neonpgvector`.
# Create that Secret in the Modal dashboard (or with `modal secret create`) and
# add a single key, `DATABASE_URL`, set to your Neon connection string.

DB_SECRET = modal.Secret.from_name("neonpgvector", required_keys=["DATABASE_URL"])

# ## Define the corpus
#
# For this example we use a small hand-picked corpus that spans four distinct
# topics.  In a real application you would load documents from a database, file,
# or API instead.

CORPUS = [
    # Machine learning
    "Gradient descent is an optimization algorithm used to minimize the loss function.",
    "Transformers use self-attention to weigh the importance of different tokens.",
    "Overfitting occurs when a model memorizes training data instead of generalizing.",
    "Batch normalization stabilizes and accelerates neural network training.",
    # Astronomy
    "Black holes are regions of spacetime where gravity prevents anything from escaping.",
    "The James Webb Space Telescope observes infrared light to peer at distant galaxies.",
    "Neutron stars are the remnants of massive stars that exploded as supernovae.",
    "Dark matter is thought to account for roughly 27 percent of the universe's mass.",
    # Cooking
    "The Maillard reaction gives browned food its distinctive flavor and aroma.",
    "Emulsification combines two immiscible liquids, such as oil and water, into a stable mixture.",
    "Fermentation uses microorganisms to transform sugars into alcohol or acids.",
    "Sous vide cooking seals food in a bag and cooks it at a precise water temperature.",
    # Programming languages
    "Python's global interpreter lock prevents true multi-threaded CPU parallelism.",
    "Rust's borrow checker enforces memory safety at compile time without a garbage collector.",
    "Functional programming treats computation as the evaluation of mathematical functions.",
]


# ## Index: embed and store the corpus
#
# The `index` Function embeds every document in the corpus (calling our GPU
# Function via `.remote()`), enables the `pgvector` extension in our Neon
# database, and upserts all vectors into a `documents` table.
#
# Running `index` multiple times is safe — we truncate the table on each run so
# the corpus stays consistent with the source list above.


@app.function(image=image, secrets=[DB_SECRET])
def index():
    import os

    import psycopg2

    print(f"Embedding {len(CORPUS)} documents…")
    vectors = embed.remote(CORPUS)

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    # pgvector must be enabled once per database; CREATE EXTENSION IF NOT EXISTS
    # is idempotent so repeated runs are safe.
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id        SERIAL PRIMARY KEY,
            text      TEXT    NOT NULL,
            embedding VECTOR({EMBEDDING_DIM})
        )
        """
    )

    # Truncate so re-indexing always reflects the current corpus.
    cur.execute("TRUNCATE documents")

    # Insert all rows in one round-trip.  We format each vector as the string
    # literal "[x,y,…]" and rely on Postgres to cast it to the VECTOR type.
    cur.executemany(
        "INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)",
        [
            (text, f"[{','.join(map(str, vec))}]")
            for text, vec in zip(CORPUS, vectors)
        ],
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Indexed {len(CORPUS)} documents.")


# ## Search: cosine similarity retrieval
#
# `search` embeds the query on a GPU, then asks pgvector to find the closest
# vectors using the cosine-distance operator (`<=>`).  Because we normalized
# our embeddings, cosine distance equals `1 - dot_product`, so
# `1 - (embedding <=> query)` gives an intuitive similarity score in [0, 1].
#
# For a 15-document corpus an exact scan is instant, but pgvector also supports
# approximate nearest-neighbour indexes (`ivfflat`, `hnsw`) that scale to
# millions of rows if you need them.


@app.function(image=image, secrets=[DB_SECRET])
def search(query: str, top_k: int = 5) -> list[tuple[float, str]]:
    import os

    import psycopg2

    # embed.remote returns a list; we only passed one string so we unpack it.
    (query_vec,) = embed.remote([query])
    vec_literal = f"[{','.join(map(str, query_vec))}]"

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 1 - (embedding <=> %s::vector) AS similarity,
               text
        FROM   documents
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
        """,
        (vec_literal, vec_literal, top_k),
    )

    results = [(float(row[0]), row[1]) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


# ## Run the demo
#
# The local entrypoint first calls `index` to populate the database, then fires
# a few example queries so you can see the semantic matching in action.
#
# Run with:
#
# ```bash
# modal run 06_gpu_and_ml/embeddings/vector_similarity_search.py
# ```


@app.local_entrypoint()
def main():
    print("=== Indexing corpus ===")
    index.remote()

    demo_queries = [
        "How do neural networks learn from data?",
        "What happens when a massive star collapses?",
        "How do you caramelize food properly?",
        "What makes Rust memory-safe without GC?",
    ]

    for query in demo_queries:
        print(f"\n=== Query: {query!r} ===")
        results = search.remote(query)
        for rank, (score, text) in enumerate(results, start=1):
            print(f"  {rank}. [{score:.4f}] {text}")
