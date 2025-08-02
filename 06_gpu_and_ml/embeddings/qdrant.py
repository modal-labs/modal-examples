from typing import Optional

import modal

app = modal.App("example-qdrant")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "qdrant-client[fastembed-gpu]==1.13.3"
)


@app.function(image=image, gpu="any")
def query(inpt):
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")

    docs = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ]

    print("querying documents:", *docs, sep="\n\t")

    client.add(collection_name="demo_collection", documents=docs)

    print("query:", inpt, sep="\n\t")

    search_results = client.query(
        collection_name="demo_collection",
        query_text=inpt,
        limit=1,
    )

    print("result:", search_results[0], sep="\n\t")

    return search_results[0].document


@app.local_entrypoint()
def main(inpt: Optional[str] = None):
    if not inpt:
        inpt = "alpaca"

    print(query.remote(inpt))
