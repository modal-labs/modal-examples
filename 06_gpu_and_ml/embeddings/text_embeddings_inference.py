# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/text_embeddings_inference.py::embed_dataset"]
# ---

# # Run TextEmbeddingsInference (TEI) on Modal

# This example runs the [Text Embedding Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) toolkit on the Hacker News BigQuery public dataset.

import json
import os
import socket
import subprocess
from pathlib import Path

import modal

GPU_CONFIG = modal.gpu.A10G()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

DATA_PATH = Path("/data/dataset.jsonl")

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
]


def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server().terminate()


volume = modal.Volume.from_name("tei-hn-data", create_if_missing=True)

app = modal.App("example-tei")


tei_image = (
    modal.Image.from_registry(
        DOCKER_IMAGE,
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)


with tei_image.imports():
    from httpx import AsyncClient


@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 20 GPU containers at once.
    concurrency_limit=20,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    @modal.enter()
    def setup_server(self):
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    @modal.exit()
    def teardown_server(self):
        self.process.terminate()

    @modal.method()
    async def embed(self, inputs_with_ids: list[tuple[int, str]]):
        ids, inputs = zip(*inputs_with_ids)
        resp = await self.client.post("/embed", json={"inputs": inputs})
        resp.raise_for_status()
        outputs = resp.json()

        return list(zip(ids, outputs))


def download_data():
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    client = bigquery.Client(credentials=credentials)

    iterator = client.list_rows(
        "bigquery-public-data.hacker_news.full",
        max_results=100_000,
    )
    df = iterator.to_dataframe(progress_bar_type="tqdm").dropna()

    df["id"] = df["id"].astype(int)
    df["text"] = df["text"].apply(lambda x: x[:512])

    data = list(zip(df["id"], df["text"]))

    with open(DATA_PATH, "w") as f:
        json.dump(data, f)

    volume.commit()


image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "google-cloud-bigquery", "pandas", "db-dtypes", "tqdm"
)

with image.imports():
    from google.cloud import bigquery
    from google.oauth2 import service_account


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("bigquery")],
    volumes={DATA_PATH.parent: volume},
)
def embed_dataset():
    model = TextEmbeddingsInference()

    if not DATA_PATH.exists():
        print("Downloading data. This takes a while...")
        download_data()

    with open(DATA_PATH) as f:
        data = json.loads(f.read())

    def generate_batches():
        batch = []
        for item in data:
            batch.append(item)

            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

    # data is of type list[tuple[str, str]].
    # starmap spreads the tuples into positional arguments.
    for output_batch in model.embed.map(
        generate_batches(), order_outputs=False
    ):
        # Do something with the outputs.
        pass
