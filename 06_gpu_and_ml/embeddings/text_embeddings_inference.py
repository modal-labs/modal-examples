import subprocess
from pathlib import Path

from modal import Image, Secret, Stub, Volume, gpu, method

GPU_CONFIG = gpu.A10G()
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
    import socket

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
    spawn_server()


image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)

volume = Volume.persisted("tei-hn-data")

stub = Stub("example-tei", image=image)


@stub.cls(
    secret=Secret.from_name("huggingface"),
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    concurrency_limit=10,
)
class TextEmbeddingsInference:
    def __enter__(self):
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    @method()
    async def embed(self, id: str, input: str):
        resp = self.client.post("/embed", json={"inputs": [input]})
        resp = await resp
        resp.raise_for_status()
        outputs = resp.json()
        return id, outputs[0]


def download_data():
    import json
    import os

    from google.cloud import bigquery
    from google.oauth2 import service_account

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    client = bigquery.Client(credentials=credentials)

    iterator = client.list_rows(
        "bigquery-public-data.hacker_news.full",
        max_results=100_000,
    )
    df = iterator.to_dataframe(progress_bar_type="tqdm")
    df["id"] = df["id"].astype(int)
    # TODO: better chunking / splitting.
    df["text"] = df["text"].apply(lambda x: x[:512])

    data = list(zip(df["id"], df["text"]))

    with open(DATA_PATH, "w") as f:
        json.dump(data, f)

    volume.commit()


@stub.function(
    image=Image.debian_slim().pip_install(
        "google-cloud-bigquery", "pandas", "db-dtypes", "tqdm"
    ),
    secrets=[Secret.from_name("bigquery")],
    volumes={DATA_PATH.parent: volume},
)
def embed_dataset():
    import json

    model = TextEmbeddingsInference()

    if not DATA_PATH.exists():
        print("Downloading data...")
        download_data()

    with open(DATA_PATH) as f:
        data = json.loads(f.read())

    # data is of type list[tuple[str, str]].
    # starmap spreads the tuples into positional arguments.
    output = list(model.embed.starmap(data))

    print(len(output))
