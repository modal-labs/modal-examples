import asyncio
import json
import subprocess

import modal

# We first set out configuration variables for our script.
## Embedding Containers Configuration
GPU_CONCURRENCY = 100
GPU_CONFIG = "A10G"
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]
BATCH_SIZE = 512
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

## Dataset-Specific Configuration
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "embedding-model-cache", create_if_missing=True
)
DATASET_NAME = "wikipedia"
DATASET_READ_VOLUME = modal.Volume.from_name(
    "embedding-wikipedia", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = modal.Volume.from_name(
    "checkpoint", create_if_missing=True
)
MODEL_DIR = "/model"
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = True

## Upload-Specific Configuration
DATASET_HF_UPLOAD_REPO_NAME = "567-labs/upload-test"
UPLOAD_TO_HF = True

## HF Text-Embedding Inference specific Configuration

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
    "--huggingface-hub-cache",
    MODEL_DIR,
]


app = modal.App("example-embeddings")


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
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


tei_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "numpy")
)

with tei_image.imports():
    import numpy as np


def generate_chunks_from_dataset(xs, chunk_size: int):
    """
    Generate chunks from a dataset.

    Args:
        xs (list): The dataset containing dictionaries with "id", "url", "title", and "text" keys.
        chunk_size (int): The size of each chunk.

    Yields:
        tuple: A tuple containing the id, url, title, and a chunk of text.

    """
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )


def generate_batches(xs, batch_size):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    max_containers=GPU_CONCURRENCY,
    retries=3,
)
@modal.concurrent(max_inputs=10)
class TextEmbeddingsInference:
    @modal.enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @modal.exit()
    def terminate_connection(self):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @modal.method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""
        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
        ]

        embeddings = np.vstack(await asyncio.gather(*coros))
        return chunks, embeddings


def load_dataset_from_disk(down_scale: float = 0.01):
    """
    Load a dataset from disk and return a subset of the training data.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 0.01.

    Returns:
        Dataset: A subset of the training data.
    """
    import time

    from datasets import load_from_disk

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/wikipedia")
    dataset = load_from_disk(f"{DATASET_DIR}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter() - start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    sample_size = int(ttl_size * down_scale)

    return dataset["train"].select(range(sample_size))


def save_dataset_to_intermediate_checkpoint(acc_chunks, embeddings, batch_size):
    """Saves the dataset to an intermediate checkpoint.

    Args:
        acc_chunks (list): Accumulated chunks
        embeddings (list): Accumulated embeddings
        batch_size (int): Batch size
    """
    import pyarrow as pa
    from datasets import Dataset

    table = pa.Table.from_arrays(
        [
            pa.array([chunk[0] for chunk in acc_chunks]),  # id
            pa.array([chunk[1] for chunk in acc_chunks]),  # url
            pa.array([chunk[2] for chunk in acc_chunks]),  # title
            pa.array([chunk[3] for chunk in acc_chunks]),  # text
            pa.array(embeddings),
        ],
        names=["id", "url", "title", "text", "embedding"],
    )
    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    dataset = Dataset(table)
    dataset.save_to_disk(path_parent_folder)
    EMBEDDING_CHECKPOINT_VOLUME.commit()
    print(f"Saved checkpoint at {path_parent_folder}")


def upload_result_to_hf(batch_size: int) -> None:
    """
    Uploads the result to the Hugging Face Hub.

    Args:
        batch_size (int): The batch size for the model.

    Returns:
        None
    """
    import os
    import time

    from huggingface_hub import HfApi

    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Pushing to hub {DATASET_HF_UPLOAD_REPO_NAME}")
    start = time.perf_counter()
    api.upload_folder(
        folder_path=path_parent_folder,
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    end = time.perf_counter()
    print(f"Uploaded in {end - start}s")


@app.function(
    image=modal.Image.debian_slim().pip_install(
        "datasets", "pyarrow", "hf_transfer", "huggingface_hub"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME,
        MODEL_DIR: MODEL_CACHE_VOLUME,
    },
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def embed_dataset(down_scale: float = 1, batch_size: int = 512 * 50):
    """
    Embeds a dataset with the Text Embeddings Inference container.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 1.
        batch_size (int): The batch size to use. Defaults to 512 * 50.

    Returns:
        dict: A dictionary containing the benchmark results.
    """
    import datetime
    import time

    if UPLOAD_TO_HF and not SAVE_TO_DISK:
        raise ValueError(
            "Uploading to HF requires SAVE_TO_DISK to be set to true in case of intermediate failure."
        )

    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    subset = load_dataset_from_disk(down_scale)
    model = TextEmbeddingsInference()
    text_chunks = generate_chunks_from_dataset(subset, chunk_size=512)
    batches = generate_batches(text_chunks, batch_size=batch_size)

    start = time.perf_counter()
    acc_chunks = []
    embeddings = []
    for resp in model.embed.map(
        batches,
        order_outputs=False,
        return_exceptions=True,
        wrap_return_exceptions=False,
    ):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue

        batch_chunks, batch_embeddings = resp

        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    end = time.perf_counter()

    duration = end - start
    characters = sum(map(len, [chunk[3] for chunk in acc_chunks]))
    characters_per_sec = int(characters / duration)
    extrapolated_duration_cps_fmt = str(
        datetime.timedelta(seconds=dataset_chars / characters_per_sec)
    )
    resp = {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": GPU_CONCURRENCY,
        "duration_mins": duration / 60,
        "characters_per_sec": characters_per_sec,
        "extrapolated_duration": extrapolated_duration_cps_fmt,
    }

    if SAVE_TO_DISK:
        save_dataset_to_intermediate_checkpoint(acc_chunks, embeddings, batch_size)

    if UPLOAD_TO_HF:
        upload_result_to_hf(batch_size)

    return resp


@app.local_entrypoint()
def full_job():
    batch_size = 512 * 150
    with open("benchmarks.json", "a") as f:
        benchmark = embed_dataset.remote(batch_size=batch_size)
        f.write(json.dumps(benchmark, indent=2) + "\n")
