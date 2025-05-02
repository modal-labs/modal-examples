# # Embedding 30 Million Amazon Reviews with GTE-Qwen2-7B-instruct
# This example demonstrates how to create embeddings for a large text dataset.

import json
import subprocess

import modal

app = modal.App(name="embeddings-example-inference")

GPU_CONCURRENCY = 300
GPU_CONFIG = "L40S"

BATCH_SIZE = 256
CHUNK_SIZE = 512

MODEL_ID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MODEL_DIR = "/model"
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "embeddings-example-model-cache", create_if_missing=True
)
MINUTES = 60


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, cache_dir=MODEL_DIR)


TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.7"
inference_image = (
    modal.Image.from_registry(TEI_IMAGE, add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("huggingface_hub[hf_transfer]", "httpx", "numpy", "tqdm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODEL_DIR})
    .run_function(download_model, volumes={MODEL_DIR: MODEL_CACHE_VOLUME})
)


@app.cls(
    image=inference_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_DIR: MODEL_CACHE_VOLUME,
    },
    max_containers=GPU_CONCURRENCY,
    scaledown_window=5 * MINUTES,
    retries=3,
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=10)
class TextEmbeddingsInference:
    @modal.enter()
    def open_connection(self):
        from httpx import AsyncClient

        print("Starting text embedding inference server...")
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @modal.exit()
    def terminate_connection(self):
        self.process.terminate()

    @modal.method()
    async def embed(self, batch):
        texts = [chunk[6] for chunk in batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return [chunk + (embedding,) for chunk, embedding in zip(batch, res.json())]


@app.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    timeout=240 * MINUTES,
)
def launch_job(down_scale: float = 1):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from datasets import load_dataset
    from tqdm import tqdm

    DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
    DATASET_CONFIG = "raw_review_Books"
    print("Loading dataset...")
    start = time.perf_counter()
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="full",
        trust_remote_code=True,
    )
    end = time.perf_counter()
    print(f"Download complete - downloaded files in {end - start:.2f}s")

    data_subset = dataset.select(range(int(len(dataset) * down_scale)))

    tei = TextEmbeddingsInference()
    batches = generate_batches_of_chunks(data_subset)

    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(tei.embed.spawn, batch) for batch in tqdm(batches)]
        function_ids = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            function_ids.append(future.result().object_id)
    end = time.perf_counter()
    print(f"Submitting embedding job complete - {end - start:.2f}s")

    return function_ids


@app.local_entrypoint()
def main(down_scale: float = 1):
    with open("embeddings-example-function_ids.json", "w") as f:
        function_ids = launch_job.remote(down_scale=down_scale)
        f.write(json.dumps(function_ids, indent=2) + "\n")


# Helper functions
def spawn_server(
    model_id: str = MODEL_ID,
    port: int = 8000,
    max_client_batch_size: int = BATCH_SIZE,
    max_batch_tokens: int = BATCH_SIZE * CHUNK_SIZE,
    huggingface_hub_cache: str = MODEL_DIR,
) -> subprocess.Popen:
    """Starts a text embedding inference server in a subprocess."""
    import socket

    LAUNCH_FLAGS = [
        "--model-id",
        model_id,
        "--port",
        str(port),
        "--max-client-batch-size",
        str(max_client_batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--huggingface-hub-cache",
        huggingface_hub_cache,
    ]

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)
    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            print("Inference server ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            retcode = process.poll()  # Check if the process has terminated.
            if retcode is not None:
                raise RuntimeError(f"Launcher exited unexpectedly with code {retcode}")


def generate_batches_of_chunks(
    dataset, chunk_size: int = CHUNK_SIZE, batch_size: int = BATCH_SIZE
):
    """Creates batches of chunks by naively slicing strings according to CHUNK_SIZE."""
    batch = []
    for entry_index, data in enumerate(dataset):
        product_id = data["asin"]
        user_id = data["user_id"]
        timestamp = data["timestamp"]
        title = data["title"]
        text = data["text"]
        for chunk_index, chunk_start in enumerate(range(0, len(text), chunk_size)):
            batch.append(
                (
                    entry_index,
                    chunk_index,
                    product_id,
                    user_id,
                    timestamp,
                    title,
                    text[chunk_start : chunk_start + chunk_size],
                )
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
