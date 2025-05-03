# ---
# args: ["--down-scale", ".0001"]
# ---

# # Embedding 30 Million Amazon Reviews with GTE-Qwen2-7B-instruct

# This example demonstrates how to create embeddings for a large text dataset. This is
# often necessary to enable semantic search, translation, and other powerful language
# processing tasks. Modal makes it easy to deploy a large embedding model and handles
# all of the scaling to process very large datasets.

# We'll create a Modal Function that will handle all of the data loading, and an
# inference class that Modal will automatically scale up to handle hundreds of large
# batches in parallel.

import json
import subprocess

import modal

app = modal.App(name="embeddings-example-inference")
MINUTES = 60

# We define our `main` function -- this is what we'll call locally to start the job on
# Modal. This function will batch the data, start the inference job, and return the
# function IDs corresponding to each batch. We can use these IDs to retrieve the
# embeddings once the job is finished. Modal will keep the results around for up to
# 7 days after completion. Take a look at
# [Job Processing](https://modal.com/docs/guide/job-queue)
# for more details.


@app.local_entrypoint()
def main(down_scale: float = 1):
    with open("embeddings-example-function_ids.json", "w") as f:
        function_ids = launch_job.remote(down_scale=down_scale)
        f.write(json.dumps(function_ids, indent=2) + "\n")


# ## Creating a Modal Function for loading the data and starting the inference job

# Next we define our function that will do the data loading and feed our inference
# `modal.method`. We define an image and install `datasets`, and give it a timeout of
# 4 hours.

# Then we use `load_dataset` do download the subset we want and cache it to the
# container's local storage, which will disappear when the job is finished. We will be
# saving the review data along with the embeddings, so we don't need to keep the dataset
# around.

# We then instantiate our `TextEmbeddingInference` `modal.App.cls` which is our
# inference class that we will define later. We also create a generator with batches of
# chunked text data to send for inference.

# Embedding a large dataset like this can take some time, and we don't want to wait
# around for it to finish. We can use `modal.Function.spawn` to call our embedding
# function and get back a handle with an ID that we can use to get the results later. We
# can speed things up by using `ThreadPoolExecutor` to submit batches using multiple
# threads.

# Once all of the batches have been sent for inference, we can return the function IDs
# to the local client to save.


@app.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    timeout=240 * MINUTES,
)
def launch_job(down_scale: float = 1):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading dataset...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Books",
        split="full",
        trust_remote_code=True,
    )

    data_subset = dataset.select(range(int(len(dataset) * down_scale)))

    tei = TextEmbeddingsInference()
    batches = generate_batches_of_chunks(data_subset)

    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(tei.embed.spawn, batch) for batch in tqdm(batches)]
        function_ids = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            function_ids.append(future.result().object_id)
    print(f"Finished submitting job: {time.perf_counter() - start:.2f}s")

    return function_ids


# ## Creating a Modal Cls for scalable inference with GPUs

# We're going to spin up many GPU containers to run inference, and we don't want each
# one to have to download the embedding model. We can download and save it to a
# Modal [Volume](https://modal.com/docs/guide/volumes)
# during the image build step using `run_function`.

# We'll use the
# [GTE-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
# model, which ranks high (70.72) on the
# [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

MODEL_ID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MODEL_DIR = "/model"
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "embeddings-example-model-cache", create_if_missing=True
)


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, cache_dir=MODEL_DIR)


# For inference, we will use
# [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
#  - a fast toolkit for deploying embedding models.

# We'll use L40S GPUs, for a good balance between cost and performance. Huggingface has
# a prebuilt Docker image we can use as a base for our Modal image. We'll use the one
# built for the Lovelace architecture, and install the rest of our dependencies.

TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.7"

inference_image = (
    modal.Image.from_registry(TEI_IMAGE, add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("huggingface_hub[hf_transfer]", "httpx", "numpy", "tqdm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODEL_DIR})
    .run_function(download_model, volumes={MODEL_DIR: MODEL_CACHE_VOLUME})
)


# Next we want to define our inference class. Modal will auto-scale the number of
# containers ready to handle inputs based on the parameters we set in the `@app.cls`
# and `@modal.concurrent` decorators. Here we limit the total number of containers to
# 100, and the maximum number of concurrent inputs to 10.

# Here we also specify the GPU type, and attach the Modal Volume where we saved the
# embedding model.

# This class will spawn a local Text Embeddings Inference server when the container
# starts, and process each batch by sending text data over HTTP, and returning a list of
# tuples with the batch text data and embeddings.


@app.cls(
    image=inference_image,
    gpu="L40S",
    volumes={
        MODEL_DIR: MODEL_CACHE_VOLUME,
    },
    max_containers=100,
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


# ## Helper Functions

# The book review dataset contains ~30M reviews with ~12B total characters.
# Embedding models have a limit on the number of tokens they can process in a single
# input. We will need to split each review into chunks that are under this limit.

# While the embedding model has a limit on the number of input tokens for a single
# embedding, the number of chunks that we can process in a single batch is limited by
# the VRAM of the GPU.

# The proper way to split text data is to use a tokenizer to ensure that any
# single request is under the models token limit, and to overlap chunks to provide
# semantic context and preserve information. For the sake of this example, we're going
# just to split by a set character length (CHUNK_SIZE).


BATCH_SIZE = 256
CHUNK_SIZE = 512


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
