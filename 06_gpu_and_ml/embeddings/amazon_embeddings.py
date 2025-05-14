# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/embeddings/amazon_embeddings.py"]
# args: ["--dataset-subset", "raw_review_Magazine_Subscriptions"]
# ---

# # Embed 30 million Amazon reviews at 575k tokens per second with Qwen2-7B

# This example demonstrates how to create embeddings for a large text dataset. This is
# often necessary to enable semantic search, translation, and other language
# processing tasks. Modal makes it easy to deploy large, capable embedding models and handles
# all of the scaling to process very large datasets in parallel on many cloud GPUs.

# We create a Modal Function that will handle all of the data loading and submit inputs to an
# inference Cls that will automatically scale up to handle hundreds of large
# batches in parallel.

# Between the time a batch is submitted and the time it is fetched, it is stored via
# Modal's `spawn` system, which can hold onto up to one million inputs for up to a week.

import json
import subprocess
from pathlib import Path

import modal

app = modal.App(name="example-amazon-embeddings")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# We define our `main` function as a `local_entrypoint`. This is what we'll call locally
# to start the job on Modal.

# You can run it with the command

# ```bash
# modal run --detach amazon_embeddings.py
# ```

# By default we `down-scale` to 1/100th of the data for demonstration purposes.
# To launch the full job, set the `--down-scale` parameter to `1`.
# But note that this will cost you!

# The entrypoint starts the job and gets back a `f`unction `c`all ID for each batch.
# We can use these IDs to retrieve the embeddings once the job is finished.
# Modal will keep the results around for up to 7 days after completion. Take a look at our
# [job processing guide](https://modal.com/docs/guide/job-queue)
# for more details.


@app.local_entrypoint()
def main(
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023",
    dataset_subset: str = "raw_review_Books",
    down_scale: float = 0.001,
):
    out_path = Path("/tmp") / "embeddings-example-fc-ids.json"
    function_ids = launch_job.remote(
        dataset_name=dataset_name, dataset_subset=dataset_subset, down_scale=down_scale
    )
    out_path.write_text(json.dumps(function_ids, indent=2) + "\n")
    print(f"output handles saved to {out_path}")


# ## Load the data and start the inference job

# Next we define the Function that will do the data loading and feed it to our embedding model.
# We define a container [Image](https://modal.com/docs/guide/images)
# with the data loading dependencies.

# In it, we download the data we need and cache it to the container's local disk,
# which will disappear when the job is finished. We will be saving the review data
# along with the embeddings, so we don't need to keep the dataset around.

# Embedding a large dataset like this can take some time, but we don't need to wait
# around for it to finish. We use `spawn` to invoke our embedding Function
# and get back a handle with an ID that we can use to get the results later.
# This can bottleneck on just sending data over the network for processing, so
# we speed things up by using `ThreadPoolExecutor` to submit batches using multiple threads.

# Once all of the batches have been sent for inference, we can return the function IDs
# to the local client to save.


@app.function(
    image=modal.Image.debian_slim().pip_install("datasets==3.5.1"), timeout=2 * HOURS
)
def launch_job(dataset_name: str, dataset_subset: str, down_scale: float):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name,
        dataset_subset,
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


# ## Massively scaling up and scaling out embedding inference on many beefy GPUs

# We're going to spin up many containers to run inference, and we don't want each
# one to have to download the embedding model from Hugging Face. We can download and save it to a
# Modal [Volume](https://modal.com/docs/guide/volumes)
# during the image build step using `run_function`.

# We'll use the
# [GTE-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
# model from Alibaba, which performs well on the
# [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard).

MODEL_ID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MODEL_DIR = "/model"
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "embeddings-example-model-cache", create_if_missing=True
)


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, cache_dir=MODEL_DIR)


# For inference, we will use Hugging Face's
# [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
# framework for embedding model deployment.

# Running lots of separate machines is "scaling out". But we can also "scale up"
# by running on large, high-performance machines.

# We'll use L40S GPUs for a good balance between cost and performance. Hugging Face has
# prebuilt Docker images we can use as a base for our Modal Image.
# We'll use the one built for the L40S's
# [SM89/Ada Lovelace architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
# and install the rest of our dependencies on top.

tei_image = "ghcr.io/huggingface/text-embeddings-inference:89-1.7"

inference_image = (
    modal.Image.from_registry(tei_image, add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install(
        "httpx==0.28.1",
        "huggingface_hub[hf_transfer]==0.30.2",
        "numpy==2.2.5",
        "tqdm==4.67.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODEL_DIR})
    .run_function(download_model, volumes={MODEL_DIR: MODEL_CACHE_VOLUME})
)


# Next we define our inference class. Modal will auto-scale the number of
# containers ready to handle inputs based on the parameters we set in the `@app.cls`
# and `@modal.concurrent` decorators. Here we limit the total number of containers to
# 100 and the maximum number of concurrent inputs to 10, which caps us at 1000 concurrent batches.
# On Modal's Starter (free) and Team plans, the maximum number of concurrent GPUs is lower,
# reducing the total number of concurrent batches and so the throughput.

# Customers on Modal's Enterprise Plan regularly scale up another order of magnitude above this.
# If you're interested in running on thousands of GPUs,
# [get in touch](https://form.fillout.com/t/onUBuQZ5vCus).

# Here we also specify the GPU type and attach the Modal Volume where we saved the
# embedding model.

# This class will spawn a local Text Embeddings Inference server when the container
# starts, and process each batch by receiving the text data over HTTP, returning a list of
# tuples with the batch text data and embeddings.


@app.cls(
    image=inference_image,
    gpu="L40S",
    volumes={MODEL_DIR: MODEL_CACHE_VOLUME},
    max_containers=100,
    scaledown_window=5 * MINUTES,  # idle for 5 min without inputs before scaling down
    retries=3,  # handle transient failures and storms in the cloud
    timeout=2 * HOURS,  # run for at most 2 hours
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
        texts = [chunk[-1] for chunk in batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return [chunk + (embedding,) for chunk, embedding in zip(batch, res.json())]


# ## Helper Functions

# The book review dataset contains ~30M reviews with ~12B total characters,
# indicating an average review length of ~500 characters. Some are much longer.
# Embedding models have a limit on the number of tokens they can process in a single
# input. We will need to split each review into chunks that are under this limit.

# The proper way to split text data is to use a tokenizer to ensure that any
# single request is under the models token limit, and to overlap chunks to provide
# semantic context and preserve information. For the sake of this example, we're going
# just to split by a set character length (`CHUNK_SIZE`).

# While the embedding model has a limit on the number of input tokens for a single
# embedding, the number of chunks that we can process in a single batch is limited by
# the VRAM of the GPU. We set the `BATCH_SIZE` accordingly.


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
):
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
