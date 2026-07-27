# # Serve Liquid AI ColBERT embeddings with llama.cpp and Modal Servers

# In this example, we serve
# [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M)
# using [llama.cpp](https://github.com/ggml-org/llama.cpp)
# in a [Modal Server](https://modal.com/docs/guide/servers).

# LFM2.5-ColBERT-350M is a 353M-parameter [late interaction embedding model](https://arxiv.org/abs/2004.12832).
# That means it produces one embedding vector per query token.
# Similarity is computed by comparing each of those vectors with each of the document's token embeddings to produce a final score,
# rather than just comparing a single vector per query per document.

# This is not supported by the OpenAI-compatible `/v1/embeddings` API,
# so we instead use the `/embeddings` API in `llama.cpp`.

# The model is intended for short queries against small documents,
# like comparing user search queries with product descriptions in e-commerce.

# The model is available under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M/blob/ac509ef9346912166a5f2f63d5ee41d9c472c330/LICENSE),
# which includes restrictions on commercial use.

# ## Why use a Modal Server?

# To minimize routing overheads, we use `@app.server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads,
# like interactive search via embeddings.
# See the [Modal Servers guide](https://modal.com/docs/guide/servers) for details.

# ## Choose the model file and engine parameters

# Liquid AI publishes official GGUF conversions of the model in
# [LiquidAI/LFM2.5-ColBERT-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M-GGUF).

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

MODEL_REPO = "LiquidAI/LFM2.5-ColBERT-350M-GGUF"
MODEL_REVISION = "bc240003aba07253e261a8aaf0d2c9683318a967"  # version-pinning
MODEL_FILE = "LFM2.5-ColBERT-350M-BF16.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"

# `llama-server` processes requests in `N_SLOTS` parallel slots
# and splits the total token context evenly across them.
# We give each slot the model's trained sequence length of 512 tokens.

MAX_INPUT_TOKENS = 512  # the model's trained sequence length
N_SLOTS = 4  # target concurrent requests per container; adjust as needed
TOKEN_EMBEDDING_DIM = 128  # the model's output embedding dimension

# Queries and documents are prefixed with special tokens before encoding.

DOCUMENT_PREFIX = "[D] "
QUERY_PREFIX = "[Q] "

# ## Cache the model weights

# We persist the llama.cpp download cache in a Modal
# [Volume](https://modal.com/docs/guide/volumes)
# so the GGUF file is downloaded from the Hub exactly once
# and loaded from the Volume on later cold starts.

CACHE_PATH = "/cache"
MODEL_PATH = f"{CACHE_PATH}/llama.cpp/{MODEL_FILE}"

volume = modal.Volume.from_name("liquidai-embeddings-cache", create_if_missing=True)

# ## Define the container image

# We build on the official llama.cpp server image.
# It contains the compiled binary but doesn't include Python,
# so `add_python` bundles an interpreter for Modal's own runtime.
# We also clear the image's entrypoint, which is the server binary itself,
# so that we can control the startup.

image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-b9917", add_python="3.12"
    )
    .entrypoint([])
    .env({"LLAMA_CACHE": f"{CACHE_PATH}/llama.cpp"})
)

# ## Define the Server

# We wrap the engine in a class decorated with `@app.server()`,
# which attaches the Image and Volume,
# defines autoscaling rules,
# fronts the containers with a proxy, and more.
# See details in the [Modal Servers guide](https://modal.com/docs/guide/servers).

MINUTES = 60  # seconds
PORT = 8000


app = modal.App("example-liquidai-embeddings-server")


@app.server(
    image=image,
    volumes={CACHE_PATH: volume},
    port=PORT,
    target_concurrency=N_SLOTS,
    min_containers=0,  # set to 1 or more to keep a warm replica for latency-sensitive use cases
    startup_timeout=10 * MINUTES,  # allow time to download and load the model
    scaledown_window=5 * MINUTES,  # retain loaded replicas across short traffic gaps
    exit_grace_period=20,  # allow in-flight embedding requests to finish before shutdown
    unauthenticated=True,
)
class LlamaServer:
    @modal.enter()
    def start(self):
        cmd = [
            "/app/llama-server",
            "--model-url",
            MODEL_URL,
            "--model",
            MODEL_PATH,
            "--embeddings",
            "--pooling",
            "none",  # return one vector per input token
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--no-ui",  # no chat interface, we're serving embeddings
            "--metrics",  # enable metrics logging
            "--parallel",
            str(N_SLOTS),
            "--ctx-size",
            str(N_SLOTS * MAX_INPUT_TOKENS),  # total context shared across slots
            # Bidirectional embedding models require all tokens in an iteration to fit
            # in one physical batch. Using the full context for batch sizes allows all
            # slots to process maximum-length inputs together.
            "--batch-size",  # max tokens per iteration during continuous batching
            str(N_SLOTS * MAX_INPUT_TOKENS),
            "--ubatch-size",  # max tokens in a physical computation batch
            str(N_SLOTS * MAX_INPUT_TOKENS),
        ]

        self.proc = subprocess.Popen(cmd, start_new_session=True)
        wait_ready(self.proc)

    @modal.exit()
    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


# Modal considers a new replica ready once the `@modal.enter` methods have exited
# and the serving process inside starts accepting connections.
# `llama-server` answers `/health` with a [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# while the model loads, so we block the startup hook from completing with this `wait_ready` function.


def wait_ready(proc: subprocess.Popen, port=PORT):
    import socket

    while True:
        try:
            if (
                returncode := proc.poll()
            ) is not None:  # fail fast if the server process died
                raise RuntimeError(f"Server process exited with code {returncode}")
            socket.create_connection(("127.0.0.1", port), timeout=5).close()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            request_with_retry(request, timeout=30).close()
            return
        except (ConnectionRefusedError, TimeoutError):
            continue


# Modal Servers also respond with a 503 when no replicas are available to handle requests, so we pull out
# a helper function that retries on 503 for use with server clients.


def request_with_retry(request: urllib.request.Request, timeout=10 * MINUTES):
    deadline = time.monotonic() + timeout
    delay = 1.0

    while (remaining := deadline - time.monotonic()) > 0:
        try:
            return urllib.request.urlopen(request, timeout=remaining)
        except urllib.error.HTTPError as exc:
            if exc.code != 503:  # 503 Service Unavailable, no containers ready
                raise exc
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 10.0)
    raise TimeoutError(f"Server not ready within {timeout} seconds")


# ## Test the server

# Running `modal run liquidai_embeddings_server.py` executes the `local_entrypoint` below
# against a temporary instance of the Server, which is useful for testing and development.
# The client requests one embedding after waiting for a live replica.


@app.local_entrypoint()
def main(input: str | None = None, test_timeout: int = 5 * MINUTES):
    url = LlamaServer.get_url()
    print(f"Server URL: {url}")

    request = urllib.request.Request(f"{url}/health")
    request_with_retry(request=request, timeout=test_timeout).close()

    if input is None:
        input = "ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT"

    request_data = {"input": [DOCUMENT_PREFIX + input]}
    request = urllib.request.Request(
        f"{url}/embeddings",
        data=json.dumps(request_data).encode(),
        headers={"Content-Type": "application/json"},
    )

    started_at = time.perf_counter()
    with request_with_retry(request, timeout=test_timeout) as response:
        elapsed = time.perf_counter() - started_at
        data = json.load(response)

    assert len(data), "empty response from server"
    embedding = data[0].get("embedding")
    assert embedding, f"server failed to respond with embedding, got {data}"

    print(
        f"client-side inference latency: {elapsed:.3f}s",
        f"embedding shape: ({len(embedding)}, {len(embedding[0])})",
        sep="\n",
    )


# ## Deploy the Server

# Deploy the Server with

# ```bash
# modal deploy liquidai_embeddings_server.py
# ```

# The deploy command prints the Server's public URL.
