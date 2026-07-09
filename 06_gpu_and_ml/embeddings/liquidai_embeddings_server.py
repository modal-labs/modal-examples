# ---
# deploy: true
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/liquidai_embeddings_server.py"]
# ---

# # Serve Liquid AI embeddings with llama.cpp and Modal Servers

# In this example, we serve
# [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M)
# behind an OpenAI-compatible `POST /v1/embeddings` API
# using [llama.cpp](https://github.com/ggml-org/llama.cpp)
# and [Modal Servers](https://modal.com/docs/guide/servers).

# LFM2.5-Embedding-350M is a 350M-parameter multilingual embedding model.
# It is a dense bidirectional encoder that produces one 1024-dimensional vector per input
# and supports retrieval across eleven languages.
# It is also small enough to serve economically on CPU, so this example doesn't require a GPU.

# For client code, use the appropriate prompt prefixes for inputs.
# Prepend `"query: "` when embedding search queries and `"document: "` when embedding passages.
# Omitting the prefixes silently degrades retrieval quality, since the model was trained as
# an asymmetric retriever.

# ## Why use a Modal Server?

# [Modal Servers](https://modal.com/docs/guide/servers) route requests directly
# to a long-lived process listening on a port inside the container,
# which is well-suited for llama.cpp.
# The `llama-server` binary speaks HTTP natively and implements the
# [OpenAI embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
# out of the box, so the Python code in this example only starts and stops
# that process, with no web framework needed.
# Support for the LFM2.5 embedding model landed in
# [ggml-org/llama.cpp#24913](https://github.com/ggml-org/llama.cpp/pull/24913).

import subprocess
import time
import urllib.error
import urllib.request

import modal

# ## Choose the model file and engine parameters

# Liquid AI publishes official GGUF conversions of the model in
# [LiquidAI/LFM2.5-Embedding-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M-GGUF).
# We serve the F16 file, a near-lossless conversion of the BF16 training precision.
# At roughly 700 MB, the file is already small enough to not require quantization.
# `llama-server` downloads the file from the Hugging Face Hub on first start.

MODEL_REPO = "LiquidAI/LFM2.5-Embedding-350M-GGUF"
MODEL_REVISION = "a80de9c5b941d429104f0038292a0ef5a860e486"  # version-pinning
MODEL_FILE = "LFM2.5-Embedding-350M-F16.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"

# `llama-server` processes requests in `N_SLOTS` parallel slots
# and splits the total token context evenly across them.
# We give each slot exactly the model's trained sequence length of 512 tokens.
# In embedding mode, llama.cpp requires each input to fit in a single physical batch,
# and caps the logical batch size at the physical batch size.
# Sizing both batches to the full context lets all `N_SLOTS` slots
# process maximal inputs in one forward pass.

MAX_INPUT_TOKENS = 512  # the model's trained sequence length
N_SLOTS = 4  # concurrent requests per container
N_CTX = N_SLOTS * MAX_INPUT_TOKENS  # total tokens, also the batch/ubatch size

# ## Cache the model weights

# We persist the llama.cpp download cache in a Modal
# [Volume](https://modal.com/docs/guide/volumes)
# so the GGUF file is downloaded from the Hub exactly once
# and loaded from the Volume on later cold starts.

CACHE_PATH = "/cache"
MODEL_PATH = f"{CACHE_PATH}/llama.cpp/{MODEL_FILE}"  # where the download lands

volume = modal.Volume.from_name("liquidai-embeddings-cache", create_if_missing=True)

# ## Define the container image

# We build on the official llama.cpp server image.
# It contains the compiled binary and doesn't include Python,
# so `add_python` bundles an interpreter for Modal's own runtime.
# We also clear the image's entrypoint, which is the server binary itself,
# because we launch that binary ourselves in the server's startup hook.

image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-b9917", add_python="3.12"
    )
    .entrypoint([])
    .env({"LLAMA_CACHE": f"{CACHE_PATH}/llama.cpp"})
)

# ## Define the Server

# We wrap the engine in a class registered with `@app.server()`,
# which attaches the image, Volume, and resources
# and fronts the containers with a proxy.
# The `@modal.enter` and `@modal.exit` lifecycle hooks below
# start and stop the `llama-server` process.
# See the [reference documentation](https://modal.com/docs/reference/modal.App#server) for details.

# Modal considers a new replica ready once the `@modal.enter` methods have exited
# and the container accepts connections.
# `llama-server` answers `/health` with a 503 status while the model loads,
# so the startup hook blocks in `wait_ready` until it answers with a 200.

MINUTES = 60  # seconds
PORT = 8000


def wait_ready(proc: subprocess.Popen, timeout: int = 10 * MINUTES):
    """Block until llama-server answers /health with a 200 status."""
    deadline = time.monotonic() + timeout
    delay = 1.0
    while (remaining := deadline - time.monotonic()) > 0:
        if (returncode := proc.poll()) is not None:  # fail fast if the engine died
            raise RuntimeError(f"llama-server exited with code {returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5):
                return
        except (urllib.error.HTTPError, OSError):
            # 503 while the model loads, or connection refused before the port binds.
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 10.0)
    raise TimeoutError(
        f"Liquid AI embeddings server not ready within {timeout} seconds"
    )


app = modal.App("example-liquidai-embeddings")

# The resource reservations follow from the engine parameters:
# 1 CPU core per slot, and 2 GB of memory to hold
# the roughly 700 MB of F16 weights plus KV cache and other overhead.
# Setting `target_concurrency` to the slot count sends each container
# only as many concurrent requests as it has slots,
# and load beyond that scales up new containers instead
# (see the [autoscaling guide](https://modal.com/docs/guide/scale)).


@app.server(
    image=image,
    volumes={CACHE_PATH: volume},
    port=PORT,
    cpu=N_SLOTS,
    memory=2048,  # MBs
    target_concurrency=N_SLOTS,
    min_containers=0,  # set to 1 or more to ensure a replica is always ready
    startup_timeout=10 * MINUTES,  # allows time to download the GGUF on startup
    scaledown_window=5 * MINUTES,
    exit_grace_period=20,
    unauthenticated=True,
)
class LlamaCppEmbeddingServer:
    @modal.enter()
    def start(self):
        # The image sets LLAMA_ARG_HOST=0.0.0.0, which binds all interfaces,
        # so we do not pass --host.
        # start_new_session blocks llama-server from receiving the container-wide
        # shutdown signal. Otherwise, it would receive that signal in addition
        # to our terminate() below, and llama.cpp would treat a second signal
        # as "abort immediately", which would skip graceful cleanup.
        cmd = [
            "/app/llama-server",
            "--model-url",
            MODEL_URL,
            "--model",
            MODEL_PATH,
            "--embeddings",
            "--port",
            str(PORT),
            "--parallel",
            str(N_SLOTS),
            "--ctx-size",
            str(N_CTX),
            "--batch-size",
            str(N_CTX),
            "--ubatch-size",
            str(N_CTX),
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


# ## Deploy the server

# Deploy the Server with

# ```bash
# modal deploy liquidai_embeddings_server.py
# ```

# The deploy command prints the server's public URL.
# You can also retrieve the URL programmatically with
# [`modal.Server.get_url`](https://modal.com/docs/reference/modal.Server).

# ## Test the server

# Running `modal run liquidai_embeddings_server.py` executes the `local_entrypoint` below
# against a temporary instance of the Server.
# The client polls `/health` until a container is ready,
# then requests one document embedding and verifies its shape.


@app.local_entrypoint()
def main(timeout_s: float = 600):
    import json

    url = LlamaCppEmbeddingServer.get_url()
    print(f"Server URL: {url}")

    # Poll /health, retrying on 503 (cold start) and connection errors.
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10):
                break
        except urllib.error.HTTPError as exc:
            reason = f"{exc.code} (container cold-starting)"
        except OSError as exc:
            reason = f"connection error ({exc.__class__.__name__})"
        print(f"  {reason}, retrying... ({remaining:.0f}s left)")
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 10.0)

    if remaining <= 0:
        raise TimeoutError(f"server not ready within {timeout_s}s")

    # Note the "document: " prompt prefix. See the intro for why it matters.
    request = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=json.dumps(
            {"input": ["document: The quick brown fox jumps over the lazy dog."]}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)

    vector = result["data"][0]["embedding"]

    assert len(vector) == 1024, f"expected 1024, got {len(vector)}"
    print(
        f"embedding dim: {len(vector)}, first 4 values: {[round(v, 4) for v in vector[:4]]}"
    )
