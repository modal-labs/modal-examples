# ---
# lambda-test: false  # requires the Server from liquidai_embeddings_server.py to be running
# ---

# # Call the Liquid AI embeddings Server

# This client invokes the OpenAI-compatible embeddings Server implemented in
# [liquidai_embeddings_server.py](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/embeddings/liquidai_embeddings_server.py).
# Deploy that Server first with `modal deploy liquidai_embeddings_server.py`, then run this client locally with

# ```bash
# python liquidai_embeddings_client.py
# ```

# As explained in the server file, Modal Servers reject requests with a
# `503 Service Unavailable` status while no container is ready,
# so the client polls `/health` until a container answers before sending requests.

import json
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "example-liquidai-embeddings"
SERVER_NAME = "LlamaCppEmbeddingServer"

MODEL = "LiquidAI/LFM2.5-Embedding-350M"

# ## Handle cold starts

# `wait_until_ready` retries through the 503s and connection errors
# that a cold start produces, backing off up to ten seconds between checks.
# When a container is already warm, the first check succeeds immediately.


def wait_until_ready(base_url: str, timeout_s: float = 600) -> None:
    """Poll /health until the server has a ready container.

    Raises TimeoutError after timeout_s seconds.
    """
    stop_at = time.monotonic() + timeout_s
    delay = 1.0
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10):
                return
        except urllib.error.HTTPError as exc:
            reason = f"{exc.code} (container cold-starting)"
        except OSError as exc:
            reason = f"connection error ({exc.__class__.__name__})"

        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"server not ready within {timeout_s}s; last: {reason}")
        print(f"  {reason}, retrying... ({remaining:.0f}s left)")
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 10.0)


# ## Request embeddings

# The endpoint accepts OpenAI-style payloads.
# `input` takes a string or a list of strings,
# and one embedding comes back per input.
# The vectors are unit-normalized,
# so the dot product of two embeddings is their cosine similarity.
# Remember to prepend `"query: "` to search queries and `"document: "` to passages,
# as described in the server file.


def post_embeddings(base_url: str, payload: dict) -> dict:
    """POST /v1/embeddings. Call wait_until_ready(base_url) first."""
    request = urllib.request.Request(
        f"{base_url}/v1/embeddings",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Surface the server's explanation (e.g. input too long) with the error.
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{exc.code} from {base_url}: {detail}") from None


def embed(vectors_json: dict) -> list[list[float]]:
    """Pull the vectors out of an OpenAI-shaped embeddings response."""
    return [d["embedding"] for d in vectors_json["data"]]


# ## Put it together

# We look up the deployed Server's URL by name,
# wait for a container to be ready, and embed one document.

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."

    try:
        server = modal.Server.from_name(APP_NAME, SERVER_NAME)
        url = server.get_url()
    except modal.exception.NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. "
            "Run `modal deploy liquidai_embeddings_server.py` first."
        )

    print(f"llama.cpp server: {url}")
    wait_until_ready(url)

    result = post_embeddings(url, {"model": MODEL, "input": [f"document: {text}"]})
    vector = embed(result)[0]
    print(f"  dim: {len(vector)}, first 4: {[round(v, 4) for v in vector[:4]]}")
