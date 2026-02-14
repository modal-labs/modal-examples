# ---
# deploy: true
# cmd: ["modal", "run", "06_gpu_and_ml/llm-serving/sglang_embeddings.py"]
# ---

# # High-throughput Qwen3-Embedding-8B with SGLang on Modal

# This example demonstrates serving Alibaba's Qwen3-Embedding-8B model
# with SGLang for high-throughput embedding generation.

# Unlike chat completion models, embedding models are prefill-only:
# they process input text and produce a fixed-dimensional vector representation.
# This has important implications for serving:

# - **No speculative decoding benefit**. Speculative decoding accelerates the decode phase
#   of token generation, but embeddings only do a single forward pass (prefill).

# - **No KV cache reuse**. Each embedding request is independent; there's no conversation
#   history to cache between requests.

# - **Throughput via data parallelism, not tensor parallelism**. The 8B model fits on a
#   single GPU, so scale out with more replicas rather than splitting across GPUs.

# The model supports:
# - 32K context length
# - 4096-dimensional embeddings
# - Matryoshka representation learning (MRL) for reduced dimensions
# - Instruction-aware embedding (queries can include task instructions for better retrieval)

# ## Set up the container image

import asyncio
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
)

# ## Model configuration

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "main"

sglang_image = sglang_image.env(
    {
        "HF_HUB_CACHE": "/root/.cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
)

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# ## GPU and parallelism configuration

GPU = "L40S"
REGION = "us-east"
TARGET_INPUTS = 100
MAX_INPUTS = 500
MIN_CONTAINERS = 0

# ## Server helpers

with sglang_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(2)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "model": MODEL_NAME,
        "input": "Hello, world!",
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/embeddings",
            json=payload,
            timeout=30,
        ).raise_for_status()


def sleep():
    requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_up():
    requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


# ## Define the embedding server

app = modal.App(name="example-sglang-embeddings")
PORT = 8000


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    region=REGION,
    min_containers=MIN_CONTAINERS,
    scaledown_window=10 * MINUTES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=[REGION],
    exit_grace_period=15,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class SGLangEmbeddings:
    @modal.enter(snap=True)
    def startup(self):
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--is-embedding",
            "--max-running-requests",
            str(MAX_INPUTS),
            "--cuda-graph-max-bs",
            str(MAX_INPUTS // 2),
            "--mem-fraction-static",
            "0.9",
            "--enable-metrics",
            "--attention-backend",
            "fa3",
            "--enable-memory-saver",
            "--enable-weights-cpu-backup",
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        sleep()

    @modal.enter(snap=False)
    def restore(self):
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Test the server


@app.local_entrypoint()
async def main():
    url = (await SGLangEmbeddings._experimental_get_flash_urls.aio())[0]
    print(f"Server URL: {url}")

    await probe_health(url)
    await test_single_embedding(url)
    await test_batch_embeddings(url)
    await test_similarity(url)
    await test_throughput(url)


async def probe_health(url: str, timeout: int = 5 * MINUTES):
    import aiohttp.client_exceptions

    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    "/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    resp.raise_for_status()
                    print("Server is healthy!")
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if isinstance(e, aiohttp.client_exceptions.ClientResponseError):
                    if e.status == 503:
                        await asyncio.sleep(2)
                        continue
                await asyncio.sleep(2)
    raise TimeoutError(f"Server not healthy within {timeout} seconds")


async def test_single_embedding(url: str):
    print("\n--- Testing single embedding ---")

    payload = {
        "model": MODEL_NAME,
        "input": "What is the capital of France?",
        "encoding_format": "float",
    }

    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.post("/v1/embeddings", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    embedding = data["data"][0]["embedding"]
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


async def test_batch_embeddings(url: str):
    print("\n--- Testing batch embeddings ---")

    payload = {
        "model": MODEL_NAME,
        "input": [
            "What is the capital of France?",
            "Explain quantum computing.",
            "How do neural networks work?",
        ],
        "encoding_format": "float",
    }

    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.post("/v1/embeddings", json=payload) as session_resp:
            session_resp.raise_for_status()
            data = await session_resp.json()

    for i, item in enumerate(data["data"]):
        print(f"Input {i}: embedding dim = {len(item['embedding'])}")


async def test_similarity(url: str):
    print("\n--- Testing cosine similarity ---")

    queries = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]

    payload = {
        "model": MODEL_NAME,
        "input": queries + documents,
        "encoding_format": "float",
    }

    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.post("/v1/embeddings", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    embeddings = [item["embedding"] for item in data["data"]]

    query_emb_0 = normalize(embeddings[0])
    query_emb_1 = normalize(embeddings[1])
    doc_emb_0 = normalize(embeddings[2])
    doc_emb_1 = normalize(embeddings[3])

    scores = [
        [
            cosine_similarity(query_emb_0, doc_emb_0),
            cosine_similarity(query_emb_0, doc_emb_1),
        ],
        [
            cosine_similarity(query_emb_1, doc_emb_0),
            cosine_similarity(query_emb_1, doc_emb_1),
        ],
    ]

    print("Similarity matrix (queries x documents):")
    print(f"  Query 0 (capital of China) -> Doc 0 (Beijing): {scores[0][0]:.4f}")
    print(f"  Query 0 (capital of China) -> Doc 1 (gravity):  {scores[0][1]:.4f}")
    print(f"  Query 1 (gravity)        -> Doc 0 (Beijing): {scores[1][0]:.4f}")
    print(f"  Query 1 (gravity)        -> Doc 1 (gravity):  {scores[1][1]:.4f}")
    print("\nExpected: diagonal scores should be higher than off-diagonal")


def normalize(vec):
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


async def test_throughput(url: str, num_requests: int = 50, concurrency: int = 10):
    print(
        f"\n--- Testing throughput: {num_requests} requests with {concurrency} concurrent ---"
    )

    import asyncio
    import random
    import string

    texts = [
        "".join(
            random.choices(
                string.ascii_letters + string.digits + " ", k=random.randint(50, 200)
            )
        )
        for _ in range(num_requests)
    ]

    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async def make_request(text: str) -> float:
        async with semaphore:
            req_start = time.perf_counter()
            payload = {"model": MODEL_NAME, "input": text, "encoding_format": "float"}
            async with aiohttp.ClientSession(base_url=url) as session:
                async with session.post("/v1/embeddings", json=payload) as resp:
                    resp.raise_for_status()
                    await resp.json()
            return time.perf_counter() - req_start

    latencies = await asyncio.gather(*[make_request(t) for t in texts])
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {num_requests / elapsed:.2f} req/s")
    print(f"Avg latency: {sum(latencies) / len(latencies) * 1000:.1f}ms")
    print(f"P50 latency: {sorted(latencies)[len(latencies) // 2] * 1000:.1f}ms")
    print(f"P99 latency: {sorted(latencies)[int(len(latencies) * 0.99)] * 1000:.1f}ms")


# ## Deploy the server

# ```bash
# modal deploy 06_gpu_and_ml/llm-serving/sglang_embeddings.py
# ```

# ## Test GPU snapshotting

# To test snapshotting, first deploy the app, then run this script directly.
# The snapshot is created after the first container starts up and goes through
# several cold starts before it becomes available.

# ```bash
# python 06_gpu_and_ml/llm-serving/sglang_embeddings.py
# ```

if __name__ == "__main__":
    import asyncio

    SGLangEmbeddings = modal.Cls.from_name(
        "example-sglang-embeddings", "SGLangEmbeddings"
    )
    print("Calling inference server...")
    try:
        url = SGLangEmbeddings._experimental_get_flash_urls()[0]
        asyncio.run(probe_health(url))
        asyncio.run(test_single_embedding(url))
        asyncio.run(test_similarity(url))
    except modal.exception.NotFoundError as e:
        raise Exception(
            f"To test GPU snapshots, deploy first with: modal deploy {__file__}"
        ) from e


# ## Use the embeddings API

# The server exposes an OpenAI-compatible `/v1/embeddings` endpoint:

# ```python
# import requests
#
# url = "https://your-workspace--example-sglang-embeddings-sglang.modal.direct"
#
# # Single embedding
# response = requests.post(
#     f"{url}/v1/embeddings",
#     json={
#         "model": "Qwen/Qwen3-Embedding-8B",
#         "input": "What is machine learning?",
#     }
# )
# embedding = response.json()["data"][0]["embedding"]
#
# # Batch embeddings
# response = requests.post(
#     f"{url}/v1/embeddings",
#     json={
#         "model": "Qwen/Qwen3-Embedding-8B",
#         "input": ["First text", "Second text", "Third text"],
#     }
# )
# embeddings = [d["embedding"] for d in response.json()["data"]]
#
# # With task instruction (improves retrieval quality)
# response = requests.post(
#     f"{url}/v1/embeddings",
#     json={
#         "model": "Qwen/Qwen3-Embedding-8B",
#         "input": "Instruct: Retrieve documents about AI\nQuery: What is deep learning?",
#     }
# )
#
# # Matryoshka embeddings (reduced dimensions for storage efficiency)
# response = requests.post(
#     f"{url}/v1/embeddings",
#     json={
#         "model": "Qwen/Qwen3-Embedding-8B",
#         "input": "Some text to embed",
#         "dimensions": 1024,  # Reduce from 4096 to 1024
#     }
# )
# ```
