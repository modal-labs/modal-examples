# # Run multiple instances of a model on a single GPU
#
# Many models are small enough to fit multiple instances onto a single GPU.
# Doing so can dramatically reduce the number of GPUs needed to handle demand.
#
# We use `allow_concurrent_inputs` to allow multiple connections into the container
# We load the model instances into a FIFO queue to ensure only one http handler can access it at once

import asyncio
import time
from contextlib import asynccontextmanager

import modal

image = modal.Image.debian_slim().pip_install("sentence-transformers==3.2.0")

app = modal.App("gpu-packing", image=image)


# ModelPool holds multiple instances of the model, using a queue
class ModelPool:
    def __init__(self):
        self.pool: asyncio.Queue = asyncio.Queue()

    async def put(self, model):
        await self.pool.put(model)

    # We provide a context manager to easily acquire and release models from the pool
    @asynccontextmanager
    async def acquire_model(self):
        model = await self.pool.get()
        try:
            yield model
        finally:
            await self.pool.put(model)


with image.imports():
    from sentence_transformers import SentenceTransformer


@app.cls(
    gpu="A10G",
    concurrency_limit=1,  # Max one container for this app, for the sake of demoing concurrent_inputs
    allow_concurrent_inputs=100,  # Allow concurrent inputs into our single container.
)
class Server:
    def __init__(self, n_models=10):
        self.model_pool = ModelPool()
        self.n_models = n_models

    @modal.build()
    def download(self):
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        model.save("/model.bge")

    @modal.enter()
    async def load_models(self):
        # Boot N models onto the gpu, and place into the pool
        t0 = time.time()
        for i in range(self.n_models):
            model = SentenceTransformer("/model.bge", device="cuda")
            await self.model_pool.put(model)

        print(f"Loading {self.n_models} models took {time.time() - t0:.4f}s")

    @modal.method()
    def prewarm(self):
        pass

    @modal.method()
    async def predict(self, sentence):
        # Block until a model is available
        async with self.model_pool.acquire_model() as model:
            # We now have exclusive access to this model instance
            embedding = model.encode(sentence)
            await asyncio.sleep(
                0.2
            )  # Simulate extra inference latency, for demo purposes
        return embedding.tolist()


@app.local_entrypoint()
async def main(n_requests: int = 100):
    # We benchmark with 100 requests in parallel.
    # Thanks to allow_concurrent_inputs=100, 100 requests will enter .predict() at the same time.

    sentences = ["Sentence {}".format(i) for i in range(n_requests)]

    # Baseline: a server with a pool size of 1 model
    print("Testing Baseline (1 Model)")
    t0 = time.time()
    server = Server(n_models=1)
    server.prewarm.remote()
    print("Container boot took {:.4f}s".format(time.time() - t0))

    t0 = time.time()
    async for result in server.predict.map.aio(sentences):
        pass
    print(f"Inference took {time.time() - t0:.4f}s\n")

    # Packing: a server with a pool size of 10 models
    # Note: this increases boot time, but reduces inference time
    print("Testing Packing (10 Models)")
    t0 = time.time()
    server = Server(n_models=10)
    server.prewarm.remote()
    print("Container boot took {:.4f}s".format(time.time() - t0))

    t0 = time.time()
    async for result in server.predict.map.aio(sentences):
        pass
    print(f"Inference took {time.time() - t0:.4f}s\n")
