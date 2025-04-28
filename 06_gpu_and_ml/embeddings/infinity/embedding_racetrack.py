# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/infinity/embedding_racetrack.py::backbone"]
# ---

# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode 10,000 images from the [wildflow sweet-coral dataset](https://huggingface.co/datasets/wildflow/sweet-corals "huggingface/wildflow/sweet-coral")
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# ## BLUF (bottom line up front)
# We have found that setting concurrency to 2 and then maximizing the batchsize will maximize GPU utilization.
# If you then enable more containers, throughput goes through the roof.

## Setup
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import modal
import numpy as np
from modal.volume import FileEntry
from more_itertools import chunked
from PIL.Image import Image

app_name = "example-embedder"

# ## CLI: Key Parameters
# There are three ways to parallelize inference for this usecase: via batching (which happens internal to Infinity),
# by packing individual GPU(s) with multiple copies of the model, and by fanning out across multiple containers.
# Here are some parmaeters for controlling these factors:
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `allow_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app.
# This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up.
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)

gpu: str = "H100"
max_containers: int = 1
allow_concurrent_inputs: int = 10
batch_size: int = 500
image_cap: int = 20000

# This timeout parameter only needs to include the maximum amount of time it takes to build a batch
# (e.g. read one batch worth of data). For huge batches it could take a few minutes.
timeout_seconds: int = 4 * 60

# This model parameter should point to a model on huggingface that is supported by Infinity.
# Note that your specifically chosen model might require specialized imports when
# designing the image. This [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT")
# takes about 4-10s to load into memory.
model_name = "openai/clip-vit-base-patch16"  # 599 MB

# ## Data setup
# We use a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# to store the images we want to encode.
vol_name = "sweet-coral-db-20k"
vol_mnt = Path("/data")
vol = modal.Volume.from_name(vol_name, environment_name="ben-dev")


def find_images_to_encode(image_cap: int = 1) -> list[FileEntry]:
    """
    You can modify this function to find an return a list of your image paths.
    """

    im_path_list = list(
        filter(lambda x: x.path.endswith(".jpg"), vol.listdir("/data", recursive=True))
    )
    print(f"Found {len(im_path_list)} JPEGs, ", end="")

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if image_cap > 0:
        im_path_list = im_path_list[: min(len(im_path_list), image_cap)]
    return im_path_list


# ## Define the image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "infinity_emb[all]==0.0.76",  # for Infinity inference lib
            "sentencepiece",  # for this particular chosen model
            "more-itertools",  # for elegant list batching
            "torchvision",  # for fast image loading
        ]
    )
    .env({"INFINITY_MODEL_ID": model_name, "HF_HOME": "/data"})
)

# Initialize the app
app = modal.App(app_name, image=simple_image, volumes={vol_mnt: vol})

# Imports inside the container
with simple_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image


# ## Inference app
# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
@app.cls(
    gpu=gpu,
    image=simple_image,
    volumes={vol_mnt: vol},
    timeout=timeout_seconds,
    allow_concurrent_inputs=allow_concurrent_inputs,
    max_containers=max_containers,
)
class InfinityEngine:
    n_engines: int = allow_concurrent_inputs

    @modal.enter()
    async def init_engines(self):
        print(f"Loading {self.n_engines} models... ", end="")
        # Init s
        self.engine_queue: asyncio.Queue[AsyncEmbeddingEngine] = asyncio.Queue()
        # start N engines and put them in
        start = perf_counter()
        for _ in range(self.n_engines):
            engine = AsyncEmbeddingEngine.from_args(
                EngineArgs(
                    model_name_or_path=model_name,
                    batch_size=batch_size,
                    model_warmup=False,
                    engine=InferenceEngine.torch,
                    dtype=Dtype.float16,
                    device="cuda",
                )
            )
            await engine.astart()
            await self.engine_queue.put(engine)
        print(f"Took {perf_counter() - start:.4}s.")

    def make_batch(self, im_path_list: list[FileEntry]) -> list[Image]:
        # Convert to a list of paths
        def readim(impath: FileEntry):
            """Read with torch, convert back to PIL for Infinity"""
            return to_pil_image(read_image(str(vol_mnt / impath.path)))

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            images = list(executor.map(readim, im_path_list))

        return images

    @modal.method()
    async def embed(self, images: list[str]) -> tuple[float, float]:
        # (0) Grab an engine from the queue
        engine = await self.engine_queue.get()

        try:
            # (1) Load batch of image data
            st = perf_counter()
            images = self.make_batch(images)
            batch_elapsed = perf_counter() - st

            # (2) Encode the batch
            st = perf_counter()
            embedding, _ = await engine.image_embed(images=images)
            embed_elapsed = perf_counter() - st
        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) Housekeeping
        print(f"Time to load batch: {batch_elapsed:.2f}s")
        print(f"Time to embed batch: {embed_elapsed:.2f}s")

        # (4) You may wish to return the embeddings themselves here
        return embed_elapsed, len(images)

    @modal.exit()
    async def exit(self) -> None:
        for _ in range(self.n_engines):
            engine = await self.engine_queue.get()
            await engine.astop()


# ## Local Entrypoint
# This code is run on your machine.
@app.local_entrypoint()
def main():
    # (1) Init the model inference app
    start_time = perf_counter()
    embedder = InfinityEngine()

    # (2) Catalog data: modify `find_images_to_encode` at the top of this file for your usecase.
    im_path_list = find_images_to_encode(image_cap)
    n_ims = len(im_path_list)
    print(f"using {n_ims}.")

    # (3) Embed batches via remote `map` call
    times, batchsizes = [], []
    for time, batchsize in embedder.embed.map(chunked(im_path_list, batch_size)):
        times.append(time)
        batchsizes.append(batchsize)

    # (4) Log
    if n_ims > 0:
        total_duration = perf_counter() - start_time
        total_throughput = n_ims / total_duration
        embed_througputs = np.array(
            [batchsize / time for batchsize, time in zip(batchsizes, times)]
        )
        avg_throughput = embed_througputs.mean()
        std_throughput = embed_througputs.std()

        log_msg = (
            f"simple_volume.py::batch_size={batch_size}::n_ims={n_ims}::concurrency={allow_concurrent_inputs}\n"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{avg_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (std dev):\t{std_throughput:.2f} im/s\n"
        )

        print(log_msg)
