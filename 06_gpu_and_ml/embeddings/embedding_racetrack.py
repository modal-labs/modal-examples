# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/embedding_racetrack.py::main"]
# ---

# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode 20,000 images from the [wildflow sweet-coral dataset](https://huggingface.co/datasets/wildflow/sweet-corals "huggingface/wildflow/sweet-coral")
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# ## Conclusions
# ### BLUF (Bottom Line Up Front)
# Set concurrency (`max_concurrent_inputs`)to 2, and set `batch_size` between 100-500.
# To set `max_containers`, divide the total number of inputs by `max_concurrent_inputs*batchsize`
# (note: if you have a massive dataset, keep an eye out for diminishing returns on max_containers; but
# Modal should handle that for you!).
# Be sure to preprocess your data in the same manner that the model is expecting (e.g., resizing images).
# If you only want to use one container, increase `batch_size` until you are maxing
# out the GPU (but keep concurrency, `max_concurrent_inputs`, set to 2).
# ### Why?
# While batchsize maximizes GPU utilization, the time to form a batch (ie reading images)
# will ultimately overtake inference, whether due to I/O, sending data across a wire, etc.
# We can make up for this by using idle GPU cores to store additional copies of the model:
# This _GPU packing_ is achieved via an async queue and the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# decorator. Once you nail down `batch_size` you can crank up the number of containers to distribute the
# computational load. High values of concurrency has diminishing returns, we believe,
# because we are already throttling the CPU with multi-threaded dataloading. The demo herein
# achieves upward of 650 images / second, and that will increase for larger datasets where the model loading
# time becomes increasingly negligable

# ## Setup
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import modal

# ## Key Parameters
# There are three ways to parallelize inference for this usecase: via batching (which happens internal to Infinity),
# by packing individual GPU(s) with multiple copies of the model, and by fanning out across multiple containers.
# Here are some parameters for controlling these factors:
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `max_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app. This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up.
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)

batch_size: int = 100
max_concurrent_inputs: int = 2
gpu: str = "L4"
max_containers: int = 40
image_cap: int = 20000

# This timeout caps the maximum time a single function call is allowed to take. In this example, that
# includes reading a batch-worth of data and running inference on it. When `batch_size`` is large (e.g. 5000)
# and with a large value of `max_concurrent_inputs`, where a batch may sit in a queue for a while,
# this could take several minutes.
timeout_seconds: int = 5 * 60

# This model parameter should point to a model on huggingface that is supported by Infinity.
# Note that your selected model might require specialized imports when
# designing the image in the next section. This [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT")
# takes about 4-10s to load into memory.
model_name = "openai/clip-vit-base-patch16"  # 599 MB

# ## Data setup
# We use a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# to store the images we want to encode. We have resized them to 224 x 224 as that is
# what the `openai/clip-vit-base-patch16` model was trained on.
vol_name = "sweet-coral-db-20k"
vol_mnt = Path("/data")
vol = modal.Volume.from_name(vol_name, environment_name="ben-dev")


def find_images_to_encode(image_cap: int = 1, batch_size: int = 1) -> list[os.PathLike]:
    """
    You can modify this function to find and return your data paths.
    """

    im_path_list = list(
        filter(
            lambda x: x.path.endswith(".jpg"), vol.listdir("/resized", recursive=True)
        )
    )
    print(f"Found {len(im_path_list)} JPEGs, ", end="")

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if image_cap > 0:
        im_path_list = im_path_list[: min(len(im_path_list), image_cap)]

    n_ims = len(im_path_list)
    print(f"using {n_ims}, ", end="")

    # Convert this list of modal.volume.FileEntry objects into a list of paths
    im_path_list = [x.path for x in im_path_list]

    # chunked re-shapes a list into a list of lists (each sublist of size batch_size)
    return im_path_list


# ## Define the image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pillow",
            "infinity_emb[all]==0.0.76",  # for Infinity inference lib
            "sentencepiece",  # for this particular chosen model
            "torchvision",  # for fast image loading
        ]
    )
    .env({"INFINITY_MODEL_ID": model_name, "HF_HOME": "/data"})
)

# Initialize the app
app = modal.App("example-embedder", image=simple_image, volumes={vol_mnt: vol})

# Imports inside the container
with simple_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from PIL.Image import Image
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image


# ## Inference app
# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
@app.cls(
    gpu=gpu,
    image=simple_image,
    volumes={vol_mnt: vol},
    timeout=timeout_seconds,
    max_containers=max_containers,
)
@modal.concurrent(max_inputs=max_concurrent_inputs)
class InfinityEngine:
    n_engines: int = max_concurrent_inputs

    @modal.enter()
    async def init_engines(self):
        print(f"Loading {self.n_engines} models... ", end="")
        # Start N engines and put them in an async queue
        self.engine_queue: asyncio.Queue[AsyncEmbeddingEngine] = asyncio.Queue()
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

    def make_batch(self, im_path_list: list[os.PathLike]) -> list["Image"]:
        # Convert to a list of paths
        def readim(impath: os.PathLike):
            """Read with torch, convert back to PIL for Infinity"""
            return to_pil_image(read_image(str(vol_mnt / impath)))

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            images = list(executor.map(readim, im_path_list))

        return images

    @modal.method()
    async def embed(self, images: list[os.PathLike]) -> tuple[float, float]:
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


def chunked(seq, size):
    """
    Chunks a sequence into subsequences of length `size`
    """
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ## Local Entrypoint
# This code is run on your machine.
@app.local_entrypoint()
def main():
    # (1) Init the model inference app
    start_time = perf_counter()
    embedder = InfinityEngine()

    # (2) Catalog data: modify `find_images_to_encode` to fetch batches of your data.
    im_path_list = find_images_to_encode(image_cap, batch_size)
    n_ims = len(im_path_list)

    # (3) Embed batches via remote `map` call
    times, batchsizes = [], []
    for time, batchsize in embedder.embed.map(chunked(im_path_list, batch_size)):
        times.append(time)
        batchsizes.append(batchsize)

    # (4) Log
    if n_ims > 0:
        total_duration = perf_counter() - start_time
        total_throughput = n_ims / total_duration
        embed_throughputs = [
            batchsize / time for batchsize, time in zip(batchsizes, times)
        ]
        avg_throughput = sum(embed_throughputs) / len(embed_throughputs)

        log_msg = (
            f"EmbeddingRacetrack::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={max_concurrent_inputs}::"
            f"max_containers={max_containers}\n"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{avg_throughput:.2f} im/s\n"
        )

        print(log_msg)
