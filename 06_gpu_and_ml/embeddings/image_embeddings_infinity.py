# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/image_embeddings_infinity.py::main"]
# ---

# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode the [Microsoft Cats & Dogs dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# ## Conclusions
# ### BLUF (Bottom Line Up Front)
# Set concurrency (`max_concurrent_inputs`) to 4, and set `batch_size` between 50-500.
# To set `max_containers`, divide the total number of inputs by `max_concurrent_inputs*batchsize`
# (note: if you have a massive dataset, keep an eye out for diminishing returns on `max_containers`; but
# Modal should handle that for you!).
# Be sure to preprocess your data in the same manner that the model is expecting (e.g., resizing images).
# If you only want to use one container, increase `batch_size` until you are maxing
# out the GPU (but keep concurrency, `max_concurrent_inputs`, capped around 4). The example herein achieves
# upward of 750 images / second overall throughput (not including initial Volume setup time).

# ### Why?
# While batchsize maximizes GPU utilization, the time to form a batch (ie reading images)
# will ultimately overtake inference, whether due to I/O, sending data across a wire, etc.
# We can make up for this by using idle GPU cores to store additional copies of the model:
# this _GPU packing_ is achieved via an async queue and the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# decorator. Once you nail down `batch_size` you can crank up the number of containers to distribute the
# computational load. High values of concurrency has diminishing returns, we believe,
# because we are already throttling the CPU with multi-threaded dataloading. The demo herein
# achieves upward of 750 images / second, and that will increase for larger datasets where the model loading
# time becomes increasingly negligable.

# ## Local env imports
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Iterator, TypeVar

import modal

# ## Key Parameters
# There are three ways to parallelize inference for this usecase: via batching (which happens internal to Infinity),
# by packing individual GPU(s) with multiple copies of the model, and by fanning out across multiple containers.
# Here are some parameters for controlling these factors:
# * `max_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app. This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up.
# * `memory_request` amount of RAM requested per container
# * `core_request` number of logical cores requested per container
# * `threads_per_core` oversubscription factor for parallelized I/O (image reading)
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)
max_concurrent_inputs: int = 4
gpu: str = "L4"
max_containers: int = 50
memory_request: float = 5 * 1024  # MB->GB
core_request: float = 4
threads_per_core: int = 8
batch_size: int = 100
image_cap: int = -1

# This timeout caps the maximum time a single function call is allowed to take. In this example, that
# includes reading a batch-worth of data and running inference on it. When `batch_size` is large (e.g. 5000)
# and with a large value of `max_concurrent_inputs`, where a batch may sit in a queue for a while,
# this could take several minutes.
timeout_seconds: int = 10 * 60

# ## Data and Model Specification
# This model parameter should point to a model on HuggingFace that is supported by Infinity.
# Note that your selected model might require specialized imports when
# designing the image in the next section. This [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT")
# takes about 4-10s to load into memory.
model_name = "openai/clip-vit-base-patch16"  # 599 MB
model_input_shape = (224, 224)

# We will use a high-performance [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# both to cache model weights and to store images we want to encode. The details of
# setting this volume up are below. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).
# Here, we just need to name it so that we can instantiate
# the Modal application.
# You may need to [set up a secret](https://modal.com/secrets/) to access HuggingFace datasets
hf_secret = modal.Secret.from_name("huggingface-secret")
# Change this global variable to use a different HF dataset:
hf_dataset_name = "microsoft/cats_vs_dogs"
# This name is important for referencing the volume in other apps or for [browsing](https://modal.com/storage):
vol_name = "example-embedding-data"
# This is the location within the container that this Volume will be mounted:
vol_mnt = Path("/data")
# Finally, the Volume object can be created:
data_volume = modal.Volume.from_name(vol_name, create_if_missing=True)


# ## Define the image
infinity_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pillow==11.3.0",  # for Infinity input typehint
            "datasets==4.0.0",  # for huggingface data download
            "hf_transfer==0.1.9",  # for fast huggingface data download
            "huggingface_hub[hf_xet]==0.33.2",
            "tqdm==4.67.1",  # progress bar for dataset download
            "sentencepiece==0.2.0",  # for this particular chosen model
            "torchvision==0.22.1",  # for fast image loading
            "infinity_emb[all]==0.0.76",  # for Infinity inference lib
            "optimum==1.26.1",  # need to pin this because newer version requires
        ]
    )
    .env(
        {
            "HF_HOME": vol_mnt.as_posix(),  # For model and data caching in our Volume
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # For fast data transfer
        }
    )
)

# Initialize the app
app = modal.App(
    "example-image-embeddings-infinity",
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    secrets=[hf_secret],
)

# Imports inside the container
with infinity_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from PIL.Image import Image
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image

## Dataset Downloading and Setup
# ## Data setup
# We use a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# to store images we want to encode. We download them from Huggingface into a Volume and then preprocess
# them to 224 x 224 JPEGs. The selected model, `openai/clip-vit-base-patch16`, was trained on 224 x 224
# sized images. If you skip this preprocess resize step, Infinity will handle image resizing for you-
# at a severe penalty to inference throughput.

# Note that Modal Volumes are optimized for datasets on the order of 50,000 - 500,000
# files and directories. If you have a larger dataset, you may need to consider other storage
# options such as a [CloudBucketMount](https://modal.com/docs/examples/rosettafold).


@app.function(
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    max_containers=1,  # We only want one container to handle volume setup
    cpu=core_request,  # HuggingFace will use multi-process parallelism to download
    timeout=timeout_seconds,  # if using a large HF dataset, this may need to be longer
)
def catalog_jpegs(dataset_namespace: str, cache_dir: str, image_cap: int):
    """
    This function checks the volume for JPEGs and, if needed, calls `download_to_volume`
    which pulls a HuggingFace dataset into the mounted volume.
    """

    def download_to_volume(dataset_namespace: str, cache_dir: str):
        """
        This function caches a hugginface dataset to the path specified in your `HF_HOME` environment
        variable, which we set when creating the image so as to point to a Modal Volume.
        """
        from datasets import load_dataset
        from torchvision.io import write_jpeg
        from torchvision.transforms import Compose, PILToTensor, Resize
        from tqdm import tqdm

        # Load cache to HF_HOME
        ds = load_dataset(
            dataset_namespace,
            split="train",
            num_proc=os.cpu_count(),  # this will be capped by huggingface based on the number of shards
        )

        # Create an `extraction` cache dir where we will create explicit JPEGs
        mounted_cache_dir = vol_mnt / cache_dir
        mounted_cache_dir.mkdir(exist_ok=True, parents=True)

        # Preprocessing pipeline: resize now instead of on-the-fly
        preprocessor = Compose(
            [
                Resize(model_input_shape),
                PILToTensor(),
            ]
        )

        def preprocess_img(idx, example):
            """
            Applies preprocessor and write as jpeg with TurboJPEG (via torchvision).
            """
            # Define output path
            write_path = mounted_cache_dir / f"img{idx:07d}.jpg"
            if write_path.is_file():
                return

            # Here, `example["image"]` is a `PIL.Image.Image`
            preprocessed = preprocessor(example["image"].convert("RGB"))

            # Write to modal.Volume
            write_jpeg(preprocessed, write_path)

        # This is a parallelized pre-processing loop that opens compressed images,
        # preprocesses them to the size expected by our model, and writes as a JPEG.
        for idx, ex in tqdm(enumerate(ds), total=len(ds), desc="Caching images"):
            if (image_cap > 0) and (idx >= image_cap):
                break
            preprocess_img(idx, ex)

        data_volume.commit()

    ds_preptime_st = perf_counter()

    def list_all_jpegs(subdir: os.PathLike = "/") -> list[os.PathLike]:
        """
        Searches a subdir within your volume for all JPEGs.
        """
        return [
            x.path
            for x in data_volume.listdir(subdir.as_posix())
            if x.path.endswith(".jpg")
        ]

    # Check for extracted-JPEG cache dir within the volume
    if (vol_mnt / cache_dir).is_dir():
        im_path_list = list_all_jpegs(cache_dir)
        n_ims = len(im_path_list)
    else:
        n_ims = 0
        print("The cache dir was not found...")

    # If needed, download dataset to a vol
    if (n_ims < image_cap) or (n_ims == 0):
        print(f"Found {n_ims} JPEGs; checking for more on HuggingFace.")
        download_to_volume(dataset_namespace, cache_dir)
        # Try again
        im_path_list = list_all_jpegs(cache_dir)
        n_ims = len(im_path_list)

    # [optional] Cap the number of images to process
    print(f"Found {n_ims} JPEGs in the Volume.", end="")
    if image_cap > 0:
        im_path_list = im_path_list[: min(image_cap, len(im_path_list))]
    print(f"using {len(im_path_list)}.")

    # Time it
    ds_time_elapsed = perf_counter() - ds_preptime_st
    return im_path_list, ds_time_elapsed


T = TypeVar("T")  # generic type for chunked typehints


def chunked(seq: list[T], subseq_size: int) -> Iterator[list[T]]:
    """
    Helper function that chunks a sequence into subsequences of length `subseq_size`.
    """
    for i in range(0, len(seq), subseq_size):
        yield seq[i : i + subseq_size]


# ## Inference app
# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
# Note that the variable `max_concurrent_inputs` is used to set `max_inputs`
# in (1) the [modal.concurrent](https://modal.com/docs/guide/concurrent-inputs#input-concurrency)
# decorator, and (2) the `n_engines` class property.
# In `init_engines`, we are creating exactly one inference
# engine for each concurrently-passed batch of data. This is critical for packing a GPU with
# multiple simultaneously operating models. The [@modal.enter](https://modal.com/docs/reference/modal.enter#modalenter)
# decorator ensures that this method is called once per container, on startup (and `exit` is
# run once, on shutdown).
@app.cls(
    gpu=gpu,
    cpu=core_request,
    memory=5 * 1024,  # MB -> GB
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    timeout=timeout_seconds,
    max_containers=max_containers,
)
@modal.concurrent(max_inputs=max_concurrent_inputs)
class InfinityEngine:
    n_engines: int = max_concurrent_inputs

    @modal.enter()
    async def init_engines(self):
        """
        On container start, starts `self.n_engines` copies of the selected model
        and puts them in an async queue.
        """
        print(f"Loading {self.n_engines} models... ", end="")
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

    def read_batch(self, im_path_list: list[os.PathLike]) -> list["Image"]:
        """
        Read a batch of data. Infinity is expecting PIL.Image.Image type
        inputs, but it's faster to read from disk with torchvision's `read_image`
        and convert to PIL than it is to read directly with PIL.

        This process is parallelized over the batch with multithreaded data reading.
        The number of threads is 4 per core, which is based on the batchsize.
        """

        def readim(impath: os.PathLike):
            """Read with torch, convert back to PIL for Infinity"""
            return to_pil_image(read_image(str(vol_mnt / impath)))

        with ThreadPoolExecutor(
            max_workers=os.cpu_count() * threads_per_core
        ) as executor:
            images = list(executor.map(readim, im_path_list))

        return images

    @modal.method()
    async def embed(self, images: list[os.PathLike]) -> tuple[float, float]:
        """
        This is the workhorse function. We select a model, prepare a batch,
        execute inference, and return the time elapsed. You probably want
        to return the embeddings in your usecase.
        """
        # (0) Grab an engine from the queue
        engine = await self.engine_queue.get()

        try:
            # (1) Load batch of image data
            images = self.read_batch(images)

            # (2) Encode the batch
            st = perf_counter()
            embedding, _ = await engine.image_embed(images=images)
            embed_elapsed = perf_counter() - st
        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) You may wish to return the embeddings themselves here
        return embed_elapsed, len(images)

    @modal.exit()
    async def exit(self) -> None:
        """
        Shut down each of the engines.
        """
        for _ in range(self.n_engines):
            engine = await self.engine_queue.get()
            await engine.astop()


# ## Local Entrypoint
# This backbone code is run on your machine. It starts up the app,
# catalogs the data, and via the remote `map` call, parses the data
# with the Infinity embedding engine. The embedder.embed executions
# across the batches are autoscaled depending on the app parameters
# `max_containers` and `max_concurrent_inputs`.
@app.local_entrypoint()
def main():
    start_time = perf_counter()

    # (1) Catalog data: modify `catalog_jpegs` to fetch batches of your data.
    extracted_path = Path("extracted") / hf_dataset_name
    im_path_list, vol_setup_time = catalog_jpegs.remote(
        dataset_namespace=hf_dataset_name, cache_dir=extracted_path, image_cap=image_cap
    )
    print(f"Took {vol_setup_time:.2f}s to setup volume.")
    n_ims = len(im_path_list)

    # (2) Init the model inference app
    start_time = perf_counter()
    embedder = InfinityEngine()

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
            f"EmbeddingRacetrack{gpu}::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={max_concurrent_inputs}::"
            f"max_containers={max_containers}::cores={core_request}\n"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tVolume setup time:\t{vol_setup_time / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{avg_throughput:.2f} im/s\n"
        )

        print(log_msg)
