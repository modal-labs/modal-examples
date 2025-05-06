# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/image_embedding_infinity.py::main"]
# ---

# # A Recipe for Throughput Maximization: GPU Packing with Infinity Inference
# In certain applications, the bottom line comes to *throughput*: process a batch of inputs as fast as possible.
# This example presents a Modal recipe for maximizing image embedding throughput using the
# [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"),
# a popular inference engine that manages asychronous queuing and model serving.
#
# Check out [this example](https://modal.com/docs/examples/image_embedding_th_compile) to see how
# to use Modal to natively accomplish these features and achieve even higher throughput (nearly 2x)!
#

# ## Conclusions
# ### BLUF (Bottom Line Up Front)
# Set concurrency (`max_concurrent_inputs`) to 2, and set `batch_size` around 100.
# To get maximum throughput at any cost, set buffer_containers to 10; otherwise set it to None
# and set max_containers based on your budget.
# Be sure to preprocess your data in the same manner that the model is expecting (e.g., resizing images).
# If you only want to use one container, increase `batch_size` until you are maxing
# out the GPU (but keep concurrency, `max_concurrent_inputs`, capped around 2). The example herein achieves
# around 700 images / second overall throughput, embedding the entire
# [cats vs dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# dataset in about 30s.

# ### Why?
# While batch size maximizes GPU utilization, the time to form a batch (ie reading images)
# will ultimately overtake inference, whether due to I/O, sending data across a wire, etc.
# We can make up for this by using idle GPU cores to store additional copies of the model:
# this high-level form of _GPU packing_ is achieved via an async queue and the
# [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# decorator, called indirectly through [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options).

# Once you nail down an effective `batch_size` for your problem, you can crank up the number of containers
# to distribute the computational load. Set buffer_containers > 0 so that Modal continuously spins up more
# and more containers until the task is complete; otherwise set it to None, and use max_containers to cap
# the number of containers allowed.

# High values of concurrency has diminishing returns, we believe,
# because we are already throttling the CPU with multi-threaded dataloading, and because of the way
# Infinity handles batches of inputs. We have a [more advanced example](https://modal.com/docs/examples/image_embedding_th_compile)
# that directly uses torch.compile (without Infinity), which uses Modal's native capabilities to
# manage queuing etc., and can take better advantage of the concurrency feature.

# The demo herein should achieve around 700 images/second (around 200-300 images/second per model),
# but that number will increase dramatically for larger datasets where more and more containers spawn.
# We have clocked 9500 images/second using Infinity on a 1M image dataset, and twice that in our torch.compile
# example.

# ## Local env imports
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterator

import modal

# ## Key Parameters
# There are three ways to parallelize inference for this usecase: via batching,
# by packing individual GPU(s) with multiple copies of the model, and by fanning out across multiple containers.
#
# Modal provides two ways to dynamically parameterize classes: through
# [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options)
# and through
# [modal.parameter](https://modal.com/docs/reference/modal.parameter#modalparameter).
# The app.local_entrypoint() main function at the bottom of this example uses these
# features to dynamically construct the inference engine class wrapper. One feature
# that is not currently support via `with_options` is the `buffer_containers` parameter.
# This tells Modal to pre-emptively warm a number of containers before they are strictly
# needed. In other words it tells Modal to continuously fire up more and more containers
# until throughput is saturated.
buffer_containers: int = 10
# If you _don't_ want to use this, set `buffer_containers = None`. The rest of the parameters
# are discussed by their implementation by the local_entrypoint.

# ## Dataset, Model, and Image Setup
# This example uses HuggingFace to download data and models. We will use a high-performance
# [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# both to cache model weights and to store the
# [image dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# that we want to embed.

# ### Volume Initialization
# You may need to [set up a secret](https://modal.com/secrets/) to access HuggingFace datasets
hf_secret = modal.Secret.from_name("huggingface-secret")

# This name is important for referencing the volume in other apps or for
# [browsing](https://modal.com/storage):
vol_name = "example-embedding-data"

# This is the location within the container where this Volume will be mounted:
vol_mnt = Path("/data")

# Finally, the Volume object can be created:
data_volume = modal.Volume.from_name(vol_name, create_if_missing=True)

# ### Define the image
infinity_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pillow",  # for Infinity input typehint
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "tqdm",  # progress bar for dataset download
            "infinity_emb[all]==0.0.76",  # for Infinity inference lib
            "sentencepiece",  # for this particular chosen model
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            # For fast HuggingFace model and data caching and download in our Volume
            "HF_HOME": vol_mnt.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Initialize the app
app = modal.App(
    "example-infinity-embedder",
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

# ## Dataset Setup
# We use a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# to store the images we want to encode. For your usecase, can simply replace the
# function `catalog_jpegs` with any function that returns a list of image paths. Just make
# sure that it's returning the _paths_: we are going to
# [map](https://modal.com/docs/reference/modal.Function#map) these inputs between containers
# so that the inference class can simply read them directly from the Volume. If you are
# shipping the images themselves across the wire, that will likely bottleneck throughput.

# Note that Modal Volumes are optimized for datasets on the order of 50,000 - 500,000
# files and directories. If you have a larger dataset, you may need to consider other storage
# options such as a [CloudBucketMount](https://modal.com/docs/examples/rosettafold).

# A note on preprocessing: Infinity will handle resizing and other preprocessing in case
# your images are not the same size as what the model is expecting; however, this will
# significantly degrade throughput. We recommend batch-processing (if possible).


@app.function(
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    max_containers=1,  # We only want one container to handle volume setup
    cpu=4,  # HuggingFace will use multi-process parallelism to download
    timeout=10 * 60,  # if using a large HF dataset, this may need to be longer
)
def catalog_jpegs(
    dataset_namespace: str,  # a HuggingFace path like `microsoft/cats_vs_dogs`
    cache_dir: str,  # a subdir where the JPEGs will be extracted into the volume long-form
    image_cap: int,  # hard cap on the number of images to be processed (e.g. for timing, debugging)
    model_input_shape: tuple[int, int, int],  # JPEGs will be preprocessed to this shape
    threads_per_cpu: int = 4,  # threads per CPU for I/O oversubscription
) -> tuple[
    list[os.PathLike],  # the function returns a list of paths,
    float,  # and the time it took to prepare
]:
    """
    This function checks the volume for JPEGs and, if needed, calls `download_to_volume`
    which pulls a HuggingFace dataset into the mounted volume, preprocessing along the way.
    """

    def download_to_volume(dataset_namespace: str, cache_dir: str):
        """
        This function:
        (1) caches a HuggingFace dataset to the path specified in your `HF_HOME` environment
        variable, which is pointed to a Modal Volume during creation of the image above.
        (2) unpacks the dataset and preprocesses them; this could be done in several different
        ways, but we want to do it all once upfront so as not to confound the timing tests later.
        """
        from datasets import load_dataset
        from torchvision.io import write_jpeg
        from torchvision.transforms import Compose, PILToTensor, Resize
        from tqdm import tqdm

        # Load dataset cache to HF_HOME
        ds = load_dataset(
            dataset_namespace,
            split="train",
            num_proc=os.cpu_count(),  # this will be capped by huggingface based on the number of shards
        )

        # Create an `extraction` cache dir where we will create explicit JPEGs
        mounted_cache_dir = vol_mnt / cache_dir
        mounted_cache_dir.mkdir(exist_ok=True, parents=True)

        # Preprocessing pipeline: resize in bulk now instead of on-the-fly later
        preprocessor = Compose(
            [
                Resize(model_input_shape),
                PILToTensor(),
            ]
        )

        def preprocess_img(idx, example):
            """
            Applies preprocessor and write as jpeg with TurboJPEG (via TorchVision).
            """
            # Define output path
            write_path = mounted_cache_dir / f"img{idx:07d}.jpg"
            # Skip if already done
            if write_path.is_file():
                return

            # Process
            preprocessed = preprocessor(example["image"].convert("RGB"))

            # Write to modal.Volume
            write_jpeg(preprocessed, write_path)

        # Note: the optimization of this loop really depends on your preprocessing stack.
        # You could use ProcessPool if there is significant work per image, or even
        # GPU acceleration and batch preprocessing. We keep it simple here for the example.
        futures = []
        with ThreadPoolExecutor(max_workers=os.cpu_count * threads_per_cpu) as executor:
            for idx, ex in enumerate(ds):
                if image_cap > 0 and idx >= image_cap:
                    break
                futures.append(executor.submit(preprocess_img, idx, ex))

            # Progress bar over completed futures
            for _ in tqdm(
                as_completed(futures), total=len(futures), desc="Caching images"
            ):
                pass  # result() is implicitly called by as_completed()

        # Save changes
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

    # Check for extracted-JPEG cache dir within the modal.Volume
    if (vol_mnt / cache_dir).is_dir():
        im_path_list = list_all_jpegs(cache_dir)
        n_ims = len(im_path_list)
    else:
        n_ims = 0
        print("The cache dir was not found...")

    # If needed, download dataset to a modal.Volume
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


def chunked(seq: list[os.PathLike], subseq_size: int) -> Iterator[list[os.PathLike]]:
    """
    Helper function that chunks a sequence into subsequences of length `subseq_size`.
    """
    for i in range(0, len(seq), subseq_size):
        yield seq[i : i + subseq_size]


# ## Inference app
# Here we define a [modal.cls](https://modal.com/docs/reference/modal.Cls#modalcls)
# that wraps [Infinity's AsyncEmbeddingEngine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").
# Some important observations:
# 1. Infinity handles asynchronous queuing internally. This is actually redundant with Modal's
# concurrency feature, but we found that using them together still helps.
# In [another example](https://modal.com/docs/examples/image_embedding_th_compile),
# we show how to achieve a similar setup without Infinity.
# 2. The variable `allow_concurrent_inputs` passed to the `main` local_entrypoint is
# used to set both the number of concurrent inputs (via with_options) and the class variable
# `n_engines` (via modal.parameters). If you aren't using `with_options` you can use the
# [modal.concurrent](https://modal.com/docs/guide/concurrent-inputs#input-concurrency)
# decorator directly.
# 3. In `init_engines`, we are creating exactly one Infinity inference
# engine for each concurrently-passed batch of data. This is a high-level version of GPU packing suitable
# for use with a high-level inference engine like Infinity.
# 4. The [@modal.enter](https://modal.com/docs/reference/modal.enter#modalenter)
# decorator ensures that this method is called once per container, on startup (and `exit` is
# run once, on shutdown).

# If buffer_containers is set, use it, otherwise rely on `with_options`.
container_config = {"buffer_containers": buffer_containers} if buffer_containers else {}


@app.cls(
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    timeout=5 * 60,  # 5min timeout for large models + batches
    cpu=4,
    memory=5 * 1024,  # MB -> GB
    **container_config,
)
class InfinityEngine:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=100)
    n_engines: int = modal.parameter(default=1)
    threads_per_core: int = modal.parameter(default=4)
    verbose_inference: bool = modal.parameter(default=False)
    # For logging
    name: str = "InfinityEngine"

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
                    model_name_or_path=self.model_name,
                    batch_size=self.batch_size,
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
        """

        def readim(impath: os.PathLike):
            """Read with torch, convert back to PIL for Infinity"""
            return to_pil_image(read_image(str(vol_mnt / impath)))

        with ThreadPoolExecutor(
            max_workers=os.cpu_count() * self.threads_per_core
        ) as executor:
            images = list(executor.map(readim, im_path_list))

        return images

    @modal.method()
    async def embed(self, images: list[os.PathLike]) -> tuple[float, float]:
        """
        This is the workhorse function. We select a model from the queue, prepare
        a batch, execute inference, and return the time elapsed.

        NOTE: we throw away the embeddings here; you probably want to return
        them or save them directly to a modal.Volume.
        """
        # (0) Grab an engine from the queue
        engine = await self.engine_queue.get()

        try:
            # (1) Load batch of image data
            st = perf_counter()
            images = self.read_batch(images)
            batch_elapsed = perf_counter() - st

            # (2) Encode the batch
            st = perf_counter()
            # Infinity Engine is async
            embedding, _ = await engine.image_embed(images=images)
            embed_elapsed = perf_counter() - st
        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) Housekeeping
        if self.verbose_inference:
            print(f"Time to load batch: {batch_elapsed:.2f}s")
            print(f"Time to embed batch: {embed_elapsed:.2f}s")

        # (4) You may wish to return the embeddings themselves here
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
# This is the backbone of the example: it parses inputs, grabs a list of data, instantiates
# the InfinityEngine embedder application, and passes data to it via `map`.
#
# Inputs:
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up. Note that this cannot
# be used with `buffer_containers`: *if you want to use this, set* `buffer_containers=None` *above!*
# * `allow_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# argument for the inference app via the
# [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options) API.
# This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `threads_per_core` oversubscription factor for parallelized I/O (image reading).
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"),
# and it means the usual thing for machine learning inference: a group of images are processed
#  through the neural network together.
# * `model_name` a HuggingFace model path a la [openai/clip-vit-base-patch16]([OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT"));
# Infinity will automatically load it and prepare it for asynchronous serving.
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)
# * `hf_dataset_name` a HuggingFace data path a la "microsoft/cats_vs_dogs"
# * `log_file` (optional) points to a local path where a CSV of times will be logged
#
# These three parameters are used to pre-process images to the correct size in a big batch
# before inference. However, if you have the wrong numbers or aren't sure, Infinity will
# automatically handle resizing (at a cost to throughput).
# * `model_input_chan`: the number of color channels your model is expecting (probably 3)
# * `model_input_imheight`: the number of pixels tall your model is expecting the images to be
# * `model_input_imwidth`: the number of color channels your model is expecting (probably 3)
#
@app.local_entrypoint()
def main(
    # with_options parameters:
    gpu: str = "A10G",
    max_containers: int = 50,  # warning: this gets overridden if buffer_containers not None
    allow_concurrent_inputs: int = 2,
    # modal.parameters:
    threads_per_core: int = 8,
    batch_size: int = 100,
    model_name: str = "openai/clip-vit-base-patch16",
    model_input_chan: int = 3,
    model_input_imheight: int = 224,
    model_input_imwidth: int = 224,
    # data
    image_cap: int = -1,
    hf_dataset_name: str = "microsoft/cats_vs_dogs",
    million_image_test: bool = False,
    # logging (optional)
    log_file: str = None,  # TODO: remove local logging from example
):
    start_time = perf_counter()

    # (0) Catalog data: modify `catalog_jpegs` to fetch batches of your data paths.
    extracted_path = Path("extracted") / hf_dataset_name
    im_path_list, vol_setup_time = catalog_jpegs.remote(
        dataset_namespace=hf_dataset_name,
        cache_dir=extracted_path,
        image_cap=image_cap,
        model_input_shape=(model_input_chan, model_input_imheight, model_input_imwidth),
    )
    print(f"Took {vol_setup_time:.2f}s to setup volume.")
    if million_image_test:
        print("WARNING: `million_image_test` FLAG RECEIVED! RESETTING BSZ ETC!")
        mil = int(1e6)
        batch_size = 700
        while len(im_path_list) < mil:
            im_path_list += im_path_list
        im_path_list = im_path_list[:mil]
    n_ims = len(im_path_list)

    # (1) Init the model inference app
    # No inputs to with_options if none provided or buffer_used aboe
    make_empty = (buffer_containers is not None) or (max_containers is None)
    container_config = {} if make_empty else {"max_containers": max_containers}
    # Build the engine
    start_time = perf_counter()
    embedder = InfinityEngine.with_options(
        gpu=gpu, allow_concurrent_inputs=allow_concurrent_inputs, **container_config
    )(
        batch_size=batch_size,
        n_engines=allow_concurrent_inputs,
        model_name=model_name,
        threads_per_core=threads_per_core,
    )

    # (2) Embed batches via remote `map` call
    times, batchsizes = [], []
    for time, batchsize in embedder.embed.map(chunked(im_path_list, batch_size)):
        times.append(time)
        batchsizes.append(batchsize)

    # (3) Log
    if n_ims > 0:
        total_duration = perf_counter() - start_time
        total_throughput = n_ims / total_duration
        embed_throughputs = [
            batchsize / time for batchsize, time in zip(batchsizes, times)
        ]
        avg_throughput = sum(embed_throughputs) / len(embed_throughputs)

        log_msg = (
            f"{embedder.name}{gpu}::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={allow_concurrent_inputs}::"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tSingle-model throughput (avg):\t{avg_throughput:.2f} im/s\n"
        )

        print(log_msg)

        if log_file is not None:
            local_logfile = Path(log_file).expanduser()
            local_logfile.parent.mkdir(parents=True, exist_ok=True)

            import csv

            csv_exists = local_logfile.exists()
            with open(local_logfile, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    # write header
                    writer.writerow(
                        [
                            "batch_size",
                            "concurrency",
                            "max_containers",
                            "gpu",
                            "n_images",
                            "total_time",
                            "total_throughput",
                            "avg_model_throughput",
                        ]
                    )
                # write your row
                writer.writerow(
                    [
                        batch_size,
                        allow_concurrent_inputs,
                        max_containers,
                        gpu,
                        n_ims,
                        total_duration,
                        total_throughput,
                        avg_throughput,
                    ]
                )
