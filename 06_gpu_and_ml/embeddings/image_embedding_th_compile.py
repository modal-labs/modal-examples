# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/image_embedding_th_compile.py::main"]
# ---

# TODO: deprecation warnings at the beginning??

# # A Recipe for Throughput Maximization: GPU Packing with torch.compile
# In certain applications, the bottom line comes to *throughput*: process a batch of inputs as fast as possible.
# This example presents a Modal recipe for maximizing image embedding throughput,
# taking advantage of Modal's concurrency features.
# ### BLUF (Bottom Line Up Front)
# recipe ABC
# ### Why?
# compile discussion

# ## Local env imports
# # Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
import shutil
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
# to (1) cache model weights, (2) store the
# [image dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# that we want to embed, and (3) cache torch.compile kernels and artifacts.


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

# The location within the volume where torch.compile's caching backends should point to:
TH_CACHE_DIR = vol_mnt / "model-compile-cache"

# ### Define the image
th_compile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "tqdm",  # progress bar for dataset download
            "torch",  # torch.compile
            "transformers",  # CLIPVisionModel etc.
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            # For fast HuggingFace model and data caching and download in our Volume
            "HF_HOME": vol_mnt.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Enables speedy caching across containers
            "TORCHINDUCTOR_CACHE_DIR": TH_CACHE_DIR.as_posix(),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        }
    )
)

# Initialize the app
app = modal.App(
    "example-compiled-embedder",
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
    secrets=[hf_secret],
)

# Imports inside the container
with th_compile_image.imports():
    import torch
    from torch.serialization import safe_globals
    from torchvision.io import read_image
    from transformers import CLIPImageProcessorFast, CLIPVisionModel


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


@app.function(
    image=th_compile_image,
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
# that wraps an AsyncQueue of `torch.compile`'d models.
# Some important notes:
# 1. We let Modal handle management of concurrent inputs via the `allow_concurrent_inputs`
# parameter, which we pass to the class constructor in our `main` local_entrypoint below. This
# parameter sets both the number of concurrent inputs (via with_options) and the class variable
# `n_engines` (via modal.parameters). If you aren't using `with_options` you can use the
# [modal.concurrent](https://modal.com/docs/guide/concurrent-inputs#input-concurrency)
# decorator directly.
# 2. In `init_engines`, we are compiling one copy of the model for each concurrently-passed
# batch of data. Higher level inference engines like
# [Infinity](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity")
# handle this under the hood, at a cost to throughput.
# 3. The [@modal.enter](https://modal.com/docs/reference/modal.enter#modalenter)
# decorator ensures that this method is called once per container, on startup (and `exit` is
# run once, on shutdown).

# If buffer_containers is set, use it, otherwise rely on `with_options`.
container_config = {"buffer_containers": buffer_containers} if buffer_containers else {}


@app.cls(
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
    timeout=5 * 60,  # 5min timeout for large models + batches
    cpu=4,
    memory=5 * 1024,  # MB -> GB
    **container_config,
)
class TorchCompileEngine:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=100)
    n_engines: int = modal.parameter(default=1)
    model_input_chan: int = modal.parameter(default=3)
    model_input_imheight: int = modal.parameter(default=224)
    model_input_imwidth: int = modal.parameter(default=224)
    threads_per_core: int = modal.parameter(default=4)
    verbose_inference: bool = modal.parameter(default=False)
    # Cannot currently gracefully set ENV vars from local_entrypoint
    cache_dir: Path = TH_CACHE_DIR
    # For logging
    name: str = "TorchCompileEngine"

    def init_th(self):
        """
        Have to manually turn this on for torch.compile.
        """
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major > 8:
            torch.set_float32_matmul_precision("high")

    @modal.enter()
    async def init_engines(self):
        """
        Once per container start, `self.n_engines` models will be initialized
        (one for each concurrently served input via Modal). The first container
        needs to compute a trace and cache the kernels to our modal.Volume; sub-
        sequent containers can use that cache (which takes 50%-60% the time
        as the first torch.compile call).
        """
        # (0) Setup
        # Torch backend
        self.init_th()
        # This makes sure n-th container finds the cache created by the first one
        data_volume.reload()
        # This is where we will cache torch.compile artifacts
        compile_cache: Path = Path(self.cache_dir) / (
            self.model_name.replace("/", "_") + "_compiled_model_cache.pt"
        )
        # Condense modal.parameter values
        model_input_shape = (
            self.batch_size,
            self.model_input_chan,
            self.model_input_imwidth,
            self.model_input_imwidth,
        )

        from torch.compiler._cache import CacheInfo

        # This tells torch to dynamically decide whether to recompile from scratch
        # or to check for a cache (we want it to check for a cache!)
        torch.compiler.set_stance("eager_on_recompile")

        # We will build up a message but just print once at the end.
        msg = "new container!"

        # (1) Load raw model weights and preprocessor once per container
        st = perf_counter()
        base = CLIPVisionModel.from_pretrained(self.model_name)
        self.preprocessor = CLIPImageProcessorFast.from_pretrained(
            self.model_name, usefast=True
        )
        msg += f"\n\ttime to call from_pretrained: {perf_counter() - st:.2E}"

        # Only save what we need
        config = base.config
        state = base.state_dict()
        del base

        # (2) Check for trace artifacts cache
        if compile_cache.is_file():
            st = perf_counter()
            cache = compile_cache.read_bytes()
            with safe_globals([CacheInfo]):
                torch.compiler.load_cache_artifacts(cache)
            msg += f"\n\tth.compile cache exists; time to load it: {perf_counter() - st:.2E}"
        else:
            msg += "\n\tth.compile cache not found"

        # (3) Build an Async Queue of compiled models
        self.engine_queue = asyncio.Queue()

        for idx in range(self.n_engines):
            # (3.a) Build a CLIPVisionModel model from weights
            st = perf_counter()
            model = CLIPVisionModel(config).eval().cuda()
            model.load_state_dict(state)

            # Uses cache under the hood (if available)
            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=True,
            )

            # (3.b) Cache the trace only in the 1st container for the 1st model copy
            if (idx == 0) and (not compile_cache.is_file()):
                # Complete the trace with an inference
                compiled_model(
                    **self.preprocessor(
                        images=torch.randn(model_input_shape),
                        device=compiled_model.device,
                        return_tensors="pt",
                    )
                )
                # Extract and save artifacts
                compile_cache.parent.mkdir(exist_ok=True, parents=True)
                artifact_bytes, cache_info = torch.compiler.save_cache_artifacts()
                compile_cache.write_bytes(artifact_bytes)
                tmp = " (incl. trace, save cache)"
            else:
                tmp = ""
            await self.engine_queue.put(compiled_model)
            elapsed = perf_counter() - st
            msg += f"\n\tmodel{idx} | load+compile{tmp} time: {elapsed:.2E}"

        # Log to std out; you could instead write to a modal.Volume
        if msg:
            print(msg)

    def read_batch(self, im_path_list: list[os.PathLike], device) -> list["Image"]:
        """
        Read a batch of data. We use Threads to parallelize this I/O-bound task,
        and finally toss the batch into the CLIPImageProcessorFast preprocessor.
        """

        def readim(impath: os.PathLike):
            """
            Prepends this container's volume mount location to the image path.
            """
            return read_image(str(vol_mnt / impath))

        with ThreadPoolExecutor(
            max_workers=os.cpu_count() * self.threads_per_core
        ) as executor:
            images = list(executor.map(readim, im_path_list))

        return self.preprocessor(
            images=torch.stack(images), device=device, return_tensors="pt"
        )

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
            images = self.read_batch(images, engine.device)
            batch_elapsed = perf_counter() - st

            # (2) Encode the batch
            st = perf_counter()
            embedding = engine(**images).pooler_output
            embed_elapsed = perf_counter() - st

        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) Housekeeping
        if self.verbose_inference:
            print(f"Time to load batch: {batch_elapsed:.2E}s")
            print(f"Time to embed batch: {embed_elapsed:.2E}s")

        # (4) You may wish to return the embeddings themselves here
        return embed_elapsed, len(images)

    @modal.exit()
    async def exit(self) -> None:
        """
        trying to get less printouts?...
        """
        # TODO: how kill async quietly..
        return


# This modal.function is a helper that you probably don't need to call:
# it deletes the torch.compile cache dir we use for sharing a cache across
# containers (for measuring startup times).


@app.function(image=th_compile_image, volumes={vol_mnt: data_volume})
def destroy_th_compile_cache():
    """
    For timing purposes: deletes torch compile cache dir.
    """
    if TH_CACHE_DIR.exists():
        num_files = sum(1 for f in TH_CACHE_DIR.rglob("*") if f.is_file())

        print(
            "\t*** DESTROYING model cache! You sure you wanna do that?! "
            f"({num_files} files)"
        )
        shutil.rmtree(TH_CACHE_DIR.as_posix())
    else:
        print(
            f"\t***destroy_cache was called, but path doesnt exist:\n\t{TH_CACHE_DIR}"
        )
    return


# ## Local Entrypoint
# This is the backbone of the example: it parses inputs, grabs a list of data, instantiates
# the TorchCompileEngine embedder application, and passes data to it via `map`.
#
# Inputs:
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up. Note that this cannot
# be used with `buffer_containers`: *if you want to use this, set* `buffer_containers=None` *above!*
# * `allow_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# argument for the inference app via the
# [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options) API.
# * `threads_per_core` oversubscription factor for parallelized I/O (image reading).
# * `batch_size` determines how many images are passed to individual instances of the model at a time.
# * `model_name` a HuggingFace model path a la [openai/clip-vit-base-patch16]([OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT"));
# It needs to be wrappable by HuggingFace's CLIPVisionModel class.
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)
# * `hf_dataset_name` a HuggingFace data path a la "microsoft/cats_vs_dogs"
# * `log_file` (optional) points to a local path where a CSV of times will be logged
# * `destroy_cache` (optional) destroys the torch.compile cache e.g. for timing/debugging
#
# These three parameters are used to pre-process images to the correct size in a big batch
# before inference and for torch.compile to optimize the trace for your predicted batch size.
# * `model_input_chan`: the number of color channels your model is expecting (probably 3)
# * `model_input_imheight`: the number of pixels tall your model is expecting the images to be
# * `model_input_imwidth`: the number of color channels your model is expecting (probably 3)
#
@app.local_entrypoint()
def main(
    # with_options parameters:
    gpu: str = "A10G",
    max_containers: int = 50,  # this gets overridden if buffer_containers is not None
    allow_concurrent_inputs: int = 2,
    # modal.parameters:
    threads_per_core: int = 8,
    batch_size: int = 32,
    model_name: str = "openai/clip-vit-base-patch16",  # 599 MB
    model_input_chan: int = 3,
    model_input_imheight: int = 224,
    model_input_imwidth: int = 224,
    # data
    image_cap: int = -1,
    hf_dataset_name: str = "microsoft/cats_vs_dogs",
    # torch.compile cache
    destroy_cache: bool = False,
    # logging (optional)
    log_file: str = None,  # TODO: remove local logging from example
):
    start_time = perf_counter()

    # (0.a) Catalog data: modify `catalog_jpegs` to fetch batches of your data paths.
    extracted_path = Path("extracted") / hf_dataset_name
    im_path_list, vol_setup_time = catalog_jpegs.remote(
        dataset_namespace=hf_dataset_name,
        cache_dir=extracted_path,
        image_cap=image_cap,
        model_input_shape=(model_input_chan, model_input_imheight, model_input_imwidth),
    )
    print(f"Took {vol_setup_time:.2f}s to setup volume.")
    n_ims = len(im_path_list)

    # (0.b) This destroys cache for timing purposes - you probably don't want to do this!
    if destroy_cache:
        destroy_th_compile_cache.remote()

    # (1.a) Init the model inference app
    # No inputs to with_options if none provided or buffer_used aboe
    make_empty = (buffer_containers is not None) or (max_containers is None)
    container_config = {} if make_empty else {"max_containers": max_containers}
    # Build the engine
    start_time = perf_counter()
    embedder = TorchCompileEngine.with_options(
        gpu=gpu, allow_concurrent_inputs=allow_concurrent_inputs, **container_config
    )(
        batch_size=batch_size,
        n_engines=allow_concurrent_inputs,
        model_name=model_name,
        model_input_chan=model_input_chan,
        model_input_imheight=model_input_imheight,
        model_input_imwidth=model_input_imwidth,
        threads_per_core=threads_per_core,
    )

    ############################
    # (1.b) Call one initial time to trigger standalone model compilation!
    if destroy_cache and (buffer_containers is None):
        # TODO: better way to avoid race condition for diff containers
        # competing to be "first done"?
        print("Encoding a test batch before `map` is called...")
        embedder.embed.remote(im_path_list[:batch_size])
        print("...now we will call map.")

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
