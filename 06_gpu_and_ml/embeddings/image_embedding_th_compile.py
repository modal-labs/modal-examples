# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/image_embedding_th_compile.py::main"]
# ---

# # Image Embedding Throughput Maximization with torch.compile
# In certain applications, the bottom line comes to *throughput*: process a batch of inputs as fast as possible.
# This example presents a Modal recipe for maximizing image embedding throughput using
# regular torch code, not worrying about complicated model gateway servers.
# This lets us control the bare-bones code and achieve the highest
# overall throughput for a multi-container app.
#
# ## BLUF (Bottom Line Up Front)
# Set concurrency (`max_concurrent_inputs`) to 2, and set `batch_size` as high as possible without
# hitting OOM errors (model-dependent).
# To get maximum throughput at any cost, set buffer_containers to 10. 
# Be sure to preprocess your data in the same manner that the model is expecting (e.g., resizing images; 
# doing this on-the-fly will greatly reduce throughput).
# If you only want to use one container, increase `batch_size` until you are maxing
# out the GPU (but keep concurrency, `max_concurrent_inputs`, capped around 2).

# ### Why?
# The two killers of throughput in this context are: cold-start time and the time to
# form a batch (i.e. reading the images from disk). While batch size maximizes GPU utilization,
# To avoid idle GPU cores during batch formation, we set use idle GPU cores to store additional
# copies of the model: this high-level form of _GPU packing_ is achieved via an async queue and the
# [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# decorator, called functionally through [modal.cls.with_concurrency](https://modal.com/docs/reference/modal.Cls#with_concurrency).

# Once you nail down an effective `batch_size` for your problem, you can crank up the number of containers
# to fan-out the computational load. Set buffer_containers > 0 so that Modal continuously spins up more
# and more containers until the task is complete; otherwise set it to None, and use max_containers to cap
# the number of containers allowed.

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
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, time_ns
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

# buffer_containers: int = 1000  # (50,)

# ## Dataset, Model, and Image Setup
# This example uses HuggingFace to download data and models. We will use a high-performance
# [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# to (1) cache model weights, (2) store the
# [image dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# that we want to embed, and (3) cache torch.compile kernels and artifacts.


# ### Volume Initialization
# You may need to [set up a secret](https://modal.com/secrets/) to access HuggingFace datasets
hf_secret = modal.Secret.from_name("huggingface-secret")


# Create a persisted dict - the data gets retained between app runs
racetrack_dict = modal.Dict.from_name("laion2B", create_if_missing=True)

# This name is important for referencing the volume in other apps or for
# [browsing](https://modal.com/storage):
data_volume = modal.Volume.from_name("example-embedding-data", create_if_missing=True)

# The location within the volume where torch.compile's caching backends should point to:
# This is the location within the container where this Volume will be mounted:
vol_mnt = Path("/data")
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
    threads_per_core: int = 8,  # threads per CPU for I/O oversubscription
    n_million_image_test: float = None,
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
        with ThreadPoolExecutor(
            max_workers=os.cpu_count * threads_per_core
        ) as executor:
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

    print(f"Took {perf_counter() - ds_preptime_st:.2f}s to setup volume.")
    if n_million_image_test > 0:
        print(f"WARNING: `{n_million_image_test} million_image_test` FLAG RECEIVED!")
        mil = int(n_million_image_test * 1e6)
        while len(im_path_list) < mil:
            im_path_list += im_path_list
        im_path_list = im_path_list[:mil]

    return im_path_list


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
# 1. We let Modal handle management of concurrent inputs via the `input_concurrency`
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
@app.cls(
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
    cpu=2.5,
    memory=2.5 * 1024,  # MB -> GB
    buffer_containers=50,
)
class TorchCompileEngine:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=100)
    n_engines: int = modal.parameter(default=1)
    model_input_chan: int = modal.parameter(default=3)
    model_input_imheight: int = modal.parameter(default=224)
    model_input_imwidth: int = modal.parameter(default=224)
    threads_per_core: int = modal.parameter(default=8)
    exp_tag: str = modal.parameter(default="default-tag")
    # Cannot currently gracefully set ENV vars from local_entrypoint
    cache_dir: Path = TH_CACHE_DIR
    # For logging
    name: str = "TorchCompileEngine"

    def init_th(self):
        """
        Have to manually turn this on for torch.compile.
        """
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        torch.set_grad_enabled(False)
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
        key = f"{self.exp_tag}-first.ctr.start"
        racetrack_dict[key] = racetrack_dict.get(key, time_ns())

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

        # (1) Load raw model weights and preprocessor once per container
        base = CLIPVisionModel.from_pretrained(self.model_name)
        self.preprocessor = CLIPImageProcessorFast.from_pretrained(
            self.model_name, usefast=True
        )

        # Only save what we need
        config = base.config
        state = base.state_dict()
        del base

        # (2) Check for trace artifacts cache
        if compile_cache.is_file():
            cache = compile_cache.read_bytes()
            with safe_globals([CacheInfo]):
                torch.compiler.load_cache_artifacts(cache)

        # (3) Build an Async Queue of compiled models
        self.engine_queue = asyncio.Queue()

        for idx in range(self.n_engines):
            # (3.a) Build a CLIPVisionModel model from weights
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

            await self.engine_queue.put(compiled_model)

            # (4) initialize threadpool for dataloading
            self.executor = ThreadPoolExecutor(
                max_workers=os.cpu_count() * self.threads_per_core,
                thread_name_prefix="img-io",
            )

    @staticmethod
    def readim(impath: os.PathLike):
        """
        Prepends this container's volume mount location to the image path.
        """
        return read_image(str(vol_mnt / impath))

    @modal.method()
    async def embed(
        self, images: list[os.PathLike], *args, **kwargs
    ) -> tuple[float, float]:
        """
        This is the workhorse function. We select a model from the queue, prepare
        a batch, execute inference, and return the time elapsed.

        NOTE: we throw away the embeddings here; you probably want to return
        them or save them directly to a modal.Volume.

        TODO: do image loading first before awaiting queue
        """

        try:
            # (0) Load batch of image data
            st = perf_counter()
            images = self.preprocessor(
                images=torch.stack(list(self.executor.map(self.readim, images))),
                device="cuda:0",
                return_tensors="pt",
            )
            batch_elapsed = perf_counter() - st

            # (1) Grab an engine from the queue
            engine = await self.engine_queue.get()

            # (2) Encode the batch
            st = perf_counter()
            embedding = engine(**images).pooler_output
            embed_elapsed = perf_counter() - st

        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) You may wish to return the embeddings themselves here
        return batch_elapsed, embed_elapsed, len(images)

    @modal.exit()
    async def exit(self) -> None:
        """
        trying to get less printouts?...
        """
        self.executor.shutdown(wait=True)
        racetrack_dict[f"{self.exp_tag}-last.ctr.complete"] = time_ns()
        return


# This modal.function is a helper that you probably don't need to call:
# it deletes the torch.compile cache dir we use for sharing a cache across
# containers (for measuring startup times).


@app.function(
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
)
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
# * `input_concurrency` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
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


@app.local_entrypoint()
def main(
    # APP CONFIG
    gpu: str = "any",
    max_containers: int = None,  # this gets overridden if buffer_containers is not None
    input_concurrency: int = 2,
    # MODEL CONFIG
    n_models: int = None,  # defaults to match `allow_concurrent_parameters`
    model_name: str = "openai/clip-vit-base-patch16",
    batch_size: int = 32,
    # DATA CONFIG
    im_chan: int = 3,
    im_height: int = 224,
    im_width: int = 224,
    hf_dataset_name: str = "microsoft/cats_vs_dogs",
    image_cap: int = -1,
    n_million_image_test: float = 0,
    # torch.compile cache
    destroy_cache: bool = False,
    exp_tag: str = "default-tag",
    log_file: str = "/home/ec2-user/modal-examples/06_gpu_and_ml/embeddings/_triton.csv",
):
    start_time = perf_counter()
    racetrack_dict[f"{exp_tag}-exp.start"] = time_ns()

    # (0.a) Catalog data: modify `catalog_jpegs` to fetch batches of your data paths.
    extracted_path = Path("extracted") / hf_dataset_name
    im_path_list = catalog_jpegs.remote(
        dataset_namespace=hf_dataset_name,
        cache_dir=extracted_path,
        image_cap=image_cap,
        model_input_shape=(im_chan, im_height, im_width),
        n_million_image_test=n_million_image_test,
    )
    print(f"Embedding {len(im_path_list)} images at batchsize {batch_size}.")

    # (0.b) This destroys cache for timing purposes - you probably don't want to do this!
    if destroy_cache:
        destroy_th_compile_cache.remote()

    # (1) Init the model inference app

    # Build the engine
    racetrack_dict[f"{exp_tag}-embedder.init"] = time_ns()

    container_config = {"max_containers": max_containers} if max_containers else {}
    embedder = TorchCompileEngine.with_concurrency(
        max_inputs=input_concurrency
    ).with_options(gpu=gpu, **container_config)(
        batch_size=batch_size,
        n_engines=n_models if n_models else input_concurrency,
        model_name=model_name,
        model_input_chan=im_chan,
        model_input_imheight=im_height,
        model_input_imwidth=im_width,
        exp_tag=exp_tag,
    )
    n_ims = len(im_path_list)
    # (2) Embed batches via remote `map` call
    # (2) Embed batches via remote `map` call
    preptimes, inftimes, batchsizes = [], [], []
    # embedder.embed.spawn_map(chunked(im_path_list, batch_size))
    for preptime, inftime, batchsize in embedder.embed.map(
        chunked(im_path_list, batch_size)
    ):
        preptimes.append(preptime)
        inftimes.append(inftime)
        batchsizes.append(batchsize)

    # (3) Log & persist results
    if n_ims > 0:
        total_duration = perf_counter() - start_time  # end-to-end wall-clock
        overall_throughput = n_ims / total_duration  # imgs / s, wall-clock

        # per-container metrics
        inf_throughputs = [bs / t if t else 0 for bs, t in zip(batchsizes, inftimes)]
        prep_throughputs = [bs / t if t else 0 for bs, t in zip(batchsizes, preptimes)]

        avg_inf_throughput = sum(inf_throughputs) / len(inf_throughputs)
        best_inf_throughput = max(inf_throughputs)

        avg_prep_throughput = sum(prep_throughputs) / len(prep_throughputs)
        best_prep_throughput = max(prep_throughputs)

        total_prep_time = sum(preptimes)
        total_inf_time = sum(inftimes)

        log_msg = (
            f"{embedder.name}{gpu}::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={input_concurrency}\n"
            f"\tTotal wall time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{overall_throughput:.2f} im/s\n"
            f"\tPrep time (sum):\t{total_prep_time:.2f} s\n"
            f"\tInference time (sum):\t{total_inf_time:.2f} s\n"
            f"\tPrep throughput  (avg/best):\t{avg_prep_throughput:.2f} / "
            f"{best_prep_throughput:.2f} im/s\n"
            f"\tInfer throughput (avg/best):\t{avg_inf_throughput:.2f} / "
            f"{best_inf_throughput:.2f} im/s\n"
        )
        print(log_msg)

        # ── optional CSV ───────────────────────────────────────────────────────────
        if log_file:
            path = Path(log_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)

            header = [
                "batch_size",
                "concurrency",
                "n_models",
                "max_containers",
                "gpu",
                "n_images",
                "total_wall_time",
                "overall_throughput",
                "total_prep_time",
                "total_inf_time",
                "avg_prep_thpt",
                "best_prep_thpt",
                "avg_inf_thpt",
                "best_inf_thpt",
            ]
            row = [
                batch_size,
                input_concurrency,
                n_models,
                max_containers,
                gpu,
                n_ims,
                total_duration,
                overall_throughput,
                total_prep_time,
                total_inf_time,
                avg_prep_throughput,
                best_prep_throughput,
                avg_inf_throughput,
                best_inf_throughput,
            ]

            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
