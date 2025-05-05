# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/compiled_embedder.py::main"]
# ---

# # Sharing the Love: Using `torch.compile` Artifacts Across Containers
# TODO: Brief into to torch.compile: fast startups
# ## BLUF (bottom line up front)
# TODO: Do A, B, C.

# ## Setup
# # Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Iterator

import modal

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/model-compile-cache"
# enable logging of hits/misses
os.environ["TORCH_LOGS"] = "+torch._inductor.codecache"
# ## Key Parameters
# TODO: remove these/ simplify as much as possible
max_concurrent_inputs: int = 2
gpu: str = "H100"
# max_containers: int = 10
memory_request: float = 5 * 1024  # MB->GB
core_request: float = 4
threads_per_core: int = 8
batch_size: int = 400
image_cap: int = 1000000

# This timeout caps the maximum time a single function call is allowed to take. In this example, that
# includes reading a batch-worth of data and running inference on it. When `batch_size` is large (e.g. 5000)
# and with a large value of `max_concurrent_inputs`, where a batch may sit in a queue for a while,
# this could take several minutes.
timeout_seconds: int = 5 * 60

# ## Data and Model Specification
# This model parameter should point to a model on HuggingFace that is supported by Infinity.
# Note that your selected model might require specialized imports when
# designing the image in the next section. This [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT")
# takes about 4-10s to load into memory.
model_name = "openai/clip-vit-base-patch16"  # 599 MB
model_input_shape = (224, 224)
model_compile_cache = "model-compile-cache"
DESTROY_CACHE = True

# We will use a high-performance [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# both to cache model weights and to store images we want to encode. The details of
# setting this volume up are below. Here, we just need to name it so that we can instantiate
# the Modal application.
# You may need to [set up a secret](https://modal.com/secrets/) to access HuggingFace datasets
hf_secret = modal.Secret.from_name("huggingface-secret")
# Change this global variable to use a different HF dataset:
hf_dataset_name = "extracted/microsoft/cats_vs_dogs"
# This name is important for referencing the volume in other apps or for [browsing](https://modal.com/storage):
vol_name = "example-embedding-data"
# This is the location within the container that this Volume will be mounted:
vol_mnt = Path("/data")
# Finally, the Volume object can be created:
data_volume = modal.Volume.from_name(vol_name, create_if_missing=True)
thcompile = vol_mnt / model_compile_cache  # "th_compile"  # for backend
model_cache_dir = vol_mnt / model_compile_cache  # for manually saved megacache

# in image creations:::
# RUN python -c " \
#     from torch.compiler import save_cache_artifacts; \
#     from transformers import CLIPVisionModel; \
#     m = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16').cuda().eval(); \
#     torch.compile(m); \
#     torch.save(torch.compiler.save_cache_artifacts(), '/cache/clip-cache.pt')"


def find_images_to_encode(image_cap: int = 1) -> list[os.PathLike]:
    """
    You can modify this function to find an return a list of your image paths.
    """

    im_path_list = [
        x.path
        for x in data_volume.listdir(hf_dataset_name, recursive=True)
        if x.path.endswith(".jpg")
    ]
    print(f"Found {len(im_path_list)} JPEGs, ", end="")

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if image_cap > 0:
        im_path_list = im_path_list[: min(len(im_path_list), image_cap)]
    return im_path_list


# ## Define the image
infinity_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pillow",  # for Infinity input typehint
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "huggingface_hub",
            "tqdm",  # progress bar for dataset download
            "torch",
            "transformers",
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            "HF_HOME": vol_mnt.as_posix(),  # For model and data caching in our Volume
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # For fast data transfer
            "TORCHINDUCTOR_CACHE_DIR": thcompile.as_posix(),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            "TORCH_LOGS": "+torch._inductor.codecache",
            # "TORCHDYNAMO_VERBOSE": "1",
            # "TORCH_LOGS": "+dynamo",
        }
    )
)

# Initialize the app
app = modal.App(
    "example-multi-compile",
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    secrets=[hf_secret],
)

# Imports inside the container
with infinity_image.imports():
    import torch
    from PIL.Image import Image
    from torch.serialization import safe_globals
    from torchvision.io import read_image
    from transformers import CLIPImageProcessorFast, CLIPVisionModel


# ## Inference app
@app.cls(
    image=infinity_image,
    volumes={vol_mnt: data_volume},
    timeout=timeout_seconds,
    buffer_containers=10,
    # max_containers=max_containers,
    gpu=gpu,
    cpu=core_request,
    memory=5 * 1024,  # MB -> GB
)
@modal.concurrent(max_inputs=max_concurrent_inputs)
class TorchCompileEngine:
    n_engines: int = max_concurrent_inputs
    model_name: str = model_name
    cache_dir: Path = model_cache_dir

    def init_th(self):
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major > 8:
            torch.set_float32_matmul_precision("high")

    @modal.enter()
    async def init_engines(self):
        # (0) Set backend torch config
        self.init_th()
        # This makes sure it gets the cache (in case DELETE is on)
        data_volume.reload()

        from torch.compiler._cache import CacheInfo

        # TODO: what this do _exactly_...
        torch.compiler.set_stance("eager_on_recompile")

        # (1) Load raw model once
        msg = "new container!"

        # Download model weights once
        st = perf_counter()
        base = CLIPVisionModel.from_pretrained(self.model_name)
        self.preprocessor = CLIPImageProcessorFast.from_pretrained(
            self.model_name, usefast=True
        )
        msg += f"\n\ttime to call from_pretrained: {perf_counter() - st:.2E}"
        config = base.config
        state = base.state_dict()

        del base  # free the base model

        # (2) Check for trace artifacts
        cache_filename = self.model_name.replace("/", "_") + "_compiled_model_cache"
        compile_cache = (self.cache_dir / cache_filename).with_suffix(".pt")
        if compile_cache.is_file():
            msg += "\n\tcache exists;"
            st = perf_counter()
            cache = compile_cache.read_bytes()
            with safe_globals([CacheInfo]):
                torch.compiler.load_cache_artifacts(cache)
            msg += f"{perf_counter() - st:.2E} time to th.load cache;"
        else:
            msg += "\n\tth.compile cache not found"

        # (3) Build an Async Queue of compiled models
        self.engine_queue = asyncio.Queue()

        for idx in range(self.n_engines):
            # (3.a) Build a model get weights
            st = perf_counter()
            model = CLIPVisionModel(config).eval().cuda()
            model.load_state_dict(state)

            # Uses cache under the hood (if available)
            compiled_model = torch.compile(model)

            # (3.b) Cache the trace only in the 1st container for the 1st model copy
            if (idx == 0) and (not compile_cache.is_file()):
                # Complete the trace with an inference
                compiled_model(
                    **self.preprocessor(
                        images=torch.randn(batch_size, 3, 224, 224),
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
            elapsed = perf_counter() - st
            await self.engine_queue.put(compiled_model)
            elapsed = perf_counter() - st
            msg += f"\n\tmodel{idx} load+compile{tmp} time {elapsed:.2E}"

        # log some times to volume
        if msg:
            print(msg)
            compile_cache.parent.mkdir(exist_ok=True, parents=True)
            with open(compile_cache.parent / "compile_times.txt", "a") as f:
                f.write(msg)

    def read_batch(self, im_path_list: list[os.PathLike], device) -> list["Image"]:
        def readim(impath: os.PathLike):
            """Read with torch, convert back to PIL for Infinity"""
            return read_image(str(vol_mnt / impath))

        with ThreadPoolExecutor(
            max_workers=os.cpu_count() * threads_per_core
        ) as executor:
            images = list(executor.map(readim, im_path_list))

        return self.preprocessor(
            images=torch.stack(images), device=device, return_tensors="pt"
        )

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
            st = perf_counter()
            images = self.read_batch(images, engine.device)
            batch_elapsed = perf_counter() - st
            # print(f"batch_shape: {images.shape}")
            # (2) Encode the batch
            st = perf_counter()
            embedding = engine(**images).pooler_output
            embed_elapsed = perf_counter() - st
            # print(f"made embedding with shape: {embedding.shape}")
        finally:
            # No matter what happens, return the engine to the queue
            await self.engine_queue.put(engine)

        # (3) Housekeeping
        # print(f"Time to load batch: {batch_elapsed:.2E}s")
        # print(f"Time to embed batch: {embed_elapsed:.2E}s")

        # (4) You may wish to return the embeddings themselves here
        return embed_elapsed, len(images)

    @modal.exit()
    async def exit(self) -> None:
        """
        trying to get less printouts?...
        """

        # how kill async quietly..

        # if DESTROY_CACHE and self.cache_dir.exists():
        #     num_files = sum(1 for f in self.cache_dir.rglob("*") if f.is_file())
        #     print(f"*** DESTROYING model cache! ({num_files} files)")
        #     shutil.rmtree(self.cache_dir.as_posix())
        #     n_deleted = num_files - sum(
        #         1 for f in self.cache_dir.rglob("*") if f.is_file()
        #     )
        #     print(f"deleted {n_deleted}/{num_files}.")
        return


@app.function(image=infinity_image, volumes={vol_mnt: data_volume})
def destroy_cache():
    if model_cache_dir.exists():
        num_files = sum(1 for f in model_cache_dir.rglob("*") if f.is_file())

        print(f"\t*** DESTROYING model cache! ({num_files} files)")
        shutil.rmtree(model_cache_dir.as_posix())
    else:
        print(
            f"\t***destroy_cache was called, but path doesnt exist:\n\t{model_cache_dir}"
        )
    return


# ## Local Entrypoint
# This backbone code is run on your machine. It starts up the app,
# catalogs the data, and via the remote `map` call, parses the data
# with the Infinity embedding engine. The embedder.embed executions
# across the batches are autoscaled depending on the app parameters
# `max_containers` and `max_concurrent_inputs`.
@app.local_entrypoint()
def main():
    start_time = perf_counter()

    if DESTROY_CACHE:
        destroy_cache.remote()

    # (1) Catalog data: modify `catalog_jpegs` to fetch batches of your data.
    im_path_list = find_images_to_encode(image_cap=image_cap)
    im_path_list += im_path_list  # 40k
    im_path_list += im_path_list  # 80k
    im_path_list += im_path_list  # 160k
    im_path_list += im_path_list  # 320k
    im_path_list += im_path_list  # 740k
    im_path_list += im_path_list  # 1480k

    def chunked(
        seq: list[os.PathLike], subseq_size: int
    ) -> Iterator[list[os.PathLike]]:
        """
        Helper function that chunks a sequence into subsequences of length `subseq_size`.
        """
        for i in range(0, len(seq), subseq_size):
            yield seq[i : i + subseq_size]

    if image_cap > 0:
        im_path_list = im_path_list[:image_cap]
    print(f"using {len(im_path_list)}.")
    # print(f"Took {vol_setup_time:.2f}s to setup volume.")
    n_ims = len(im_path_list)

    # (2) Init the model inference app
    start_time = perf_counter()
    embedder = TorchCompileEngine()
    # Call an initial time to trigger standalone model compilation
    # print("Encoding a test batch before `map` is called...")
    # embedder.embed.remote(im_path_list[:batch_size])
    # print("...finna call map.")

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
            "****************************\n"
            "****************************\n"
            "****************************\n"
            "****************************\n"
            f"EmbeddingRacetrack{gpu}::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={max_concurrent_inputs}::"
            f"max_containers={'inf'}::cores={core_request}\n"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{avg_throughput:.2f} im/s\n"
            "****************************\n"
            "****************************\n"
            "****************************\n"
            "****************************\n"
        )

        print(log_msg)
