# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/image_embedding_th_compile.py::main"]
# ---

# # Image Embedding Throughput Maximization with the Triton Inference Server
# The [Triton Inference Server](https://github.com/triton-inference-server)
# is a powerful model serving gateway (or inference engine) that uses advanced,
# CUDA-level memory optimization that yields extremely high throughput. This
# demo shows how to serve an image embedding model with Triton, including a
# zero-copy inference subroutine.

# ## Local env imports
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterator, List

import modal

# ## Dataset, Model, and Image Setup
# This example uses HuggingFace to download data and models. We will use a high-performance
# [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume")
# both to cache model weights and to store the
# [image dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
# that we want to embed.

# ### Volume Initialization
# You may need to [set up a secret](https://modal.com/secrets/) to access HuggingFace datasets
hf_secret = modal.Secret.from_name("huggingface-secret")
data_volume = modal.Volume.from_name("example-embedding-data", create_if_missing=True)
VOL_MNT = Path("/data")
MODEL_REPO = VOL_MNT / "triton_repo"  # will hold model.plan + config

# Constants used to built Triton config on-the-fly
IN_NAME, IN_PATH = "clip_input", "/clip_input"
OUT_NAME, OUT_PATH = "clip_output", "/clip_output"
DTYPE = "FP16"


# image with Triton + torch + tritonclient (tiny helper)
triton_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tritonserver:24.03-py3", add_python="3.10"
    )
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --no-cache-dir torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "uv pip install --system --no-cache-dir transformers pillow tritonclient[all] "
        "tqdm hf_transfer tensorrt onnx "
    )
    .run_commands("uv pip install --system pynvml")
    .env(
        {
            "HF_HOME": VOL_MNT.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Tell Triton where the repo will be mounted
            "MODEL_REPO": MODEL_REPO.as_posix(),
        }
    )
    .entrypoint([])
)

app = modal.App(
    "example-triton-embedder",
    image=triton_image,
    volumes={VOL_MNT: data_volume},
    secrets=[hf_secret],
)

with triton_image.imports():
    import numpy as np
    import torch  # noqa: F401   – for torchscript
    import torchvision
    import tritonclient.grpc as grpcclient
    from torchvision.io import read_image
    from tqdm import tqdm
    from transformers import CLIPImageProcessorFast, CLIPVisionModel
    from tritonclient.utils import shared_memory as shm


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
    image=triton_image,
    volumes={VOL_MNT: data_volume},
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
        mounted_cache_dir = VOL_MNT / cache_dir
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
    if (VOL_MNT / cache_dir).is_dir():
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
# that manages a Triton Inference Server.
# Some important notes:
# 1. We let Modal handle management of concurrent inputs via the `max_concurrent_inputs`
# parameter, which we pass to the class constructor in our `main` local_entrypoint below. This
# parameter sets both the number of concurrent inputs (via with_options) and the class variable
# `n_engines` (via modal.parameters). If you aren't using `with_options` you can use the
# [modal.concurrent](https://modal.com/docs/guide/concurrent-inputs#input-concurrency)
# decorator directly.
# 2. In `_start_triton`, the first step is to organize the artifacts and configs Triton
# needs to start up a model. This is how we pass `n_engines` and specify a backend.
# Triton supports [several backends](https://github.com/triton-inference-server#:~:text=TensorRT%2C%20TensorFlow%2C%20PyTorch%2C%20Python%2C%20ONNX%20Runtime%2C%20and%20OpenVino.),
# but we have only sussed out the PyTorch backend for this example, so it can be compared
# with our [bare-bones torch.compile](https://modal.com/docs/examples/image_embedding_th_compile)
# peer example. If the server fails to set up properly and return a heartbeat, an error is raised.
# 3. `ensure_region`, `read_batch`, and `embed` are much more complicated than in the other
# image embedding examples: this is because Triton provides a (relatively) convenient interface
# for zero-copy data transfer from the client (i.e. this Modal app) to the server.


@app.cls(
    image=triton_image,
    volumes={VOL_MNT: data_volume},
    cpu=4,
    memory=2.5 * 1024,  # MB -> GB
)
class TritonServer:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=1)
    n_engines: int = modal.parameter(default=1)
    triton_backend: str = modal.parameter(default="pytorch")

    model_input_chan: int = modal.parameter(default=3)
    model_input_imheight: int = modal.parameter(default=224)
    model_input_imwidth: int = modal.parameter(default=224)
    output_dim: int = modal.parameter(default=768)

    force_rebuild: bool = modal.parameter(default=False)
    name: str = "Triton"

    def set_names(
        self,
    ):
        """
        Turn 'openai/clip-vit-base-patch16' → 'openai_clip-vit-base-patch16'
        (slashes, spaces and dots are not allowed in model dir names)
        """
        safe_name = (
            self.model_name.replace("/", "_").replace(" ", "_").replace(".", "_")
        )
        self.triton_model_name = f"{safe_name}_cc{self.n_engines}_bsz{self.batch_size}"
        self.in_shape = (
            self.batch_size,
            self.model_input_chan,
            self.model_input_imheight,
            self.model_input_imwidth,
        )

    @modal.enter()
    async def _start_triton(self):
        self.set_names()
        self.build_triton_repo()

        import subprocess
        import time

        import tritonclient.http as http

        self._client = http.InferenceServerClient(url="localhost:8000")
        # start triton in background
        self._proc = subprocess.Popen(
            [
                "tritonserver",
                f"--model-repository={MODEL_REPO}",
                "--exit-on-error=true",
                "--model-control-mode=none",  # autoload
                *self.gpu_pool_flags(),
            ]
        )

        # Load
        if "--model-control-mode=explicit" in self._proc.args:
            self._client.load_model(self.triton_model_name)

        # Heartbeat
        self._client = grpcclient.InferenceServerClient(url="localhost:8001")

        # Wait for Triton to start; crash if it fails.
        minutes_wait = 2
        check_rate_hz = 2
        n_iter = minutes_wait * 60 * check_rate_hz
        for idx in tqdm(
            range(n_iter), total=n_iter, desc="Waiting for server hearbeat"
        ):
            try:
                if self._client.is_model_ready(self.triton_model_name):
                    break
            except Exception:
                pass
            time.sleep(1 / check_rate_hz)
            if (idx / check_rate_hz) == int(idx / check_rate_hz):
                print(".", end="")
        else:
            raise RuntimeError("Triton failed to become ready")

        self.executor = ThreadPoolExecutor(
            max_workers=os.cpu_count() * 8,
            thread_name_prefix="img-io",
        )

    def gpu_pool_flags(self, headroom_pct: float = 0.10):
        """
        Return CLI flag strings that give Triton ~90 % of every visible GPU
        and ¼ of that amount for pinned host memory.
        """
        import pynvml

        pynvml.nvmlInit()
        flags = []
        total_pool = 0

        for idx in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            total = pynvml.nvmlDeviceGetMemoryInfo(h).total  # bytes
            pool = int(total * (1 - headroom_pct))
            flags.append(f"--cuda-memory-pool-byte-size={idx}:{pool}")
            total_pool = max(total_pool, pool)  # use biggest for pin
        flags.append(f"--pinned-memory-pool-byte-size={total_pool // 4}")
        return flags

    def build_triton_repo(
        self,
        version: str = "1",
        fp16: bool = True,
    ):
        """
        Build a Triton-ready repo for CLIP vision encoder.
        """
        from pathlib import Path

        import torch

        repo_dir = Path(MODEL_REPO) / self.triton_model_name / version
        repo_dir.mkdir(parents=True, exist_ok=True)

        # 0. short-circuit if artifacts & config already exist
        artifact = repo_dir / (
            "model.pt" if self.triton_backend == "pytorch" else "model.plan"
        )
        cfg_file = Path(MODEL_REPO) / self.triton_model_name / "config.pbtxt"
        if artifact.exists() and cfg_file.exists() and (not self.force_rebuild):
            print("Model repo already complete – skip build.")
            return

        print("Building torch model...", end="")
        st = perf_counter()

        # 1.  Build Torch module (used for all backends)
        class ClipEmbedder(torch.nn.Module):
            def __init__(self, hf_name: str, fp16: bool):
                super().__init__()
                self.clip = CLIPVisionModel.from_pretrained(hf_name)
                if fp16:
                    self.clip.half()
                self.clip.eval()

            @torch.no_grad()
            def forward(self, pixels: torch.Tensor):
                return self.clip(pixel_values=pixels).pooler_output

        model = ClipEmbedder(self.model_name, fp16).eval().cuda()
        example = torch.randn(
            self.in_shape,
            device="cuda",
            dtype=torch.float16 if fp16 else torch.float32,
        )
        print(f"took {perf_counter() - st:.2E}s")

        # 2.  Write backend-specific artifact
        if self.triton_backend == "pytorch":
            print("doing torch trace...", end="")
            st = perf_counter()
            traced = torch.jit.trace(model, example, strict=False).cpu()
            # rename io so we have input0 / output0
            graph = traced.inlined_graph
            g_inputs, g_outputs = list(graph.inputs()), list(graph.outputs())
            g_inputs[0].setDebugName("input0")
            g_outputs[0].setDebugName("output0")
            traced.save(artifact)
            # Free GPU memory
            del model, traced
            torch.cuda.empty_cache()
            print(f"took {perf_counter() - st:.2E}s")

        else:
            raise NotImplementedError(
                f"Triton backend `{self.triton_backend}` not"
                "implemented yet; try `pytorch`!"
            )

        # 3.  Generate config.pbtxt
        dtype = "TYPE_FP16" if fp16 else "TYPE_FP32"
        cfg_text = self.make_config(
            name=self.triton_model_name,
            dtype=dtype,
            output_dim=self.output_dim,
            instances=self.n_engines,
        )
        cfg_file.write_text(cfg_text)

        data_volume.commit()  # persist for future containers
        print(f"✓ wrote {artifact.name} + config for backend='{self.triton_backend}'")

    def make_config(
        self,
        name: str,
        dtype: str,
        output_dim: int,
        instances: int = 1,
    ) -> str:
        """Return a minimal, left-aligned Triton config.pbtxt."""
        from textwrap import dedent

        # Config basics: choose a backend
        cfg = f"""\
            name: "{name}"
            backend: "{self.triton_backend}"
            max_batch_size: {self.batch_size}
            """
        # Set inputs/outputs info
        cfg += f"""\
            input [
            {{
                name: "input0"
                data_type: {dtype}
                dims: [ {", ".join(map(str, self.in_shape[1:]))} ]
            }}
            ]

            output [
            {{
                name: "output0"
                data_type: {dtype}
                dims: [ {output_dim} ]
            }}
            ]
            """
        # Multi-model concurrency within a single (each) GPU
        cfg += f"""
            instance_group [
            {{ kind: KIND_GPU, count: {instances} }}
            ]
            
            """  # noqa: W293

        cfg += f"""
            optimization {{ execution_accelerators {{
            gpu_execution_accelerator : [ {{
                name : "{self.triton_backend}"
                parameters {{ key: "precision_mode" value: "{dtype}" }}
                parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
                }}]
            }}}}
            """
        return dedent(cfg)

    @staticmethod
    def readim(impath: os.PathLike):
        """
        Prepends this container's volume mount location to the image path.
        """
        return read_image(str(VOL_MNT / impath))

    def _ensure_region(self, name: str, path: os.PathLike, byte_size: int):
        """
        Create a system shared-memory block and remember its handle.
        """

        if not hasattr(self, "_shm"):
            self._shm = {}

        # first time
        if name not in self._shm:
            self._shm[name] = shm.create_shared_memory_region(name, path, byte_size)
            self._client.register_system_shared_memory(name, path, byte_size)
            return

    def _load_batch(self, img_paths: List[str]):
        """
        Given a list of image paths, load them into a shared memory block.
        """
        batch = (
            torch.stack(list(self.executor.map(self.readim, img_paths))).to(
                torch.float16
            )
            / 255
        ).numpy()

        # input SHM
        self._ensure_region(IN_NAME, IN_PATH, batch.nbytes)
        shm.set_shared_memory_region(self._shm[IN_NAME], [batch])

        # output SHM
        out_bytes = batch.shape[0] * self.output_dim * batch.dtype.itemsize
        self._ensure_region(OUT_NAME, OUT_PATH, out_bytes)

        return batch.shape, batch.nbytes, out_bytes

    @modal.method()
    async def embed(self, imgs: list[os.PathLike]) -> tuple[float, float, int]:
        """
        This is the workhorse function. We select a model from the queue, prepare
        a batch, execute inference, and return the time elapsed.

        NOTE: we throw away the embeddings here; you probably want to return
        them or save them directly to a modal.Volume.
        """
        # Load data from volume into shared memory block
        t0 = perf_counter()
        in_shape, in_bytes, out_bytes = self._load_batch(imgs)
        t_prep = perf_counter() - t0

        # Tell Triton where to get the data
        inp = grpcclient.InferInput("input0", in_shape, DTYPE)
        inp.set_shared_memory(IN_NAME, in_bytes)

        out = grpcclient.InferRequestedOutput("output0")
        out.set_shared_memory(OUT_NAME, out_bytes)

        # Inference
        t1 = perf_counter()
        self._client.infer(self.triton_model_name, [inp], outputs=[out])
        t_inf = perf_counter() - t1

        # # (If you need the vectors:)
        # vecs = shm.get_contents_as_numpy(
        #     self._shm[OUT_NAME], (in_shape[0], self.output_dim), DTYPE
        # )

        print(f"\tBatchCreate={t_prep * 1e3:.1f} ms\n\tInference={t_inf * 1e3:.1f} ms")
        return t_prep, t_inf, len(imgs)

    @modal.exit()
    def _cleanup(self):
        self._proc.terminate()
        self.executor.shutdown()


# This modal.function is a helper that you probably don't need to call:
# it deletes the torch.compile cache dir we use for sharing a cache across
# containers (for measuring startup times).


@app.function(image=triton_image, volumes={VOL_MNT: data_volume})
def destroy_triton_cache():
    """
    For timing purposes: deletes torch compile cache dir.
    """
    import shutil

    if MODEL_REPO.exists():
        num_files = sum(1 for f in MODEL_REPO.rglob("*") if f.is_file())

        print(
            "\t*** DESTROYING model cache! You sure you wanna do that?! "
            f"({num_files} files)"
        )
        shutil.rmtree(MODEL_REPO.as_posix())
    else:
        print(f"\t***destroy_cache was called, but path doesnt exist:\n\t{MODEL_REPO}")
    return


# ## Local Entrypoint
# This is the backbone of the example: it parses inputs, grabs a list of data, instantiates
# the TritonServer embedder application, and passes data to it via `map`. `map` spawns
# more and more containers until the list of batches are all processed.
# ### Class Parameterization
# Modal provides two ways to dynamically parameterize classes: through
# [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options)
# and through
# [modal.parameter](https://modal.com/docs/reference/modal.parameter#modalparameter).
# The app.local_entrypoint() main function at the bottom of this example uses these
# features to dynamically construct the inference engine class wrapper. Some features
# are not currently support via `with_options`, e.g. the `buffer_containers` and
# `min_containers` parameters.
# `buffer_containers` this tells Modal to pre-emptively warm a number of containers before they are strictly
# needed. In other words it tells Modal to continuously fire up more and more containers
# until throughput is saturated. To maximize throughput, set `buffer_containers` in the
# app.cls decorator.
#
# ### Inputs:
# * `gpu` is a string specifying the GPU to be used.
# * `max_containers` caps the number of containers allowed to spin-up. Note that this cannot
# be used with `buffer_containers`: *if you want to use this, set* `buffer_containers=None` *above!*
# * `max_concurrent_inputs` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency")
# argument for the inference app via the
# [modal.cls.with_options](https://modal.com/docs/reference/modal.Cls#with_options) API.
# This takes advantage of the asynchronous nature of the embedding inference app.
# * `threads_per_core` oversubscription factor for parallelized I/O (image reading).
# * `batch_size` means the usual thing for machine learning inference: a group of images are processed
#  through the neural network together. This is used during model compilation and `embed`,
# * `model_name` a HuggingFace model path a la [openai/clip-vit-base-patch16]([OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT"));
# * `image_cap` caps the number of images used in this example (e.g. for debugging/testing)
# * `hf_dataset_name` a HuggingFace data path a la "microsoft/cats_vs_dogs"
# * `triton_backend`: 'pytorch' for now; can modify to use other backends
#
# These three parameters are used to pre-process images to the correct size in a big batch
# before inference.
# * `im_chan`: the number of color channels your model is expecting (probably 3)
# * `im_height`: the number of pixels tall your model is expecting the images to be
# * `im_width`: the number of color channels your model is expecting (probably 3)
##
@app.local_entrypoint()
def main(
    # with_options parameters:
    gpu: str = "A10G",
    max_containers: int = None,  # this gets overridden if buffer_containers is not None
    max_concurrent_inputs: int = 2,
    # modal.parameters:
    model_name: str = "openai/clip-vit-base-patch16",
    batch_size: int = 512,
    im_chan: int = 3,
    im_height: int = 224,
    im_width: int = 224,
    # data
    image_cap: int = -1,
    hf_dataset_name: str = "microsoft/cats_vs_dogs",
    n_million_image_test: float = 0,
    # triton cache
    destroy_cache: bool = False,
    triton_backend: str = "pytorch",
    force_rebuild: bool = False,
):
    start_time = perf_counter()

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

    n_ims = len(im_path_list)

    # (0.b) This destroys cache for timing purposes - you probably don't want to do this!

    if destroy_cache:
        destroy_triton_cache.remote()

    # (1.a) Init the model inference app
    # No inputs to with_options if none provided or buffer_used aboe
    autoscaling_config = {"max_containers": max_containers} if max_containers else {}
    # Build the engine
    start_time = perf_counter()

    embedder = TritonServer.with_concurrency(
        max_inputs=max_concurrent_inputs,
    ).with_options(gpu=f"{gpu}", *autoscaling_config)(
        batch_size=batch_size,
        n_engines=max_concurrent_inputs,
        triton_backend=triton_backend,
        model_name=model_name,
        model_input_chan=im_chan,
        model_input_imheight=im_height,
        model_input_imwidth=im_width,
        force_rebuild=force_rebuild,
    )

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
            f"n_ims={n_ims}::concurrency={max_concurrent_inputs}\n"
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
