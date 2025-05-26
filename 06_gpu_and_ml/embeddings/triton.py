#!/usr/bin/env python3

# # Maximizing throughput on Triton Inference Server


# ## Local env imports
# # Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import os
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Iterator, Sequence, Tuple, List

import modal


# ────────────────────────────── Constants ──────────────────────────────
HF_SECRET = modal.Secret.from_name("huggingface-secret")
VOL_NAME = "example-embedding-data"
VOL_MNT = Path("/data")
data_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
MODEL_REPO = VOL_MNT / "triton_repo"  # will hold model.plan + config


# image with Triton + torch + tritonclient (tiny helper)
TRITON_IMAGE = (
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
    "clip-triton-embed",
    image=TRITON_IMAGE,
    volumes={VOL_MNT: data_volume},
    secrets=[HF_SECRET],
)

with TRITON_IMAGE.imports():
    import torch, torchvision  # noqa: F401   – for torchscript
    from transformers import CLIPVisionModel, CLIPImageProcessorFast
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image
    import tritonclient.http as httpclient

##
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
    image=TRITON_IMAGE,
    volumes={VOL_MNT: data_volume},
    max_containers=1,  # We only want one container to handle volume setup
    cpu=4,  # HuggingFace will use multi-process parallelism to download
    timeout=24 * 60 * 60,  # if using a large HF dataset, this may need to be longer
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


# ────────────────────── Triton Server wrapper class ─────────────────────
@app.cls(
    image=TRITON_IMAGE,
    volumes={VOL_MNT: data_volume},
    cpu=4,
    memory=5 * 1024,  # MB -> GB
    timeout=10 * 60,
)
class TritonServer:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=1)
    n_engines: int = modal.parameter(default=1)
    triton_backend: str = modal.parameter(default="tensorrt")

    model_input_chan: int = modal.parameter(default=3)
    model_input_imheight: int = modal.parameter(default=224)
    model_input_imwidth: int = modal.parameter(default=224)
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
            self._client.load_model(self.triton_model_name)  # ← added line

        # Heartbeat
        self._client = httpclient.InferenceServerClient(url="localhost:8000")
        seconds_wait = 60
        for _ in range(seconds_wait * 2):  # wait up to 1min
            try:
                if self._client.is_model_ready(self.triton_model_name):
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Triton failed to become ready")

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

        Parameters
        ----------
        model_name : str      HuggingFace model id
        version    : str      Triton version directory
        fp16       : bool     Trace / build in FP16 mode
        engine     : str      'pytorch'  → TorchScript + PyTorch backend
                            'tensorrt' → TensorRT engine + TensorRT backend
        """
        import subprocess, torch, json, os
        from pathlib import Path
        from textwrap import dedent
        from torchvision.io import read_image
        from torch import jit
        from torch.onnx import export as onnx_export

        repo_dir = Path(MODEL_REPO) / self.triton_model_name / version
        repo_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # 0. short-circuit if artifacts & config already exist
        artifact = repo_dir / (
            "model.pt" if self.triton_backend == "pytorch" else "model.plan"
        )
        cfg_file = Path(MODEL_REPO) / self.triton_model_name / "config.pbtxt"
        if artifact.exists() and cfg_file.exists():
            print("Model repo already complete – skip build.")
            return

        # ------------------------------------------------------------------ #
        print("Building torch model...", end="")
        st = perf_counter()

        # 1.  Build Torch module (used for *both* backends)
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
        # ------------------------------------------------------------------ #
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

        elif self.triton_backend == "tensorrt":
            onnx_path = repo_dir / "model.onnx"
            print("Exporting ONXX... ", end="")
            st = perf_counter()
            onnx_export(
                model.cpu(),  # ONNX must be on CPU
                example.cpu(),
                onnx_path,
                input_names=["input0"],
                output_names=["output0"],
                dynamic_axes={"input0": {0: "batch"}, "output0": {0: "batch"}},
                opset_version=17,
            )
            print(f"took {perf_counter() - st:.2E}s")

            size_str = "x".join(self.in_shape)
            print(f"SIZESTR=={size_str}")

            print("Running:\n\t", " ".join(cmd))
            st = perf_counter()
            plan_path = repo_dir / "model.plan"
            # --fp16 flag assumes GPU supports it; change to --fp32 if not
            cmd = [
                "/usr/src/tensorrt/bin/trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={plan_path}",
                "--fp16" if fp16 else "--fp32",
                f"--minShapes=input0:1x{'x'.join(self.in_shapes[1:])}",
                f"--optShapes=input0:{size_str}",
                f"--maxShapes=input0:{size_str}",
                "--workspace=4096",
                "--verbose",
            ]
            subprocess.run(cmd, check=True)
            print(f"\n\t.....->took {perf_counter() - st:.2E}s")

        else:
            raise ValueError(
                f"Triton backend `{self.triton_backend}` not"
                "recognized; try `pytorch` or `tensorrt`"
            )
        # ------------------------------------------------------------------ #
        # 3.  Generate config.pbtxt
        dtype = "TYPE_FP16" if fp16 else "TYPE_FP32"
        cfg_text = self.make_config(
            name=self.triton_model_name,
            dtype=dtype,
            output_dim=512,
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
            
            """

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

    def read_batch(
        self,
        im_path_list: list[os.PathLike],
    ) -> list["Image"]:  # TODO: fix this typehint
        """
        Read a batch of data. We use Threads to parallelize this I/O-bound task,
        and finally toss the batch into the CLIPImageProcessorFast preprocessor.
        """

        def readim(impath: os.PathLike):
            """
            Prepends this container's volume mount location to the image path.
            """
            return read_image(str(VOL_MNT / impath))

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 8) as executor:
            images = list(executor.map(readim, im_path_list))

        return (torch.stack(images).to(torch.float16) / 255).numpy()

    @modal.method()
    async def embed(self, imgs: list[os.PathLike]) -> Tuple[float, int]:
        """Read batch → POST to Triton → return latency + count"""
        st = perf_counter()
        batch = self.read_batch(imgs)
        msg = f"Time to create batch: {perf_counter() - st:.2e}"
        st = perf_counter()
        inp = httpclient.InferInput("input0", batch.shape, "FP16")
        inp.set_data_from_numpy(batch, binary_data=True)

        out = httpclient.InferRequestedOutput("output0")
        _ = self._client.infer(self.triton_model_name, [inp], outputs=[out])
        inftime = perf_counter() - st
        msg += f" || Time to inf: {inftime:.2e}"
        return inftime, len(imgs)

    @modal.exit()
    def _cleanup(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()


@app.function(image=TRITON_IMAGE, volumes={VOL_MNT: data_volume})
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


# ───────────────────────────── Local entrypoint ─────────────────────────
#
@app.local_entrypoint()
def main(
    # with_options parameters:
    gpu: str = "A10G",
    min_containers: int = 1,
    max_containers: int = 50,  # this gets overridden if buffer_containers is not None
    allow_concurrent_inputs: int = 1,
    # modal.parameters:
    n_models: int = None,  # defaults to match `allow_concurrent_parameters`
    model_name: str = "openai/clip-vit-base-patch16",
    batch_size: int = 100,
    im_chan: int = 3,
    im_height: int = 224,
    im_width: int = 224,
    # data
    image_cap: int = -1,
    hf_dataset_name: str = "microsoft/cats_vs_dogs",
    million_image_test: bool = False,
    # triton cache
    destroy_cache: bool = False,
    # logging (optional)
    log_file: str = None,  # TODO: remove local logging from example
    triton_backend: str = "pytorch",
    n_gpu: int = 1,
):
    start_time = perf_counter()

    # (0.a) Catalog data: modify `catalog_jpegs` to fetch batches of your data paths.
    extracted_path = Path("extracted") / hf_dataset_name
    im_path_list, vol_setup_time = catalog_jpegs.remote(
        dataset_namespace=hf_dataset_name,
        cache_dir=extracted_path,
        image_cap=image_cap,
        model_input_shape=(im_chan, im_height, im_width),
    )
    print(f"Took {vol_setup_time:.2f}s to setup volume.")
    if million_image_test:
        print("WARNING: `million_image_test` FLAG RECEIVED! RESETTING BSZ ETC!")
        mil = int(1e6)
        while len(im_path_list) < mil:
            im_path_list += im_path_list
        im_path_list = im_path_list[:mil]
    n_ims = len(im_path_list)

    # (0.b) This destroys cache for timing purposes - you probably don't want to do this!

    if destroy_cache:
        destroy_triton_cache.remote()

    # (1.a) Init the model inference app
    # No inputs to with_options if none provided or buffer_used aboe
    buffer_containers = None
    make_empty = (buffer_containers is not None) or (max_containers is None)
    container_config = {} if make_empty else {"max_containers": max_containers}
    # Build the engine
    start_time = perf_counter()
    # embedder = TorchCompileEngine.with_options(
    #     gpu=gpu, allow_concurrent_inputs=allow_concurrent_inputs, **container_config
    # )(
    #     batch_size=batch_size,
    #     n_engines=n_models if n_models else allow_concurrent_inputs,
    #     model_name=model_name,
    #     model_input_chan=model_input_chan,
    #     model_input_imheight=model_input_imheight,
    #     model_input_imwidth=model_input_imwidth,
    #     threads_per_core=threads_per_core,
    # )

    embedder = TritonServer.with_options(
        gpu=f"{gpu}:{n_gpu}",
        max_containers=max_containers,
    ).with_concurrency(
        max_inputs=allow_concurrent_inputs,
    )(
        batch_size=batch_size,
        n_engines=allow_concurrent_inputs,
        triton_backend=triton_backend,
        model_name=model_name,
        model_input_chan=im_chan,
        model_input_imheight=im_height,
        model_input_imwidth=im_width,
    )

    # (2) Embed batches via remote `map` call
    times, batchsizes = [], []
    # embedder.embed.spawn_map(chunked(im_path_list, batch_size))
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
