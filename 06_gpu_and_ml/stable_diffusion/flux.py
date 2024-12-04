# ---
# output-directory: "/tmp/flux"
# args: ["--no-compile"]
# ---

# # Run Flux fast on H100s with `torch.compile`

# In this guide, we'll run Flux as fast as possible on Modal using open source tools.
# We'll use `torch.compile` and NVIDIA H100 GPUs.

# ## Setting up the image and dependencies

import time
from io import BytesIO
from pathlib import Path

import modal

# We'll make use of the full [CUDA toolkit](https://modal.com/docs/guide/cuda)
# in this example, so we'll build our container image off of the `nvidia/cuda` base.

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# Now we install most of our dependencies with `apt` and `pip`.
# For Hugging Face's [Diffusers](https://github.com/huggingface/diffusers) library
# we install from GitHub source and so pin to a specific commit.

# PyTorch added [faster attention kernels for Hopper GPUs in version 2.5

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Later, we'll also use `torch.compile` to increase the speed further.
# Torch compilation needs to be re-executed when each new container starts,
# So we turn on some extra caching to reduce compile times for later containers.

flux_image = flux_image.env(
    {"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"}
).env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})

# Finally, we construct our Modal [App](https://modal.com/docs/reference/modal.App),
# set its default image to the one we just constructed,
# and import `FluxPipeline` for downloading and running Flux.1.

app = modal.App("example-flux", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline

# ## Defining a parameterized `Model` inference class

# Next, we map the model's setup and inference code onto Modal.

# 1. We run any setup that can be persisted to disk in methods decorated with `@build`.
# In this example, that includes downloading the model weights.
# 2. We run any additional setup, like moving the model to the GPU, in methods decorated with `@enter`.
# We do our model optimizations in this step. For details, see the section on `torch.compile` below.
# 3. We run the actual inference in methods decorated with `@method`.

MINUTES = 60  # seconds
VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF
NUM_INFERENCE_STEPS = 4  # use ~50 for [dev], smaller for [schnell]


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    container_idle_timeout=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
)
class Model:
    compile: int = (  # see section on torch.compile below for details
        modal.parameter(default=0)
    )

    def setup_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        snapshot_download(f"black-forest-labs/FLUX.1-{VARIANT}")

        move_cache()

        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16
        )

        return pipe

    @modal.build()
    def build(self):
        self.setup_model()

    @modal.enter()
    def enter(self):
        pipe = self.setup_model()
        pipe.to("cuda")  # move model to GPU
        self.pipe = optimize(pipe, compile=bool(self.compile))

    @modal.method()
    def inference(self, prompt: str) -> bytes:
        print("🎨 generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


# ## Calling our inference function

# To generate an image we just need to call the `Model`'s `generate` method
# with `.remote` appended to it.
# You can call `.generate.remote` from any Python environment that has access to your Modal credentials.
# The local environment will get back the image as bytes.

# Here, we wrap the call in a Modal [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
# so that it can be run with `modal run`:

# ```bash
# modal run flux.py
# ```

# By default, we call `generate` twice to demonstrate how much faster
# the inference is after cold start. In our tests, clients received images in about 1.2 seconds.
# We save the output bytes to a temporary file.


@app.local_entrypoint()
def main(
    prompt: str = "a computer screen showing ASCII terminal art of the"
    " word 'Modal' in neon green. two programmers are pointing excitedly"
    " at the screen.",
    twice: bool = True,
    compile: bool = False,
):
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)


# ## Speeding up Flux with `torch.compile`

# By default, we do some basic optimizations, like adjusting memory layout
# and re-expressing the attention head projections as a single matrix multiplication.
# But there are additional speedups to be had!

# PyTorch 2 added a compiler that optimizes the
# compute graphs created dynamically during PyTorch execution.
# This feature helps close the gap with the performance of static graph frameworks
# like TensorRT and TensorFlow.

# Here, we follow the suggestions from Hugging Face's
# [guide to fast diffusion inference](https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion),
# which we verified with our own internal benchmarks.
# Review that guide for detailed explanations of the choices made below.

# The resulting compiled Flux `schnell` deployment returns images to the client in under a second (~700 ms), according to our testing.
# _Super schnell_!

# Compilation takes up to twenty minutes on first iteration.
# As of time of writing in late 2024,
# the compilation artifacts cannot be fully serialized,
# so some compilation work must be re-executed every time a new container is started.
# That includes when scaling up an existing deployment or the first time a Function is invoked with `modal run`.

# We cache compilation outputs from `nvcc`, `triton`, and `inductor`,
# which can reduce compilation time by up to an order of magnitude.
# For details see [this tutorial](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

# You can turn on compilation with the `--compile` flag.
# Try it out with:

# ```bash
# modal run flux.py --compile
# ```

# The `compile` option is passed by a [`modal.parameter`](https://modal.com/docs/reference/modal.parameter#modalparameter) on our class.
# Each different choice for a `parameter` creates a [separate auto-scaling deployment](https://modal.com/docs/guide/parameterized-functions).
# That means your client can use arbitrary logic to decide whether to hit a compiled or eager endpoint.


def optimize(pipe, compile=True):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compiliation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=NUM_INFERENCE_STEPS,  # use ~50 for [dev], smaller for [schnell]
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe
