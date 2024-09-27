# ---
# output-directory: "/tmp/flux"
# ---
# # Run Flux.1 (Schnell) on Modal
#
# Thanks to [@Arro](https://github.com/Arro) for the original contribution.
#
# This example runs the popular [Flux.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) text-to-image model on Modal.
# Flux was created by the folks from [Black Forest Labs](https://blackforestlabs.ai/)
# who previously created Stable Diffusion.
#
# To make this example go [brrrr](https://horace.io/brrr_intro.html) we will use
# the latest version of [Flash Attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file)
# which significanly reduces the latency of the Query, Key, Value (QKV)
# operations of Transformer models on GPUs. The key insight of Flash Attention
# is to break the QKV operations into tiny blocks that can be computed quickly
# in high bandwidth SRAM rather than constantly using low bandwidth VRAM. Read
# more [here](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad).
# Note that Flash Attention 3 (FA3) is specifically designed for H100 GPUs.

# We start off by importing `BytesIO` for return image byte streams,
# along with `Path` and `modal`.

import time
from io import BytesIO
from pathlib import Path

import modal

# We'll use the schnell variant of Flux.1 which is faster but less accurate.
# If you want to use the dev variant, get yourself a Hugging Face  API key,
# put it in your [Secrets](https://modal.com/docs/guide/secrets) and include
# it in the `@app.cls()` below.

VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF


# Build the image with the required dependencies. We use an image with the
# the full CUDA toolkit to ensure we can compile the latest Flash Attention

# diffusers_commit_sha = "1fcb811a8e6351be60304b1d4a4a749c36541651"
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

flux_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .run_commands(
        f"pip install git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} 'numpy<2'"
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
    )
    # Since Hugging Face does not support Flash Attention 3 (FA3) yet (Sep 2024),
    # let's build it from source and add it to our `PYTHONPATH`.
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch==2.4.1",
    )
    .run_commands(
        "ln -s /usr/bin/g++ /usr/bin/clang++",  # Use clang
        "git clone https://github.com/Dao-AILab/flash-attention.git",
        "cd flash-attention/hopper && python setup.py install",
    )
    .env({"PYTHONPATH": "/root/flash-attention/hopper"})
)

# Next we construct our [App](https://modal.com/docs/reference/modal.App)
# and import `torch` & the `FluxPipeline` from Hugging Face's `diffusers` library.

app = modal.App("example-flux")

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline


# Now we can define our `Model` class which will
# 1) Download the model from Hugging Face Hub
# 2) Optimize the Flux's internal Transformer and VAE models to use FA3
# 3) Call `torch.compile() `with the `max-autotune` mode to optimize the models
# even further.
@app.cls(
    gpu="H100",  # Necessary for FA3
    container_idle_timeout=60 * 3,
    image=flux_image,
    timeout=60 * 10,  # 5m -> 10m to leave room for compile time.
)
class Model:
    def _download_model(self):
        from huggingface_hub import snapshot_download

        snapshot_download(f"black-forest-labs/FLUX.1-{VARIANT}")

    @modal.build()
    def build(self):
        self._download_model()

    @modal.enter()
    def enter(self):
        from transformers.utils import move_cache

        self._download_model()
        self.pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16
        )

        self.pipe.to("cuda")
        self.pipe.transformer.fuse_qkv_projections()
        self.pipe.vae.fuse_qkv_projections()
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="max-autotune", fullgraph=True
        )
        self.pipe.vae.to(memory_format=torch.channels_last)
        self.pipe.vae.decode = torch.compile(
            self.pipe.vae.decode, mode="max-autotune", fullgraph=True
        )

        move_cache()

        # Runs for 1-2minutes
        print("Calling pipe() to trigger torch compiliation...")

        self.pipe(
            "Test prompt to trigger torch compilation.",
            output_type="pil",
            num_inference_steps=4,  # use ~50 for [dev], smaller for [schnell]
        ).images[0]

        print("Finished compilation.")

    @modal.method()
    def inference(self, prompt):
        print("Generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,  # use ~50 for [dev], smaller for [schnell]
        ).images[0]
        print("Generated.")

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


# Finally we write our `main` entrypoint to  utilize the `Model`
# class to generate an image. We'll use `BytesIO` to receive the image and
# save it to a JPEG file for easy viewing.

# By default, we hit generate an image twice to demonstrate how much faster
# the inference is once the server is running.


@app.local_entrypoint()
def main(
    prompt: str = "a computer screen showing ASCII terminal art of the"
    " word 'Modal' in neon green. two programmers are pointing excitedly"
    " at the screen.",
    twice=True,
):
    t0 = time.time()
    image_bytes = Model().inference.remote(prompt)
    print(f"1st latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model().inference.remote(prompt)
        print(f"2nd latency: {time.time() - t0:.2f} seconds")

    dir = Path("/tmp/flux")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.jpg"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
