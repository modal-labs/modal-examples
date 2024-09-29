# ---
# output-directory: "/tmp/flux"
# ---
# # Run Flux.1 on Modal
#
# When you want to generate the highest quality images there's no better model
# than Flux.1, made by the folks at [Black Forest Labs](https://blackforestlabs.ai/).
# But why pay someone else to generate the images when you can do it yourself
# for a fraction of the cost in just a few minutes.
# This example will show you how on Modal.
#
# ## Overview
# In this guide we'll show you how to compile Flux to run at bleeding edge speeds,
# create a Modal class for downloading and serving the model, and generate
# unlimited images via `remote` calls in a Modal entrypoint.

# ### Flux Variants
# We'll use the `schnell` variant of Flux.1 which is the fastest but lowest quality model in the series.
# If you want to use the `dev` variant: get yourself a Hugging Face API key,
# put it in your Modal [Secrets](https://modal.com/docs/guide/secrets) and include
# it in the `@app.cls()` below.

# ### Background: Bleeding Edge Inference with Flash Attention 3
# To run Flux.1 at bleeding edge speeds we will use [Flash Attention (FA3)](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file)
# which makes the attention blocks in Transformers go
# [brrrr](https://horace.io/brrr_intro.html) on GPUs. FA3 does this by
# breaking the Query, Key, and Value (QKV) operations into tiny block that can be
# computed quickly in high bandwidth SRAM rather than overusing low
# bandwidth VRAM. Read more [here](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad).

# ### Quick note
# Thanks to [@Arro](https://github.com/Arro) for the original contribution to
# this example.
#
# ## Setting up the image and dependencies
# To set up we'll need to import `BytesIO` for returning image bytes,
# use a specific CUDA image to ensure we can compile FA3, build and install FA3,
# and import the Hugging Face `diffusers` library to download and run Flux via
# the `FluxPipeline` class.

import time
from io import BytesIO
from pathlib import Path

import modal

VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF

# Here we set the parameters for the CUDA image.

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Now we create our image and install the required dependencies.

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"
flash_commit_sha = "53a4f341634fcbc96bb999a3c804c192ea14f2ea"

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
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        # Below are for FA3
        "ninja==1.11.1.1",
        "packaging==24.1",
        "wheel==0.44.0",
        "torch==2.4.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} 'numpy<2'",
    )
    .run_commands(
        # Build Flash Attention 3 from source and add it to PYTHONPATH
        "ln -s /usr/bin/g++ /usr/bin/clang++",  # Use clang
        "git clone https://github.com/Dao-AILab/flash-attention.git",
        f"cd flash-attention && git checkout {flash_commit_sha}",
        "cd flash-attention/hopper && python setup.py install",
    )
    .env({"PYTHONPATH": "/root/flash-attention/hopper"})
)

# Next we construct our [App](https://modal.com/docs/reference/modal.App)
# and import `FluxPipeline` for downloading and running Flux.1.

app = modal.App("example-flux")

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline

# ## Serving inference at 1 image every second
# We'll define our `Model` class which will set us up for fast inference in
# 3 steps:
# 1. Download the model from Hugging Face Hub.
# 2. Swap in FA3 into Flux's internal Transformer and Variational Auto Encoder
# (VAE) models.
# 3. Call `torch.compile()` in `max-autotune` mode to optimize the models

# Once we do that we define the `generate` method which generates images via
# Flux given a text `prompt`. At Modal's `H100` pricing, generating an image every
# second works out to less than 1 cent per image!


@app.cls(
    gpu="H100",  # Necessary for FA3
    container_idle_timeout=60 * 3,
    image=flux_image,
    timeout=60 * 20,  # 20 minutes to leave plenty of room for compile time.
)
class Model:
    def download_model(self):
        from huggingface_hub import snapshot_download

        snapshot_download(f"black-forest-labs/FLUX.1-{VARIANT}")

    @modal.build()
    def build(self):
        self.download_model()

    @modal.enter()
    def enter(self):
        from transformers.utils import move_cache

        self.download_model()  # Ensure model is downloaded.
        self.pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16
        )

        self.pipe.to("cuda")  # Move to GPU
        # Fush QKV projections in Transformer and VAE and apply FA3
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

        # Trigger torch compilation
        print(
            "Calling pipe() to trigger torch compiliation (may take ~10"
            " minutes)..."
        )

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


# ## Calling our inference function
# To generate an image we just need to call the `Model`'s `.generate` method
# with `.remote` appended to it. Then we we can save the returned bytes into a
# JPEG file for easy viewing.

# We wrap this call in a Modal entrypoint function called `main`.
# By default, we `generate` an image twice to demonstrate how much faster
# the inference is once the server is running and the model is compiled with
# FA3.


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


# That's it! The first time you run it will take a several minutes but the
# second latency will be around ~1 second.
