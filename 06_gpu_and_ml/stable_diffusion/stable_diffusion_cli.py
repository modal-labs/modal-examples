# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A 1600s oil painting of the New York City skyline"]
# runtimes: ["runc", "gvisor"]
# tags: ["use-case-image-video-3d"]
# ---
# # Stable Diffusion CLI
#
# This example shows Stable Diffusion 3.5 Medium with a number of optimizations
# that makes it run faster on Modal. The example takes about 30s to cold start
# and about 6.0s per image generated.
#
# To use the XL 1.0 model, see the example posted [here](/docs/examples/stable_diffusion_xl).
#
# For instance, here are 9 images produced by the prompt
# `A 1600s oil painting of the New York City skyline`
#
# ![stable diffusion montage](./stable_diffusion_montage.png)
#
# As mentioned, we use a few optimizations to run this faster:
#
# * Use a [container lifecycle method](https://modal.com/docs/guide/lifecycle-functions) to initialize the model on container startup
# * Use A10G GPUs
# * Use 16 bit floating point math


# ## Basic setup
from __future__ import annotations

import io
import time
from pathlib import Path

import modal

# All Modal programs need a [`App`](/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = modal.App("stable-diffusion-cli")

# ## Model dependencies
#
# Your model will be running remotely inside a container. We will be installing
# all the model dependencies in the next step. We will also be "baking the model"
# into the image by downloading the weights as part of image build.
# This lets us start containers much faster, since all the data that's needed is
# already inside the image.

model_id = "stabilityai/stable-diffusion-3.5-medium"

tag = "12.6.2-cudnn-devel-ubuntu22.04"

image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).pip_install(
    "accelerate==0.33.0",
    "diffusers==0.31.0",
    # "ftfy==6.3.1",
    # "safetensors==0.4.4",
    "sentencepiece==0.2.0",
    "torch==2.5.0",
    "torchvision==0.20.0",
    "transformers~=4.44.0",
    # "triton~=3.1.0",
    # "xformers==0.0.28.post2",
)

with image.imports():
    import diffusers
    import torch

# ## Using container lifecycle methods
#
# Modal lets you implement code that runs every time a container starts. This
# can be a huge optimization when you're calling a function multiple times,
# since Modal reuses the same containers when possible.
#
# The way to implement this is to turn the Modal function into a method on a
# class that also has lifecycle methods (decorated with `@enter()` and/or `@exit()`).
#
# We have also have applied a few model optimizations to make the model run
# faster. On an A10G, the model takes about 6.5s to load into memory, and then
# 1.6s per generation on average. On a T4, it takes 13s to load and 3.7s per
# generation. Other optimizations are also available [here](https://huggingface.co/docs/diffusers/optimization/fp16#memory-and-speed).

# This is our Modal function. The function runs through the `StableDiffusion3Pipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret-ren")],
    timeout=6000,
)
class StableDiffusion:
    @modal.build()
    @modal.enter()
    def initialize(self):
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.bfloat16,
        )

    @modal.method()
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
        # Move the pipeline to CUDA
        self.pipe.to("cuda")

        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [prompt] * batch_size,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


# This is the command we'll use to generate images. It takes a `prompt`,
# `samples` (the number of images you want to generate), `steps` which
# configures the number of inference steps the model will make, and `batch_size`
# which determines how many images to generate for a given prompt.


@app.local_entrypoint()
def entrypoint(
    prompt: str = "A 1600s oil painting of the New York City skyline",
    samples: int = 5,
    steps: int = 10,
    batch_size: int = 1,
):
    print(
        f"prompt => {prompt}, steps => {steps}, samples => {samples}, batch_size => {batch_size}"
    )

    dir = Path("/tmp/stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    sd = StableDiffusion()
    for i in range(samples):
        t0 = time.time()
        images = sd.run_inference.remote(prompt, steps, batch_size)
        total_time = time.time() - t0
        print(
            f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image)."
        )
        for j, image_bytes in enumerate(images):
            output_path = dir / f"output_{j}_{i}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_cli.py --help`
#
# ## Performance
#
# This example can generate pictures in about a second, with startup time of about 10s for the first picture.
#
# See distribution of latencies below. This data was gathered by running 500 requests in sequence (meaning only
# the first request incurs a cold start). As you can see, the 90th percentile is 1.2s and the 99th percentile is 2.30s.
#
# ![latencies](./stable_diffusion_latencies.png)
