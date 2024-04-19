# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A 1600s oil painting of the New York City skyline"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion CLI
#
# This example shows Stable Diffusion 1.5 with a number of optimizations
# that makes it run faster on Modal. The example takes about 10s to cold start
# and about 1.0s per image generated.
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
# * Use [run_function](/docs/reference/modal.Image#run_function) to download the model while building the container image
# * Use a [container lifecycle method](https://modal.com/docs/guide/lifecycle-functions) to initialize the model on container startup
# * Use A10G GPUs
# * Use 16 bit floating point math


# ## Basic setup
from __future__ import annotations

import io
import time
from pathlib import Path

from modal import App, Image, build, enter, method

# All Modal programs need a [`App`](/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = App("stable-diffusion-cli")

# ## Model dependencies
#
# Your model will be running remotely inside a container. We will be installing
# all the model dependencies in the next step. We will also be "baking the model"
# into the image by running a Python function as a part of building the image.
# This lets us start containers much faster, since all the data that's needed is
# already inside the image.

model_id = "runwayml/stable-diffusion-v1-5"

image = Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.29.2",
    "diffusers==0.15.1",
    "ftfy==6.2.0",
    "safetensors==0.4.2",
    "torch==2.2.2",
    "torchvision",
    "transformers~=4.25.1",
    "triton~=2.2.0",
    "xformers==0.0.25post1",
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

# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.


@app.cls(image=image, gpu="A10G")
class StableDiffusion:
    @build()
    @enter()
    def initialize(self):
        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
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
