# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["a software engineer smoking a pumpkin pipe"]
# ---
# # Stable Diffusion CLI
#
# This example shows Stable Ddiffusion 1.5 with a number of optimizations
# that makes it run faster on Modal. The example takes about 20s to cold start
# and about 1.5s per image generated.
#
# For instance, here are 9 images produced by the prompt
# `An 1600s oil painting of the New York City skyline`
#
# ![stable diffusion slackbot](./stable_diffusion_montage.png)
#
# There is also a [Stable Diffusion Slack bot example](/docs/guide/ex/stable_diffusion_slackbot)
# which does not have all the optimizations, but shows how you can set up a Slack command to
# trigger Stable Diffusion.
#
# ## Optimizations used in this example
#
# As mentioned, we use a few optimizations to run this faster:
#
# * Use [run_function](/docs/reference/modal.Image#run_function) to download the model while building the container image
# * Use a [container lifecycle method](https://modal.com/docs/guide/lifecycle-functions) to initialize the model on container startup
# * Use A100 GPUs
# * Use 16 bit floating point math


# ## Basic setup
import io
import os
import time
from pathlib import Path

import modal

# All Modal programs need a [`Stub`](/docs/reference/modal.Stub) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

stub = modal.Stub("stable-diffusion-cli")

# We will be using `typer` to create our CLI interface.

import typer

app = typer.Typer()

# ## Model dependencies
#
# Your model will be running remotely inside a container. We will be installing
# all the model dependencies in the next step. We will also be "baking the model"
# into the image by running a Python function as a part of building the image.
# This lets us start containers much faster, since all the data that's needed is
# already inside the image.

model_id = "runwayml/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    import torch
    import diffusers

    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

    # Download the Euler A scheduler configuration. Faster than the default and
    # high quality output with fewer steps.
    euler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", use_auth_token=hugging_face_token, cache_dir=cache_path
    )
    euler.save_pretrained(cache_path)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=hugging_face_token, revision="fp16", torch_dtype=torch.float16, cache_dir=cache_path
    )
    pipe.save_pretrained(cache_path)


image = (
    modal.Image.conda()
    .run_commands(
        [
            "conda install xformers -c xformers/label/dev",
            "conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia",
        ]
    )
    .run_commands(["pip install diffusers[torch] transformers ftfy accelerate"])
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)
stub.image = image

# ## Using container lifecycle methods
#
# Modal lets you implement code that runs every time a container starts. This
# can be a huge optimization when you're calling a function multiple times,
# since Modal reuses the same containers when possible.
#
# The way to implement this is to turn the Modal function into a method on a
# class that also implement the Python context manager interface, meaning it
# has the `__enter__` method (the `__exit__` method is optional).
#
# We have also have applied a few model optimizations to make the model run
# faster. On an A100, the model takes about 6.5s to load into memory, and then
# 1.6s per generation on average. On a T4, it takes 13s to load and 3.7s per
# generation. Other optimizations are also available [here](https://huggingface.co/docs/diffusers/optimization/fp16#memory-and-speed).

# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.


class StableDiffusion:
    def __enter__(self):
        import torch
        import diffusers

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        euler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(cache_path, subfolder="scheduler")
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(cache_path, scheduler=euler).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    @stub.function(gpu=modal.gpu.A100())
    def run_inference(self, prompt: str, steps: int = 20) -> bytes:
        import torch

        with torch.inference_mode():
            image = self.pipe(prompt, num_inference_steps=steps, guidance_scale=7.0).images[0]

        # Convert to PNG bytes
        with io.BytesIO() as buf:
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        return image_bytes


# This is the command we'll use to generate images. It takes a `prompt`,
# `samples` (the number of images you want to generate), and `steps` which
# configures the number of inference steps the model will make.


@app.command()
def entrypoint(prompt: str, samples: int = 10, steps: int = 20):
    typer.echo(f"prompt => {prompt}, steps => {steps}, samples => {samples}")

    dir = Path("/tmp/stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    with stub.run():
        sd = StableDiffusion()
        for i in range(samples):
            t0 = time.time()
            image_bytes = sd.run_inference.call(prompt, steps)
            output_path = dir / f"output_{i}.png"
            print(f"Sample {i} took {time.time()-t0:.3f}s. Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `python stable_diffusion_cli.py --help`

if __name__ == "__main__":
    app()
