# ---
# output-directory: "/tmp/stable-diffusion"
# ---
# # Stable Diffusion CLI
#
# This tutorial shows how you can create a CLI tool that runs GPU-intensive
# work remotely but feels like you are running locally. We will be building
# a tool that generates an image based on a prompt against Stable Diffusion
# using the HuggingFace Hub and the `diffusers` library.

# ## Basic setup
import modal
import os
import time
from pathlib import Path

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
# into the image using the script `download_stable_diffusion_models.py`.
# This is technique that allows you to copy model files to
# a worker more efficiently because they only need to be moved once.

model_id = "runwayml/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    import torch
    import diffusers

    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

    # Download the Euler A scheduler configuration. Faster than the default and
    # high quality output with fewer steps.
    euler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        model_id,
        subfolder="scheduler",
        use_auth_token=hugging_face_token,
        cache_dir=cache_path)
    euler.save_config(cache_path)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=hugging_face_token,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path)
    pipe.save_pretrained(cache_path)


image = (
    modal.Image.conda()
    .apt_install(["curl"])
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
        gpu=modal.gpu.A100(),
    )
)
stub.image = image

# ## Global context
#
# Modal allows for you to create a global context that is valid only inside a
# container. It is often useful to load models in this context because it can
# make subsequent calls to the same predict method much faster given that they
# no longer need to instantiate the model. We'll get performance improvements
# using this technique.
#
# We have also have applied a few model optimizations to make the model run
# faster. On an A100, the model takes about 6.5s to load into memory, and then
# 1.6s per generation on average. On a T4, it takes 13s to load and 3.7s per
# generation. Other optimizations are also available [here](https://huggingface.co/docs/diffusers/optimization/fp16#memory-and-speed).

try:
    is_inside = stub.is_inside()
except KeyError:
    # TODO: Fix this bug
    is_inside = False

if is_inside:
    import torch
    import diffusers

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    euler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
        cache_path, subfolder="scheduler", cache_dir=cache_path
    )
    PIPE = diffusers.StableDiffusionPipeline.from_pretrained(
        cache_path, torch_dtype=torch.float16, scheduler=euler, cache_dir=cache_path
    ).to("cuda")
    PIPE.enable_xformers_memory_efficient_attention()


# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.


@stub.function(gpu=modal.gpu.A100())
def _run_inference(prompt: str, steps: int = 20) -> str:
    with torch.inference_mode():
        image = PIPE(prompt, num_inference_steps=steps, guidance_scale=7.0).images[0]

    return image


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
        for i in range(samples):
            t0 = time.time()
            image = _run_inference(prompt, steps)
            output_path = dir / f"output_{i}.png"
            print(f"Sample {i} took {time.time()-t0:.3f}s. Saving it to {output_path}")
            image.save(output_path)


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `python stable_diffusion_cli.py --help`

if __name__ == "__main__":
    app()
