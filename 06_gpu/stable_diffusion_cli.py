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
# into the image. This is technique that allows you to load models much faster
# by using our [high-performance blob storage file server](https://github.com/modal-labs/blobnet).

image = modal.Image.debian_slim().apt_install(["curl"]).run_commands(
    [
    "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117",
    "pip install diffusers[torch] transformers ftfy accelerate"]
).run_commands([
    "curl -L https://gist.github.com/luiscape/36a8cd29b8ed54cfbfcf56d51fe23cc0/raw/a6bf16996efe7c59114eea7944b0f99741d83d54/download_stable_diffusion_models.py | python"
], secrets=[modal.Secret.from_name("huggingface-secret")])
stub.image = image

# ## Global context
#
# Modal allows for you to create a global context that is valid only inside a
# container. It is often useful to load models in this context because it can
# make subsequent calls to the same predict method much faster given that they
# no longer need to instantiate the model. We'll get substantial speedups
# using this technique.

if stub.is_inside():
    import torch
    import diffusers

    # optimizations from https://huggingface.co/docs/diffusers/optimization/fp16#memory-and-speed
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cache_path = "/vol/cache"
    euler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
        cache_path,
        subfolder="scheduler",
        cache_dir=cache_path)
    PIPE = diffusers.StableDiffusionPipeline.from_pretrained(
        cache_path, torch_dtype=torch.float16, scheduler=euler, cache_dir=cache_path).to("cuda")
    # PIPE.enable_attention_slicing()


# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.

@stub.function(gpu=modal.gpu.A100())
def _run_inference(prompt:str, steps:int = 20) -> str:
    with torch.inference_mode():
        image = PIPE(prompt, num_inference_steps=steps, guidance_scale=7.0).images[0]

    return image


# This is the CLI command that we'll use to generate images.

@app.command()
def entrypoint(prompt: str, samples:int = 10):
    typer.echo(f"prompt => {prompt}, samples => {samples}")
    with stub.run():
        for i in range(samples):
            image = _run_inference(prompt)
            image.save(f"output_{i}.png")

# And this is our entrypoint; where the CLI is invoked.

if __name__ == "__main__":
    app()
