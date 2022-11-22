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

model_cache_path = "/model_cache"
image = modal.Image.debian_slim().run_commands(
    [
    "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117",
    "pip install diffusers[torch] transformers ftfy"]
).run_commands([
    f"""python -c 'import diffusers; import os; euler = diffusers.EulerAncestralDiscreteScheduler.from_config("runwayml/stable-diffusion-v1-5",subfolder="scheduler",use_auth_token=os.environ["HUGGINGFACE_TOKEN"], cache_dir="{model_cache_path}"); euler.save_config("{model_cache_path}")'""",
    f"""python -c 'import diffusers; import os; import torch; pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",use_auth_token=os.environ["HUGGINGFACE_TOKEN"],revision="fp16",torch_dtype=torch.float16,cache_dir="{model_cache_path}"); pipe.save_pretrained("{model_cache_path}")'"""
], secrets=[modal.Secret.from_name("huggingface-secret")]).run_commands([
    f"ls -lah {model_cache_path}"
])
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

    model_id = "/model_cache"
    euler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        model_id,
        subfolder="scheduler",
        cache_dir=model_cache_path)
    PIPE = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, scheduler=euler, cache_dir=model_cache_path).to("cuda")
    PIPE.enable_attention_slicing()


# This is our Modal function. The function runs through the `StableDiffusionPipeline` pipeline.
# It sends the PIL image back to our CLI where we save the resulting image in a local file.

@stub.function(gpu=modal.gpu.A100())
def _run_inference(prompt:str) -> str:
    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = PIPE(prompt, num_inference_steps=20, guidance_scale=7.0).images[0]

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
