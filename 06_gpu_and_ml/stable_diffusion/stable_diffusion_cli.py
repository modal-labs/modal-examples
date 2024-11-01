# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A 1600s oil painting of the New York City skyline"]
# tags: ["use-case-image-video-3d"]
# ---

# # Run Stable Diffusion 3.5 Large Turbo from the command line

# This example shows how to run [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) on Modal
# and generate images from your local command line.

# Inference takes about one minute to cold start,
# at which point images are generated at a rate of one image every 1-2 seconds
# for batch sizes between one and 16.

# Below are four images produced by the prompt
# "A princess riding on a pony".

# ![stable diffusion montage](./stable-diffusion-montage-princess.jpg)

# ## Basic setup

import io
import random
import time
from pathlib import Path

import modal

MINUTES = 60

# All Modal programs need an [`App`](https://modal.com/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = modal.App("example-stable-diffusion-cli")

# ## Configuring dependencies

# The model runs remotely inside a [container](https://modal.com/docs/guide/custom-container).
# That means we need to install the necessary dependencies in that container's image.

# Below, we start from a lightweight base Linux image
# and then install our Python dependencies, like Hugging Face's `diffusers` library and `torch`.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .entrypoint([])  # deactivate default entrypoint to reduce log verbosity
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

with image.imports():
    import diffusers
    import torch

# ## Implementing SD3.5 Large Turbo inference on Modal

# We wrap inference in a Modal [Cls](https://modal.com/docs/guide/lifecycle-methods)
# that ensures models are downloaded when we `build` our container image (just like our dependencies)
# and that models are loaded and then moved to the GPU when a new container starts.

# The `run_inference` function just wraps a `diffusers` pipeline.
# It sends the output image back to the client as bytes.

model_id = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
model_revision_id = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
)
class StableDiffusion:
    @modal.build()
    @modal.enter()
    def initialize(self):
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            model_id,
            revision=model_revision_id,
            torch_dtype=torch.bfloat16,
        )

    @modal.enter()
    def move_to_gpu(self):
        self.pipe.to("cuda")

    @modal.method()
    def run_inference(
        self, prompt: str, batch_size: int = 4, seed: int = None
    ) -> list[bytes]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)
        images = self.pipe(
            prompt,
            num_images_per_prompt=batch_size,  # outputting multiple images per prompt is much cheaper than separate calls
            num_inference_steps=4,  # turbo is tuned to run in four steps
            guidance_scale=0.0,  # turbo doesn't use CFG
            max_sequence_length=512,  # T5-XXL text encoder supports longer sequences, more complex prompts
        ).images

        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()  # reduce fragmentation
        return image_output


# ## Generating images from the command line

# This is the command we'll use to generate images. It takes a text `prompt`,
# a `batch_size` that determines the number of images to generate per prompt,
# and the number of times to run image generation (`samples`).

# You can also provide a `seed` to make sampling more deterministic.


@app.local_entrypoint()
def entrypoint(
    samples: int = 4,
    prompt: str = "A princess riding on a pony",
    batch_size: int = 4,
    seed: int = None,
):
    print(
        f"prompt => {prompt}",
        f"samples => {samples}",
        f"batch_size => {batch_size}",
        f"seed => {seed}",
        sep="\n",
    )

    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    sd = StableDiffusion()

    for sample_idx in range(samples):
        start = time.time()
        images = sd.run_inference.remote(prompt, batch_size, seed)
        duration = time.time() - start
        print(f"Run {sample_idx+1} took {duration:.3f}s")
        if sample_idx:
            print(
                f"\tGenerated {len(images)} image(s) at {(duration)/len(images):.3f}s / image."
            )
        for batch_idx, image_bytes in enumerate(images):
            output_path = (
                output_dir
                / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            )
            if not batch_idx:
                print("Saving outputs", end="\n\t")
            print(
                output_path,
                end="\n" + ("\t" if batch_idx < len(images) - 1 else ""),
            )
            output_path.write_bytes(image_bytes)


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")
