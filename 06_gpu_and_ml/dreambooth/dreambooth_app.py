# ---
# deploy: true
# ---
#
# # Pet Art Dreambooth with Hugging Face and Gradio
#
# This example finetunes the [Stable Diffusion v1.5 model](https://huggingface.co/runwayml/stable-diffusion-v1-5)
# on images of a pet (by default, a puppy named Qwerty)
# using a technique called textual inversion from [the "Dreambooth" paper](https://dreambooth.github.io/).
# Effectively, it teaches a general image generation model a new "proper noun",
# allowing for the personalized generation of art and photos.
# It then makes the model shareable with others using the [Gradio.app](https://gradio.app/)
# web interface framework.
#
# It demonstrates a simple, productive, and cost-effective pathway
# to building on large pretrained models
# by using Modal's building blocks, like
# [GPU-accelerated](https://modal.com/docs/guide/gpu) Modal Functions, [volumes](/docs/guide/volumes) for caching, and [Modal webhooks](https://modal.com/docs/guide/webhooks#webhook).
#
# And with some light customization, you can use it to generate images of your pet!
#
# ![Gradio.app image generation interface](./gradio-image-generate.png)
#
# ## Setting up the dependencies
#
# We can start from a base image and specify all of our dependencies.

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from modal import (
    Image,
    Mount,
    Secret,
    Stub,
    Volume,
    asgi_app,
    enter,
    method,
)

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"
stub = Stub(name="example-dreambooth-app")

# Commit in `diffusers` to checkout the training script from.
GIT_SHA = "abd922bd0c43a504e47eca2ed354c3634bd00834"

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.27.2",
        "datasets~=2.13.0",
        "ftfy~=6.1.0",
        "gradio~=3.50.2",
        "smart_open~=6.4.0",
        "transformers~=4.38.1",
        "torch~=2.2.0",
        "torchvision~=0.16",
        "triton~=2.2.0",
        "peft==0.7.0",
        "wandb==0.16.3"
    )
    .apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's current working directory, /root. Then install
    # the `diffusers` package.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)

def download_models():
    import torch
    from diffusers import AutoencoderKL, DiffusionPipeline
    from transformers.utils import move_cache

    DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
        torch_dtype=torch.float16
    )
    move_cache()

# ## TODO: Explain this

image = image.run_function(download_models)


# A persisted `modal.Volume` will store model artefacts across Modal app runs.
# This is crucial as finetuning runs are separate from the Gradio app we run as a webhook.

volume = Volume.persisted("dreambooth-finetuning-volume")
MODEL_DIR = "/model"  # TODO: loras dir?

# ## Config
#
# All configs get their own dataclasses to avoid scattering special/magic values throughout code.
# You can read more about how the values in `TrainConfig` are chosen and adjusted [in this blog post on Hugging Face](https://huggingface.co/blog/dreambooth).
# To run training on images of your own pet, upload the images to separate URLs and edit the contents of the file at `TrainConfig.instance_example_urls_file` to point to them.


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"
    # identifier for pretrained model on Hugging Face
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 1024
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 100
    checkpointing_steps: int = 1000
    seed: int = 117


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


# ## Get finetuning dataset
#
# Part of the magic of Dreambooth is that we only need 3-10 images for finetuning.
# So we can fetch just a few images, stored on consumer platforms like Imgur or Google Drive
# -- no need for expensive data collection or data engineering.


def load_images(image_urls: list[str]) -> Path:
    import PIL.Image
    from smart_open import open

    img_path = Path("/img")

    img_path.mkdir(parents=True, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(img_path / f"{ii}.png")
    print(f"{ii + 1} images loaded")

    return img_path


# ## Finetuning a text-to-image model
#
# This model is trained to do a sort of "reverse [ekphrasis](https://en.wikipedia.org/wiki/Ekphrasis)":
# it attempts to recreate a visual work of art or image from only its description.
#
# We can use a trained model to synthesize wholly new images
# by combining the concepts it has learned from the training data.
#
# We use a pretrained model, version 1.5 of the Stable Diffusion model. In this example, we "finetune" SD v1.5, making only small adjustments to the weights,
# in order to just teach it a new word: the name of our pet.
#
# The result is a model that can generate novel images of our pet:
# as an astronaut in space, as painted by Van Gogh or Bastiat, etc.
#
# ### Finetuning with Hugging Face 🧨 Diffusers and Accelerate
#
# The model weights, libraries, and training script are all provided by [🤗 Hugging Face](https://huggingface.co).
#
# You can kick off a training job with the command `modal run dreambooth_app.py::stub.train`.
# It should take about ten minutes.
#
# Tip: if the results you're seeing don't match the prompt too well, and instead produce an image
# of your subject again, the model has likely overfit. In this case, repeat training with a lower
# value of `max_train_steps`. On the other hand, if the results don't look like your subject, you
# might need to increase `max_train_steps`.


@stub.function(
    image=image,
    gpu="A100",  # fine-tuning is VRAM-heavy and requires an A100 GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1800,  # 30 minutes
    secrets=[Secret.from_name("my-wandb-secret")]  # TODO: optional wandb secret
)
def train(instance_example_urls):
    import subprocess

    from accelerate.utils import write_basic_config

    # set up TrainConfig
    config = TrainConfig()

    # set up runner-local image and shared model weight directories
    img_path = load_images(instance_example_urls)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_sdxl.py",
            # "--train_text_encoder",  # needs at least 16GB of GPU RAM.
            "--mixed_precision=fp16",
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={img_path}",
            f"--pretrained_vae_model_name_or_path={config.vae_name}",  # required for numerical stability in fp16
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt={prompt}",
            f"--validation_prompt={prompt} in space",
            f"--validation_epochs={config.max_train_steps // 5}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",
            "--report_to=wandb",
        ]
    )
    # The trained model artefacts have been output to the volume mounted at `MODEL_DIR`.
    # To persist these artefacts for use in future inference function calls, we 'commit' the changes
    # to the volume.
    volume.commit()


# ## The inference function.
#
# To generate images from prompts using our fine-tuned model, we define a function called `inference`.
# In order to initialize the model just once on container startup, we use Modal's [container
# lifecycle](https://modal.com/docs/guide/lifecycle-functions) feature, which requires the function to be part
# of a class.  The `modal.Volume` is mounted at `MODEL_DIR`, so that the fine-tuned model created  by `train` is then available to `inference`.


@stub.cls(
    image=image,
    gpu="A10G",
    volumes={MODEL_DIR: volume},
)
class Model:
    @enter()
    def load_model(self):
        import torch
        from diffusers import AutoencoderKL, DiffusionPipeline

        config = TrainConfig()

        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()

        # set up a hugging face inference pipeline using our model
        # dpm = DPMSolverMultistepScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")

        # TODO: load the pretrained models in the container image with run_function
        pipe = DiffusionPipeline.from_pretrained(
            config.model_name,
            vae=AutoencoderKL.from_pretrained(config.vae_name, torch_dtype=torch.float16),
            # scheduler=dpm,  # TODO: investigate schedulers
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        # pipe.enable_xformers_memory_efficient_attention()
        self.pipe = pipe

    @method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


# ## Wrap the trained model in Gradio's web UI
#
# Gradio.app makes it super easy to expose a model's functionality
# in an easy-to-use, responsive web interface.
#
# This model is a text-to-image generator,
# so we set up an interface that includes a user-entry text box
# and a frame for displaying images.
#
# We also provide some example text inputs to help
# guide users and to kick-start their creative juices.
#
# You can deploy the app on Modal forever with the command
# `modal deploy dreambooth_app.py`.


@stub.function(
    image=image,
    concurrency_limit=3,
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call to the GPU inference function on Modal.
    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/examples/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

### Learn how to make a "Dreambooth" for your own pet [here]({modal_example_url}).
    """

    @web_app.get('/favicon.ico', include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/static-gradient.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/static-gradient.svg")

    # add a gradio UI around inference
    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(primary_hue="green", secondary_hue="emerald", neutral_hue="neutral")
    with gr.Blocks(theme=theme, css=css, title="Pet Dreambooth on Modal") as interface:
        gr.Markdown(
            f"# Dream up images of {instance_phrase}.\n\n{description}",
        )
        with gr.Row():
            inp = gr.Textbox(
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                # placeholder=example_prompts[0],
                lines=10,
            )
            out = gr.Image(height=512, width=512, label="", min_width=512, elem_id="output")
        with gr.Row():
            btn = gr.Button("Dream", variant="primary", scale=2)
            btn.click(fn=go, inputs=inp, outputs=out)

            gr.Button("⚡️ Powered by Modal", variant="secondary", link="https://modal.com")

        with gr.Column(variant="compact"):
            for ii, prompt in enumerate(example_prompts):
                    btn = gr.Button(prompt, variant="secondary")
                    btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# ## Running this on the command line
#
# You can use the `modal` command-line interface to interact with this code,
# in particular training the model and running the interactive Gradio service
#
# - `modal run dreambooth_app.py` will train the model
# - `modal serve dreambooth_app.py` will [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the Gradio interface at a temporarily location.
# - `modal shell dreambooth_app.py` is a convenient helper to open a bash [shell](https://modal.com/docs/guide/developing-debugging#stubinteractive_shell) in our image (for debugging)
#
# Remember, once you've trained your own fine-tuned model, you can deploy it using `modal deploy dreambooth_app.py`.
#
# This app is already deployed on Modal and you can try it out at https://modal-labs-example-dreambooth-app-fastapi-app.modal.run


@stub.local_entrypoint()
def run():
    with open(TrainConfig().instance_example_urls_file) as f:
        instance_example_urls = [line.strip() for line in f.readlines()]
    train.remote(instance_example_urls)
