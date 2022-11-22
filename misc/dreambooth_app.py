# ---
# args: ["train"]
# deploy: true
# integration-test: false
# lambda-test: false
# ---
#
# # Pet Art Dreambooth with Hugging Face and Gradio
#
# This example finetunes the [Stable Diffusion v1.4 model](https://huggingface.co/CompVis/stable-diffusion-v1-4?text=pikachu-at-the-offices-of-a-unicorn-mlops-startup)
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
# GPU-accelerated Modal Functions, shared volumes for caching, and Modal webhooks.
#
# And with some light customization, you can use it to generate images of your pet!
#
# ![Gradio.app image generation interface](./gradio-image-generate.png)
#
# ## Setting up the dependencies
#
# We can start from a slim Debian OS image and install all of our dependencies
# with the `pip` Python package installer.
import os
from pathlib import Path
import sys

from dataclasses import dataclass
from fastapi import FastAPI
import modal

web_app = FastAPI()
assets_path = Path(__file__).parent / "dreambooth_app" / "assets"
stub = modal.Stub(name="example-dreambooth-app")


image = modal.Image.debian_slim().pip_install(
    [
        "diffusers~=0.5",
        "accelerate~=0.14.0",
        "torchvision~=0.14",
        "transformers~=4.21",
        "ftfy",
        "tensorboard",
        "smart_open~=6.2.0",
        "gradio~=3.10",
    ]
)

# A persistent shared volume will store model artefacts across Modal app runs.
# This is crucial as finetuning runs are separate from the Gradio app we run as a webhook.

volume = modal.SharedVolume().persist(f"dreambooth-finetuning-vol")
MODEL_DIR = Path(f"/model")

# Finetuning Stable Diffusion at 16-bit precision requires a lot of VRAM,
# so we request a beefy NVIDIA A100 GPU.
gpu = modal.gpu.A100()

# ## Config
#
# All configs get their own dataclasses to avoid scattering special/magic values throughout code.
# You can read more about how the values in `TrainConfig` are chosen and adjusted [in this blog post on Hugging Face](https://huggingface.co/blog/dreambooth).


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # url of plaintext file with urls for images of target instance
    images_file_url: str = (
        "https://raw.githubusercontent.com/modal-labs/modal-examples/main/misc/dreambooth_app/data/qwerty_images.txt"
    )

    # identifier for pretrained model on Hugging Face
    model_name: str = "CompVis/stable-diffusion-v1-4"

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 400


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


# ## Get finetuning dataset
#
# Part of the magic of Dreambooth is that we only need 4-10 images for finetuning.
# So we can fetch just a few images, stored on consumer platforms like Imgur or Google Drive
# -- no need for expensive data collection or data engineering.

IMG_PATH = Path("/img")


def load_images(path):
    from smart_open import open
    import PIL.Image

    print("Loading images.")
    with open(path) as f:
        lines = f.readlines()

    image_urls = [line.strip() for line in lines]

    os.makedirs(IMG_PATH, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(IMG_PATH / f"{ii}.png")
    print("Images loaded.")

    return IMG_PATH


# ## Finetuning a text-to-image model
#
# This model is trained to do a sort of "reverse [ekphrasis](https://en.wikipedia.org/wiki/Ekphrasis)":
# it attempts to recreate a visual work of art or image from only its description.
#
# We can use a trained model to synthesize wholly new images
# by combining the concepts it has learned from the training data.
#
# We use a pretrained model, version 1.4 of the Stable Diffusion model,
# trained by the [Ommer Lab](https://ommer-lab.com/) at LMU Munich.
#
# In this example, we "finetune" SD v1.4, making only small adjustments to the weights,
# in order to just teach it a new word: the name of our pet.
#
# The result is a model that can generate novel images of our pet:
# as an astronaut in space, as painted by Van Gogh or Bastiat, etc.
#
# ### Finetuning with Hugging Face 🧨 Diffusers and Accelerate
#
# The model weights, libraries, and training script are all provided by [🤗 Hugging Face](https://huggingface.co).
#
# To access the model weights, you'll need a [Hugging Face account](https://huggingface.co/join)
# and from that account you'll need to accept the model license [here](https://huggingface.co/CompVis/stable-diffusion-v1-4).
#
# Lastly, you'll need to create a token from that account and share it with Modal
# under the name `"huggingface"`. Follow the instructions [here](https://modal.com/secrets).
#
# Then, you can kick off a training job with
# `modal app run dreambooth_app.py --function-name train`
# It should take about five minutes.


@stub.function(
    image=image,
    gpu=gpu,  # finetuning is VRAM hungry, so this should be an A100
    cpu=8,  # request enough CPUs to feed the GPU
    shared_volumes={
        str(MODEL_DIR): volume,
    },
    timeout=600,  # 10 minutes
    secrets=[modal.Secret.from_name("huggingface")],
    interactive=True,
)
def train(config=TrainConfig()):
    import subprocess

    from accelerate.utils import write_basic_config
    import huggingface_hub
    from smart_open import open
    from transformers import CLIPTokenizer

    # set up runner-local image and shared model weight directories
    img_path = load_images(config.images_file_url)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # authenticate to hugging face so we can download the model weights
    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    # check whether we can access to model repo
    try:
        CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    except OSError as e:  # handle error raised when license is not accepted
        license_error_msg = f"Unable to load tokenizer. Access to this model requires acceptance of the license on Hugging Face here: https://huggingface.co/{config.model_name}."
        raise Exception(license_error_msg) from e

    # fetch the training script from Hugging Face's GitHub repo
    raw_repo_url = "https://raw.githubusercontent.com/huggingface/diffusers"
    script_commit_hash = "30220905c4319e46e114cf7dc8047d94eca226f7"
    script_path = "examples/dreambooth/train_dreambooth.py"
    script_url = f"{raw_repo_url}/{script_commit_hash}/{script_path}"

    with open(script_url) as from_file:
        script_content = from_file.readlines()
        with open("train_dreambooth.py", "w") as to_file:
            to_file.writelines(script_content)

    # define the training prompt
    instance_phrase = f"{config.instance_name} {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    # run training -- see huggingface accelerate docs for details
    subprocess.run(
        [
            "accelerate",
            "launch",
            "train_dreambooth.py",
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt='{prompt}'",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
        ],
        check=True,
    )


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


@stub.asgi(
    image=image,
    gpu=gpu,
    cpu=1,  # during inference, CPU is less of a bottleneck
    shared_volumes={str(MODEL_DIR): volume},
    mounts=[modal.Mount("/assets", local_dir=assets_path)],
)
def fastapi_app(config=AppConfig()):
    import os

    from diffusers import StableDiffusionPipeline
    import gradio as gr
    from gradio.routes import mount_gradio_app
    import torch

    # set up a hugging face inference pipeline using our model
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")

    # wrap inference in a text-to-image function
    def go(text):
        image = pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"{instance_phrase} flying through space as an astronaut",
        f"low polygon count art of {instance_phrase} from the Nintendo 64 game {instance_phrase.title()} 64",
    ]

    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/ex/dreambooth-app"

    description = f"""Describe what they are doing or pick an artist. Be fantastical! Try the examples below.",

    ### Learn how to make your own [here]({modal_example_url}).
    """

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title=f"Generate images of {instance_phrase}.",
        description="Describe what they are doing or pick an artist. Be fantastical! Try the examples below.",
        examples=example_prompts,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) >= 2 else "train"
    if cmd == "train":
        with stub.run():
            train()
    elif cmd == "serve":
        stub.serve()
    elif cmd == "shell":
        stub.interactive_shell()
    else:
        print(f"Invalid cmd '{cmd}'.")

# This app is already deployed on Modal and you can try it out at https://modal-labs-example-dreambooth-app.modal.run.
