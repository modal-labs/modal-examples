# # Running Diffusers example scripts on Modal

# The [Diffusers library](https://github.com/huggingface/diffusers) by HuggingFace provides a set of example training scripts that make it easy to experiment with various image fine-tuning techniques. This tutorial will show you how to run a Diffusers example script on Modal.

# ## Select training script

# You can see an up-to-date list of all the available examples in the [examples subdirectory](https://github.com/huggingface/diffusers/tree/main/examples). It includes, among others, examples for:

# - Dreambooth
# - Lora
# - Text-to-image
# - Fine-tuning Controlnet
# - Fine-tuning Kandinsky

# ## Set up the dependencies

# You can put all of the sample code on this page in a single file, for example, `train_and_serve_diffusers_script.py`. In all the code below, we will be using the [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) script as an example, but you should modify depending on which Diffusers script you are using.

# Start by specifying the Python modules that the training will depend on, including the Diffusers library, which contains the actual training script.

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from modal import (
    Image,
    Secret,
    Stub,
    Volume,
    asgi_app,
    method,
)

GIT_SHA = "ed616bd8a8740927770eebe017aedb6204c6105f"

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.19",
        "datasets~=2.13",
        "ftfy",
        "gradio~=3.10",
        "smart_open",
        "transformers==4.26.0",
        "safetensors==0.2.8",
        "torch",
        "torchvision",
        "triton",
    )
    .pip_install("xformers", pre=True)
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

# ## Set up `PersistentVolume`s for training data and model output

# Modal can't access your local filesystem, so you should set up a `PersistentVolume` to eventually save the model once training is finished.

web_app = FastAPI()
stub = Stub(name="example-diffusers-app")

MODEL_DIR = Path("/model")
stub.training_data_volume = Volume.persisted("training-data-volume")
stub.model_volume = Volume.persisted("output-model-volume")

VOLUME_CONFIG = {
    "/training_data": stub.training_data_volume,
    "/model": stub.model_volume,
}

# ## Set up config

# Each Diffusers example script takes a different set of hyperparameters, so you will need to customize the config depending on the hyperparameters of the script. The code below shows some example parameters.


@dataclass
class TrainConfig:
    """Configuration for the finetuning training."""

    # identifier for pretrained model on Hugging Face
    model_name: str = "runwayml/stable-diffusion-v1-5"

    resume_from_checkpoint: str = "latest"
    # HuggingFace Hub dataset
    dataset_name = "yirenlu/heroicons"

    # Hyperparameters/constants from some of the Diffusers examples
    # You should modify these to match the hyperparameters of the script you are using.
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-05
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 100
    checkpointing_steps: int = 2000
    mixed_precision: str = "fp16"
    caption_column: str = "text"
    max_grad_norm: int = 1
    validation_prompt: str = "an icon of a dragon creature"


@dataclass
class AppConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


# ## Set up finetuning dataset

# Each of the diffusers training scripts will utilize different argnames to refer to your input finetuning dataset. For example, it might be `--instance_data_dir` or `--dataset_name`. You will need to modify the code below to match the argname used by the training script you are using.
# Generally speaking, these argnames will correspond to either the name of a HuggingFace Hub dataset, or the path of a local directory containing your training dataset.
# This means that you should either upload your dataset to HuggingFace Hub, or push the dataset to a `PersistentVolume` and then attach that volume to the training function.

# ### Upload to HuggingFace Hub
# You can follow the instructions [here](https://huggingface.co/docs/datasets/upload_dataset#upload-with-python) to upload your dataset to the HuggingFace Hub.

# ### Push dataset to `PersistentVolume`
# To push your dataset to the `/training_data` volume you set up above, you can use [`modal volume put`](https://modal.com/docs/reference/cli/volume) command to push an entire local directory to a location in the volume.
# For example, if your dataset is located at `/path/to/dataset`, you can push it to the volume with the following command:
# ```bash
# modal volume put <volume-name> /path/to/dataset /training_data
# ```
# You can double check that the training data was properly uploaded to the volume by using `modal volume ls`:
# ```bash
# modal volume ls <volume-name> /training_data
# ```
# You should see the contents of your dataset listed in the output.

# ## Set up `stub.function` decorator on the training function.
# Next, let's write the `stub.function` decorator that will be used to launch the training function on Modal.
# The `@stub.function` decorator takes several arguments, including:
# - `image` - the Docker image that you want to use for training. In this case, we are using the `image` object that we defined above.
# - `gpu` - the type of GPU you want to use for training. This argument is optional, but if you don't specify a GPU, Modal will use a CPU.
# - `mounts` - the local directories that you want to mount to the Modal container. In this case, we are mounting the local directory where the training images reside.
# - `volumes` - the Modal volumes that you want to mount to the Modal container. In this case, we are mounting the `PersistentVolume` that we defined above.
# - `timeout` argument - an integer representing the number of seconds that the training job should run for. This argument is optional, but if you don't specify a timeout, Modal will use a default timeout of 300 seconds, or 5 minutes. The timeout argument has an upper limit of 24 hours.
# - `secrets` - the Modal secrets that you want to mount to the Modal container. In this case, we are mounting the HuggingFace API token secret.
@stub.function(
    image=image,
    gpu="A100",  # finetuning is VRAM hungry, so this should be an A100
    volumes=VOLUME_CONFIG,
    timeout=3600 * 2,  # multiple hours
    secrets=[Secret.from_name("huggingface")],
)
# ## Define the training function
# Now, finally, we define the training function itself. This training function does a bunch of preparatory things, but the core of it is the `_exec_subprocess` call to `accelerate launch` that launches the actual Diffusers training script. Depending on which Diffusers script you are using, you will want to modify the script name, and the arguments that are passed to it.
def train():

    import huggingface_hub
    from accelerate import notebook_launcher
    from accelerate.utils import write_basic_config
    from examples.text_to_image.train_text_to_image import main
    from transformers import CLIPTokenizer

    # set up TrainConfig
    config = TrainConfig()

    # set up runner-local image and shared model weight directories
    os.makedirs(MODEL_DIR, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # authenticate to hugging face so we can download the model weights
    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    # check whether we can access the model repo
    try:
        CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    except OSError as e:  # handle error raised when license is not accepted
        license_error_msg = f"Unable to load tokenizer. Access to this model requires acceptance of the license on Hugging Face here: https://huggingface.co/{config.model_name}."
        raise Exception(license_error_msg) from e

    def launch_training():

        sys.argv = [
            "examples/text_to_image/train_text_to_image.py",  # potentially modify
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--dataset_name={config.dataset_name}",
            "--use_ema",
            f"--output_dir={MODEL_DIR}",
            f"--resolution={config.resolution}",
            "--center_crop",
            "--random_flip",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            "--gradient_checkpointing",
            f"--train_batch_size={config.train_batch_size}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--max_train_steps={config.max_train_steps}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
        ]

        main()

    # run training -- see huggingface accelerate docs for details
    print("launching fine-tuning training script")

    notebook_launcher(launch_training, num_processes=1)
    # The trained model artefacts have been output to the volume mounted at `MODEL_DIR`.
    # To persist these artefacts for use in future inference function calls, we 'commit' the changes
    # to the volume.
    stub.model_volume.commit()


@stub.local_entrypoint()
def run():
    train.remote()


# ## Run training function

# To run this training function:

# ```bash
# modal run train_and_serve_diffusers_script.py
# ```

# ## Set up inference function

# Depending on which Diffusers training script you are using, you may need to use an alternative pipeline to `StableDiffusionPipeline`. The READMEs of the example training scripts will generally provide instructions for which inference pipeline to use. For example, if you are fine-tuning Kandinsky, it tells you to use [`AutoPipelineForText2Image`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky#diffusers.AutoPipelineForText2Image) instead of `StableDiffusionPipeline`.


@stub.cls(
    image=image,
    gpu="A100",
    volumes=VOLUME_CONFIG,  # mount the location where your model weights were saved to
)
class Model:
    def __enter__(self):
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        # Reload the modal.Volume to ensure the latest state is accessible.
        stub.volume.reload()

        # set up a hugging face inference pipeline using our model
        ddim = DDIMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
        # potentially use different pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            scheduler=ddim,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        self.pipe = pipe

    @method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


# ## Set up Gradio app

# Finally, we set up a Gradio app that will allow you to interact with your model. This Gradio app will be mounted to the Modal container, and will be accessible at the URL of your Modal deployment. You can refer to the [Gradio docs](https://www.gradio.app/docs/interface) for more information on how to customize a Gradio app.


@stub.function(
    image=image,
    concurrency_limit=3,
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call to the GPU inference function on Modal.
    def go(text):
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()

    HCON_prefix = "In the HCON style, an icon of"

    example_prompts = [
        f"{HCON_prefix} a movie ticket",
        f"{HCON_prefix} barack obama",
        f"{HCON_prefix} a castle",
        f"{HCON_prefix} a german shepherd",
    ]

    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/ex/diffusers"

    description = f"""Describe a concept that you would like drawn as a Heroicon icon. Try the examples below for inspiration.

### Learn how to make your own [here]({modal_example_url}).
    """

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title="Generate custom heroicons",
        examples=example_prompts,
        description=description,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# ## Run Gradio app

# Finally, we run the Gradio app. This will launch the Gradio app on Modal.

# ```bash
# modal serve train_and_serve_diffusers_script.py
# ```
