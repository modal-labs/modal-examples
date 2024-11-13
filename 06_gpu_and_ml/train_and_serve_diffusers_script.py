# # Running Diffusers example scripts on Modal
#
# The [Diffusers library](https://github.com/huggingface/diffusers) by HuggingFace provides a set of example training scripts that make it easy to experiment with various image fine-tuning techniques. This tutorial will show you how to run a Diffusers example script on Modal.
#
# ## Select training script
#
# You can see an up-to-date list of all the available examples in the [examples subdirectory](https://github.com/huggingface/diffusers/tree/main/examples). It includes, among others, examples for:
#
# - Dreambooth
# - Lora
# - Text-to-image
# - Fine-tuning Controlnet
# - Fine-tuning Kandinsky

# ## Fine-tuning on Heroicons
#
# In this tutorial, we'll cover fine-tuning Stable Diffusion on the [Heroicons](https://heroicons.com/) dataset to stylize icons, using the Diffusers [text-to-image](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) script. Heroicons are website icons developed by the maker of TailwindCSS. They are open source, but there are only ~300 of them representing common concepts. What if you want icons depicting other concepts not covered by the original 300? Generative AI makes this possible - by using text-to-image models, you can just input your target concept and get a Heroicon of it back!

# | | | |
# | --- | --- | --- |
# | ![film.png](./film.png) | ![users.png](./users.png) | ![calendar-days.png](./calendar-days.png) |

# ## Fine-tuning results

# Here are some of the results of the fine-tuning. As you can see, it's not perfect - the model sometimes outputs multiple objects when prompted for one, and the outputs aren't always sufficiently abstract/simple.
# But it's very cool that the model is able to visualize even abstract concepts like "international monetary system" in the Heroicon style, and come up with an icon that actually makes sense.
# You can play around with the fine-tuned model yourself [here](https://yirenlu92--example-text-to-image-no-lora-app-fastapi-app.modal.run/).

# | | | | |
# | --- | --- | --- | --- |
# | ![fine-tuned results](./heroicon_camera.png) | ![fine-tuned results](./heroicon_golden_retriever.png)  | ![fine-tuned results](./heroicon_piano.png) | ![fine-tuned results](./heroicon_ebike.png) |
# | *In the HCON style, an icon of a camera* | *In the HCON style, an icon of a golden retriever* | *In the HCON style, an icon of a baby grand piano* | *In the HCON style, an icon of a single ebike* |
# | ![fine-tuned results](./heroicon_barack_obama.png)  | ![fine-tuned results](./heroicon_bmw.png) | ![fine-tuned results](./heroicon_castle.png)  | ![fine-tuned results](./heroicon_fountain_pen.png) |
# | *In the HCON style, an icon of barack obama's head* | *In the HCON style, an icon of a BMW X5, from the front. Please show the entire car.* | *In the HCON style, an icon of a castle* | *In the HCON style, an icon of a single fountain pen* |
# | ![fine-tuned results](./heroicon_apple_computer.png)  | ![fine-tuned results](./heroicon_library.png)  | ![fine-tuned results](./heroicon_snowflake.png)  | ![fine-tuned results](./heroicon_snowman.png)  |
# | *In the HCON style, an icon of a macbook pro computer* | *In the HCON style, an icon of the interior of a library* | *In the HCON style, an icon of a snowflake* | *In the HCON style, an icon of a snowman* |
# | ![fine-tuned results](./heroicon_german_shepherd.png)  | ![fine-tuned results](./heroicon_water_bottle.png)  | ![fine-tuned results](./heroicon_jail_cell.png)  | ![fine-tuned results](./heroicon_travel.png)  |
# | *In the HCON style, an icon of a german shepherd* | *In the HCON style, an icon of a water bottle* | *In the HCON style, an icon representing a jail cell* | *In the HCON style, an icon representing travel* |
# | ![fine-tuned results](./heroicon_future_of_AI.png) | ![fine-tuned results](./heroicon_skiing.png) | ![fine-tuned results](./heroicon_international_monetary_system.png) | ![fine-tuned results](./heroicon_chemistry.png)  |
# | *In the HCON style, an icon that represents the future of AI* | *In the HCON style, an icon representing skiing* | *In the HCON style, an icon representing the international monetary system* | *In the HCON style, an icon representing chemistry* |

# ## Creating the dataset

# Most tutorials skip over dataset creation, but since we're training on a novel dataset, we'll cover the full process.

# 1. Download all the Heroicons from the Heroicons [Github repo](https://github.com/tailwindlabs/heroicons)

# ```bash
# git clone git@github.com:tailwindlabs/heroicons.git
# cd optimized/24/outline # we are using the optimized, outline icons. You can also try using the solid icons
# ```

# 2. Postprocess the SVGs

# *Convert SVGs to PNGs*

# Most training models are unable to process SVGs, so we convert them first to PNGs.

# *Add white backgrounds to the PNGs*

# We also need to add white backgrounds to the PNGs. This is important - transparent backgrounds really confuse the model.

# ```python
# def add_white_background(input_path, output_path):
#     # Open the image
#     img = Image.open(input_path)

#     # Ensure the image has an alpha channel for transparency
#     img = img.convert('RGBA')

#     # Create a white background image
#     bg = Image.new('RGBA', img.size, (255, 255, 255))

#     # Combine the images
#     combined = Image.alpha_composite(bg, img)

#     # Save the result
#     combined.save(output_path, "PNG")
# ```

# 3. Add captions to create a `metadata.csv` file.

# Since the Heroicon filenames match the concept they represent, we can parse them into captions. We also add a prefix to each caption: `“In the HCON style, an icon of an <object>.”` The purpose of this prefix is to associate a rare keyword, `HCON`to the particular Heroicon style.

# We then create a `metadata.csv` file, where each row is an image file name with the associated caption. The `metadata.csv` file should be placed in the same directory as all the training images images, and contain a header row with the string `file_name,text`

# ```python

# heroicons_training_dir/
# 		arrow.png
# 		bike.png
# 		cruiseShip.png
# 		metadata.csv
# ```

# ```
# file_name,text
# arrow.png,"In the HCON style, an icon of an arrow"
# bike.png,"In the HCON style, an icon of an arrow"
# cruiseShip.png,"In the HCON style, an icon of an arrow"
# ```

# 4. Upload the dataset to HuggingFace Hub.

# This converts the dataset into an optimized Parquet file.

# ```python
# import os
# from datasets import load_dataset
# import huggingface_hub

# # login to huggingface
# hf_key = os.environ["HUGGINGFACE_TOKEN"]
# huggingface_hub.login(hf_key)

# dataset = load_dataset("imagefolder", data_dir="/lg_white_bg_heroicon_png_img", split="train")

# dataset.push_to_hub("yirenlu/heroicons", private=True)
# ```

# The final Heroicons dataset on HuggingFace Hub is [here](https://huggingface.co/datasets/yirenlu/heroicons).

#
# ## Set up the dependencies for fine-tuning on Modal
#
# You can put all of the sample code that follows in a single file, for example, `train_and_serve_diffusers_script.py`. In all the code below, we will be using the [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) script as an example, but you should modify depending on which Diffusers script that makes sense for your use case.
#
# Start by specifying the Python modules that the training will depend on, including the Diffusers library, which contains the actual training script.

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from modal import App, Image, Secret, Volume, asgi_app, enter, method

GIT_SHA = (
    "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch
)

image = Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.31.0",
    "datasets~=2.13.0",
    "ftfy~=6.1.0",
    "gradio~=4.29.0",
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
    "smart_open~=6.4.0",
    "transformers~=4.41.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "peft==0.11.1",
    "wandb==0.17.6",
)

image = (
    image.apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)


# ## Set up `Volume`s for training data and model output
#
# Modal can't access your local filesystem, so you should set up a `Volume` to eventually save the model once training is finished.

web_app = FastAPI()
app = App(name="example-diffusers-app")

MODEL_DIR = Path("/model")
training_data_volume = Volume.from_name(
    "training-data-volume", create_if_missing=True
)
model_volume = Volume.from_name("output-model-volume", create_if_missing=True)

VOLUME_CONFIG = {
    "/training_data": training_data_volume,
    "/model": model_volume,
}

USE_WANDB = True


# ## Set up config
#
# Each Diffusers example script takes a different set of hyperparameters, so you will need to customize the config depending on the hyperparameters of the script. The code below shows some example parameters.


@dataclass
class TrainConfig:
    """Configuration for the finetuning training."""

    # identifier for pretrained model on Hugging Face
    model_name: str = "runwayml/stable-diffusion-v1-5"

    resume_from_checkpoint: str = "latest"
    # HuggingFace Hub dataset
    dataset_name = "yirenlu/heroicons-subset-10"

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
#
# Each of the diffusers training scripts will utilize different argnames to refer to your input finetuning dataset. For example, it might be `--instance_data_dir` or `--dataset_name`. You will need to modify the code below to match the argname used by the training script you are using.
# Generally speaking, these argnames will correspond to either the name of a HuggingFace Hub dataset, or the path of a local directory containing your training dataset.
# This means that you should either upload your dataset to HuggingFace Hub, or push the dataset to a `Volume` and then attach that volume to the training function.
#
# ### Upload to HuggingFace Hub
# You can follow the instructions [here](https://huggingface.co/docs/datasets/upload_dataset#upload-with-python) to upload your dataset to the HuggingFace Hub.
#
# ### Push dataset to `Volume`
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
#
# ## Set up `app.function` decorator on the training function.
# Next, let's write the `app.function` decorator that will be used to launch the training function on Modal.
# The `@app.function` decorator takes several arguments, including:
# - `image` - the Docker image that you want to use for training. In this case, we are using the `image` object that we defined above.
# - `gpu` - the type of GPU you want to use for training. This argument is optional, but if you don't specify a GPU, Modal will use a CPU.
# - `mounts` - the local directories that you want to mount to the Modal container. In this case, we are mounting the local directory where the training images reside.
# - `volumes` - the Modal volumes that you want to mount to the Modal container. In this case, we are mounting the `Volume` that we defined above.
# - `timeout` argument - an integer representing the number of seconds that the training job should run for. This argument is optional, but if you don't specify a timeout, Modal will use a default timeout of 300 seconds, or 5 minutes. The timeout argument has an upper limit of 24 hours.
# - `secrets` - the Modal secrets that you want to mount to the Modal container. In this case, we are mounting the HuggingFace API token secret.
@app.function(
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
    from accelerate.utils import write_basic_config

    # change this line to import the training script you want to use
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

        # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate

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
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--dataset_name={config.dataset_name}",
            # f"--instance_data_dir={img_path}",
            f"--output_dir={MODEL_DIR}",
            "--instance_prompt='heroicon in the style of TOK'",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )
    # The trained model information has been output to the volume mounted at `MODEL_DIR`.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()

    # def launch_training():
    #     sys.argv = [
    #         "examples/text_to_image/train_text_to_image.py",  # potentially modify
    #         f"--pretrained_model_name_or_path={config.model_name}",
    #         f"--dataset_name={config.dataset_name}",
    #         "--use_ema",
    #         f"--output_dir={MODEL_DIR}",
    #         f"--resolution={config.resolution}",
    #         "--center_crop",
    #         "--random_flip",
    #         f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
    #         "--gradient_checkpointing",
    #         f"--train_batch_size={config.train_batch_size}",
    #         f"--learning_rate={config.learning_rate}",
    #         f"--lr_scheduler={config.lr_scheduler}",
    #         f"--max_train_steps={config.max_train_steps}",
    #         f"--lr_warmup_steps={config.lr_warmup_steps}",
    #         f"--checkpointing_steps={config.checkpointing_steps}",
    #     ]

    #     main()

    # # run training -- see huggingface accelerate docs for details
    # print("launching fine-tuning training script")

    # notebook_launcher(launch_training, num_processes=1)
    # # The trained model artefacts have been output to the volume mounted at `MODEL_DIR`.
    # # To persist these artefacts for use in future inference function calls, we 'commit' the changes
    # # to the volume.
    # app.model_volume.commit()


@app.local_entrypoint()
def run():
    train.remote()


# ## Run training function
#
# To run this training function:
#
# ```bash
# modal run train_and_serve_diffusers_script.py
# ```
#
# ## Set up inference function
#
# Depending on which Diffusers training script you are using, you may need to use an alternative pipeline to `StableDiffusionPipeline`. The READMEs of the example training scripts will generally provide instructions for which inference pipeline to use. For example, if you are fine-tuning Kandinsky, it tells you to use [`AutoPipelineForText2Image`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky#diffusers.AutoPipelineForText2Image) instead of `StableDiffusionPipeline`.


@app.cls(
    image=image,
    gpu="A100",
    volumes=VOLUME_CONFIG,  # mount the location where your model weights were saved to
)
class Model:
    @enter()
    def setup(self):
        import torch
        from diffusers import DiffusionPipeline

        config = TrainConfig()

        # Reload the modal.Volume to ensure the latest state is accessible.
        app.volume.reload()
        # potentially use different pipeline
        pipe = DiffusionPipeline.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
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
#
# Finally, we set up a Gradio app that will allow you to interact with your model. This will be mounted to the Modal container, and will be accessible at the URL of your Modal deployment. You can refer to the [Gradio docs](https://www.gradio.app/docs/interface) for more information on how to customize the app.


@app.function(
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
#
# Finally, we run the Gradio app. This will launch the Gradio app on Modal.
#
# ```bash
# modal serve train_and_serve_diffusers_script.py
# ```
