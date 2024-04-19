# ---
# deploy: true
# lambda-test: false
# ---
#
# # FastAI model training with Weights & Biases and Gradio
#
# This example trains a vision model to 98-99% accuracy on the CIFAR-10 dataset,
# and then makes this trained model shareable with others using the [Gradio.app](https://gradio.app/)
# web interface framework (Huggingface's competitor to Streamlit).
#
# Combining GPU-accelerated Modal Functions, a network file system for caching, and Modal
# webhooks for the model demo, we have a simple, productive, and cost-effective
# pathway to building and deploying ML in the cloud!
#
# ![Gradio.app image classifer interface](./gradio-image-classify.png)
#
# ## Setting up the dependencies
#
# Our GPU training is done with PyTorch which bundles its CUDA dependencies, so
# we can start with a slim Debian OS image and install a handful of `pip` packages into it.

import dataclasses
import os
import pathlib
import sys
from typing import List, Optional, Tuple

from fastapi import FastAPI
from modal import (
    App,
    Image,
    Mount,
    Secret,
    Volume,
    asgi_app,
    enter,
    method,
)

web_app = FastAPI()
assets_path = pathlib.Path(__file__).parent / "vision_model_training" / "assets"
app = App(name="example-fastai-wandb-gradio-cifar10-demo")
image = Image.debian_slim(python_version="3.10").pip_install(
    "fastai~=2.7.9",
    "gradio~=3.6.0",
    "httpx~=0.23.0",
    # When using pip PyTorch is not automatically installed by fastai.
    "torch~=1.12.1",
    "torchvision~=0.13.1",
    "wandb~=0.13.4",
)

# A persisted volume will store trained model artefacts across Modal app runs.
# This is crucial as training runs are separate from the Gradio.app we run as a webhook.

volume = Volume.from_name("cifar10-training-vol", create_if_missing=True)

FASTAI_HOME = "/fastai_home"
MODEL_CACHE = pathlib.Path(FASTAI_HOME, "models")
USE_GPU = os.environ.get("MODAL_GPU")
MODEL_EXPORT_PATH = pathlib.Path(MODEL_CACHE, "model-exports", "inference.pkl")
os.environ[
    "FASTAI_HOME"
] = FASTAI_HOME  # Ensure fastai saves data into persistent volume path.

# ## Config
#
# Training config gets its own dataclass to avoid smattering special/magic values throughout code.


@dataclasses.dataclass
class WandBConfig:
    project: str = "fast-cifar10"
    entity: Optional[str] = None


@dataclasses.dataclass
class Config:
    epochs: int = 10
    img_dims: Tuple[int, int] = (32, 224)
    gpu: str = USE_GPU
    wandb: WandBConfig = dataclasses.field(default_factory=WandBConfig)


# ## Get CIFAR-10 dataset
#
# The `fastai` framework famously requires very little code to get things done,
# so our downloading function is very short and simple. The CIFAR-10 dataset is
# also not large, about 150MB, so we don't bother persisting it in a network file system
# and just download and unpack it to ephemeral disk.


def download_dataset():
    from fastai.data.external import URLs, untar_data

    path = untar_data(URLs.CIFAR)
    print(f"Finished downloading and unpacking CIFAR-10 dataset to {path}.")
    return path


# ## Training a vision model with FastAI
#
# To address the CIFAR-10 image classification problem, we use the high-level fastAI framework
# to train a Deep Residual Network (https://arxiv.org/pdf/1512.03385.pdf) with 18-layers, called `resnet18`.
#
# We don't train the model from scratch. A transfer learning approach is sufficient to reach 98-99%
# accuracy on the classification task. The main tweak we make is to adjust the image size of the CIFAR-10
# examples to be 224x224, which was the image size the `resnet18` model was originally trained on.
#
# Just to demonstrate the affect of the image upscaling on classification performance, we still train on
# the original size 32x32 images.
#
# ### Tracking with Weights & Biases
#
# ![weights & biases UI](./wandb-ui.png)
#
# Weights & Biases (W&B) is a product that provides out-of-the-box model training observability. With a few
# lines of code and an account, we gain a dashboard will key metrics such as training loss, accuracy, and GPU
# utilization.
#
# If you want to run this example without setting up Weights & Biases, just remove the
# `secrets=[Secret.from_name("wandb")]` line from the Function decorator below; this will disable Weights & Biases
# functionality.
#
# ### Detaching our training run
#
# Fine-tuning the base ResNet model takes about 30-40 minutes on a GPU. To avoid
# needing to keep our terminal active, we can run training as a 'detached run'.
#
# `MODAL_GPU=any modal run --detach vision_model_training.py::app.train`
#


@app.function(
    image=image,
    gpu=USE_GPU,
    volumes={str(MODEL_CACHE): volume},
    secrets=[Secret.from_name("my-wandb-secret")],
    timeout=2700,  # 45 minutes
)
def train():
    import wandb
    from fastai.callback.wandb import WandbCallback
    from fastai.data.transforms import parent_label
    from fastai.metrics import accuracy
    from fastai.vision.all import Resize, models, vision_learner
    from fastai.vision.data import (
        CategoryBlock,
        DataBlock,
        ImageBlock,
        TensorCategory,
        get_image_files,
    )

    config: Config = Config()

    print("Downloading dataset")
    dataset_path = download_dataset()

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        wandb.init(
            id=app.app_id,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        callbacks = WandbCallback()
    else:
        callbacks = None

    for dim in config.img_dims:
        print(f"Training on {dim}x{dim} size images.")
        dblock = DataBlock(
            blocks=(ImageBlock(), CategoryBlock()),
            get_items=get_image_files,
            get_y=parent_label,
            item_tfms=Resize(dim),
        )

        dls = dblock.dataloaders(dataset_path, bs=64)

        learn = vision_learner(
            dls, models.resnet18, metrics=accuracy, cbs=callbacks
        ).to_fp16()
        learn.fine_tune(config.epochs, freeze_epochs=3)
        learn.save(f"cifar10_{dim}")

        # run on test set
        test_files = get_image_files(dataset_path / "test")
        label = TensorCategory(
            [dls.vocab.o2i[parent_label(f)] for f in test_files]
        )

        test_set = learn.dls.test_dl(test_files)
        pred = learn.get_preds(0, test_set)
        acc = accuracy(pred[0], label).item()
        print(f"{dim}x{dim}, test accuracy={acc}")

    # 🐝 Close wandb run
    if wandb_enabled:
        wandb.finish()

    print("Exporting model for later inference.")
    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)
    learn.remove_cbs(
        WandbCallback
    )  # Added W&B callback is not compatible with inference.
    learn.export(MODEL_EXPORT_PATH)
    volume.commit()


# ## Trained model plumbing
#
# To load a trained model into the demo app, we write a small
# amount of harness code that loads the saved model from persistent
# disk once on container start.
#
# The model's predict function accepts an image as bytes or a numpy array.


@app.cls(
    image=image,
    volumes={str(MODEL_CACHE): volume},
)
class ClassifierModel:
    @enter()
    def load_model(self):
        from fastai.learner import load_learner

        self.model = load_learner(MODEL_EXPORT_PATH)

    @method()
    def predict(self, image) -> str:
        prediction = self.model.predict(image)
        classification = prediction[0]
        return classification


@app.function(
    image=image,
)
def classify_url(image_url: str) -> None:
    """Utility function for command-line classification runs."""
    import httpx

    r = httpx.get(image_url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download '{image_url}'")

    classifier = ClassifierModel()
    label = classifier.predict.remote(image=r.content)
    print(f"Classification: {label}")


# ## Wrap the trained model in Gradio's web UI
#
# Gradio.app makes it super easy to expose a model's functionality
# in an intuitive web interface.
#
# This model is an image classifier, so we set up an interface that
# accepts an image via drag-and-drop and uses the trained model to
# output a classification label.
#
# Remember, this model was trained on tiny CIFAR-10 images so it's
# going to perform best against similarly simple and scaled-down images.


def create_demo_examples() -> List[str]:
    # NB: Don't download these images to a network FS as it doesn't play well with Gradio.
    import httpx

    example_imgs = {
        "lion.jpg": "https://i0.wp.com/lioncenter.umn.edu/wp-content/uploads/2018/10/cropped-DSC4884_Lion_Hildur-1.jpg",
        "mouse.jpg": "https://static-s.aa-cdn.net/img/ios/1077591533/18f74754ae55ee78e96e04d14e8bff35?v=1",
        "plane.jpg": "https://x-plane.hu/L-410/img/about/2.png",
        "modal.jpg": "https://pbs.twimg.com/profile_images/1567270019075031040/Hnrebn0M_400x400.jpg",
    }
    available_examples = []
    for dest, url in example_imgs.items():
        filepath = pathlib.Path(dest)
        r = httpx.get(url)
        if r.status_code != 200:
            print(f"Could not download '{url}'", file=sys.stderr)
            continue

        with open(filepath, "wb") as f:
            f.write(r.content)
        available_examples.append(str(filepath))
    return available_examples


@app.function(
    image=image,
    volumes={str(MODEL_CACHE): volume},
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    classifier = ClassifierModel()
    interface = gr.Interface(
        fn=classifier.predict.remote,
        inputs=gr.Image(shape=(224, 224)),
        outputs="label",
        examples=create_demo_examples(),
        css="/assets/index.css",
    )
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


## Running this
#
# To run training as an ephemeral app:
#
# ```shell
# modal run vision_model_training.py::app.train
# ```
#
# To test the model on an image, run:
#
# ```shell
# modal run vision_model_training.py::app.classify_url --image-url <url>
# ```
#
# To run the Gradio server, run:
#
# ```shell
# modal serve vision_model_training.py
# ```
#
# This ML app is already deployed on Modal and you can try it out at https://modal-labs-example-fastai-wandb-gradio-cifar10-demo-fastapi-app.modal.run.
