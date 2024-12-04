# ---
# args: ["--no-quick-check"]
# ---

# # Fine-tune open source YOLO models for object detection

# Example by [@Erik-Dunteman](https://github.com/erik-dunteman) and [@AnirudhRahul](https://github.com/AnirudhRahul/).

# The popular "You Only Look Once" (YOLO) model line provides high-quality object detection in an economical package.
# In this example, we use the [YOLOv10](https://docs.ultralytics.com/models/yolov10/) model, released on May 23, 2024.

# We will:

# - Download two custom datasets from the [Roboflow](https://roboflow.com/) computer vision platform: a dataset of birds and a dataset of bees

# - Fine-tune the model on those datasets, in parallel, using the [Ultralytics package](https://docs.ultralytics.com/)

# - Run inference with the fine-tuned models on single images and on streaming frames

# For commercial use, be sure to consult the [Ultralytics software license options](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed),
# which include AGPL-3.0.

# ## Set up the environment

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal

# Modal runs your code in the cloud inside containers. So to use it, we have to define the dependencies
# of our code as part of the container's [image](https://modal.com/docs/guide/custom-container).

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(  # install system libraries for graphics handling
        ["libgl1-mesa-glx", "libglib2.0-0"]
    )
    .pip_install(  # install python libraries for computer vision
        ["ultralytics~=8.2.68", "roboflow~=1.1.37", "opencv-python~=4.10.0"]
    )
    .pip_install(  # add an optional extra that renders images in the terminal
        "term-image==0.7.1"
    )
)

# We also create a persistent [Volume](https://modal.com/docs/guide/volumes) for storing datasets, trained weights, and inference outputs.

volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)
volume_path = (  # the path to the volume from within the container
    Path("/root") / "data"
)

# We attach both of these to a Modal [App](https://modal.com/docs/guide/apps).
app = modal.App("yolo-finetune", image=image, volumes={volume_path: volume})


# ## Download a dataset

# We'll be downloading our data from the [Roboflow](https://roboflow.com/) computer vision platform, so to follow along you'll need to:

# - Create a free account on [Roboflow](https://app.roboflow.com/)

# - [Generate a Private API key](https://app.roboflow.com/settings/api)

# - Set up a Modal [Secret](https://modal.com/docs/guide/secrets) called `roboflow-api-key` in the Modal UI [here](https://modal.com/secrets),
# setting the `ROBOFLOW_API_KEY` to the value of your API key.

# You're also free to bring your own dataset with a config in YOLOv10-compatible yaml format.

# We'll be training on the medium size model, but you're free to experiment with [other model sizes](https://docs.ultralytics.com/models/yolov10/#model-variants).


@dataclass
class DatasetConfig:
    """Information required to download a dataset from Roboflow."""

    workspace_id: str
    project_id: str
    version: int
    format: str
    target_class: str

    @property
    def id(self) -> str:
        return f"{self.workspace_id}/{self.project_id}/{self.version}"


@app.function(
    secrets=[
        modal.Secret.from_name(
            "roboflow-api-key", required_keys=["ROBOFLOW_API_KEY"]
        )
    ]
)
def download_dataset(config: DatasetConfig):
    import os

    from roboflow import Roboflow

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = (
        rf.workspace(config.workspace_id)
        .project(config.project_id)
        .version(config.version)
    )
    dataset_dir = volume_path / "dataset" / config.id
    project.download(config.format, location=str(dataset_dir))


# ## Train a model

# We train the model on a single A100 GPU. Training usually takes only a few minutes.

MINUTES = 60

TRAIN_GPU_COUNT = 1
TRAIN_GPU = modal.gpu.A100(count=TRAIN_GPU_COUNT)
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout=60 * MINUTES,
)
def train(
    model_id: str,
    dataset: DatasetConfig,
    model_size="yolov10m.pt",
    quick_check=False,
):
    from ultralytics import YOLO

    volume.reload()  # make sure volume is synced

    model_path = volume_path / "runs" / model_id
    model_path.mkdir(parents=True, exist_ok=True)

    data_path = volume_path / "dataset" / dataset.id / "data.yaml"

    model = YOLO(model_size)
    model.train(
        # dataset config
        data=data_path,
        fraction=0.4
        if not quick_check
        else 0.04,  # fraction of dataset to use for training/validation
        # optimization config
        device=list(range(TRAIN_GPU_COUNT)),  # use the GPU(s)
        epochs=8
        if not quick_check
        else 1,  # pass over entire dataset this many times
        batch=0.95,  # automatic batch size to target fraction of GPU util
        seed=117,  # set seed for reproducibility
        # data processing config
        workers=max(
            TRAIN_CPU_COUNT // TRAIN_GPU_COUNT, 1
        ),  # split CPUs evenly across GPUs
        cache=False,  # cache preprocessed images in RAM?
        # model saving config
        project=f"{volume_path}/runs",
        name=model_id,
        exist_ok=True,  # overwrite previous model if it exists
        verbose=True,  # detailed logs
    )


# ## Run inference on single inputs and on streams

# We demonstrate two different ways to run inference -- on single images and on a stream of images.

# The images we use for inference are loaded from the test set, which was added to our Volume when we downloaded the dataset.
# Each image read takes ~50ms, and inference can take ~5ms, so the disk read would be our biggest bottleneck if we just looped over the image paths.
# To avoid it, we parallelize the disk reads across many workers using Modal's [`.map`](https://modal.com/docs/guide/scale),
# streaming the images to the model. This roughly mimics the behavior of an interactive object detection pipeline.
# This can increase throughput up to ~60 images/s, or ~17 milliseconds/image, depending on image size.


@app.function()
def read_image(image_path: str):
    import cv2

    source = cv2.imread(image_path)
    return source


# We use the `@enter` feature of [`modal.Cls`](https://modal.com/docs/guide/lifecycle-functions)
# to load the model only once on container start and reuse it for future inferences.
# We use a generator to stream images to the model.


@app.cls(gpu="a10g")
class Inference:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)

    @modal.method()
    def predict(self, model_id: str, image_path: str, display: bool = False):
        """A simple method for running inference on one image at a time."""
        results = self.model.predict(
            image_path,
            half=True,  # use fp16
            save=True,
            exist_ok=True,
            project=f"{volume_path}/predictions/{model_id}",
        )
        if display:
            from term_image.image import from_file

            terminal_image = from_file(results[0].path)
            terminal_image.draw()
        # you can view the output file via the Volumes UI in the Modal dashboard -- https://modal.com/storage

    @modal.method()
    def streaming_count(self, batch_dir: str, threshold: float | None = None):
        """Counts the number of objects in a directory of images.

        Intended as a demonstration of high-throughput streaming inference."""
        import os
        import time

        image_files = [
            os.path.join(batch_dir, f) for f in os.listdir(batch_dir)
        ]

        completed, start = 0, time.monotonic_ns()
        for image in read_image.map(image_files):
            # note that we run predict on a single input at a time.
            # each individual inference is usually done before the next image arrives, so there's no throughput benefit to batching.
            results = self.model.predict(
                image,
                half=True,  # use fp16
                save=False,  # don't save to disk, as it slows down the pipeline significantly
                verbose=False,
            )
            completed += 1
            for res in results:
                for conf in res.boxes.conf:
                    if threshold is None:
                        yield 1
                        continue
                    if conf.item() >= threshold:
                        yield 1
            yield 0

        elapsed_seconds = (time.monotonic_ns() - start) / 1e9
        print(
            "Inferences per second:",
            round(completed / elapsed_seconds, 2),
        )


# ## Running the example

# We'll kick off our parallel training jobs and run inference from the command line.

# ```bash
# modal run finetune_yolo.py
# ```

# This runs the training in `quick_check` mode, useful for debugging the pipeline and getting a feel for it.
# To do a longer run that actually meaningfully improves performance, use:

# ```bash
# modal run finetune_yolo.py --no-quick-check
# ```


@app.local_entrypoint()
def main(quick_check: bool = True, inference_only: bool = False):
    """Run fine-tuning and inference on two datasets.

    Args:
        quick_check: fine-tune on a small subset. Lower quality results, but faster iteration.
        inference_only: skip fine-tuning and only run inference
    """

    birds = DatasetConfig(
        workspace_id="birds-s35xe",
        project_id="birds-u8mti",
        version=2,
        format="yolov9",
        target_class="🐥",
    )
    bees = DatasetConfig(
        workspace_id="bees-tbdsg",
        project_id="bee-counting",
        version=11,
        format="yolov9",
        target_class="🐝",
    )
    datasets = [birds, bees]

    # .for_each runs a function once on each element of the input iterators
    # here, that means download each dataset, in parallel
    if not inference_only:
        download_dataset.for_each(datasets)

    today = datetime.now().strftime("%Y-%m-%d")
    model_ids = [dataset.id + f"/{today}" for dataset in datasets]

    if not inference_only:
        train.for_each(model_ids, datasets, kwargs={"quick_check": quick_check})

    # let's run inference!
    for model_id, dataset in zip(model_ids, datasets):
        inference = Inference(
            volume_path / "runs" / model_id / "weights" / "best.pt"
        )

        # predict on a single image and save output to the volume
        test_images = volume.listdir(
            str(Path("dataset") / dataset.id / "test" / "images")
        )
        # run inference on the first 5 images
        for ii, image in enumerate(test_images):
            print(f"{model_id}: Single image prediction on image", image.path)
            inference.predict.remote(
                model_id=model_id,
                image_path=f"{volume_path}/{image.path}",
                display=(
                    ii == 0  # display inference results only on first image
                ),
            )
            if ii >= 4:
                break

        # streaming inference on images from the test set
        print(
            f"{model_id}: Streaming inferences on all images in the test set..."
        )
        count = 0
        for detection in inference.streaming_count.remote_gen(
            batch_dir=f"{volume_path}/dataset/{dataset.id}/test/images"
        ):
            if detection:
                print(f"{dataset.target_class}", end="")
                count += 1
            else:
                print("🎞️", end="", flush=True)
        print(f"\n{model_id}: Counted {count} {dataset.target_class}s!")


# ## Addenda

# The rest of the code in this example is utility code.

warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)
