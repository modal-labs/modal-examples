# # Training and Running YOLOv10
#
# Example by [@Erik-Dunteman](https://github.com/erik-dunteman)
#

# The popular YOLO "You Only Look Once" model line provides high-quality object detection in an economical package. 
# In this example, we use the [YOLOv10](https://docs.ultralytics.com/models/yolov10/) model, released on May 23, 2024
#
# We will:
# - Download two custom datasets from the [Roboflow](https://roboflow.com/) computer vision platform
# - Fine tune on those datasets, in parallel, using the [Ultralytics package](https://docs.ultralytics.com/)
# - Run inference with the fine-tuned models.

# For commercial use, be sure to consult the [ultralytics software license](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed).

import modal
from modal import Image
import modal.parallel_map
from torch import futures
from ultralytics import download


# ## Building Our Image
# 
# All packages are installed into a Debian Slim base image
# using the `pip_install` function.
# 
# We also create a persistent volume for storing dataset, trained weights, and inference outputs between function calls.

# build container image
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "roboflow", "opencv-python"])  # TODO: pin versions
)
 
# add our volume
volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)
volume_path = "/root/data" # the path to the volume from within the container

# create our app
app = modal.App("yolo-finetune", image=image, volumes={volume_path: volume})


#
# ## Training
# Training will be done as two functions: one for downloading the dataset, and one for training the model.
#
# We'll be downloading our data from the [Roboflow](https://roboflow.com/) computer vision platform, so you'll need to:
# - Create a free account on [Roboflow](https://app.roboflow.com/)
# - [Generate a Private API key](https://app.roboflow.com/settings/api)
# - Set up a Modal [Secret](https://modal.com/docs/guide/secrets) called `roboflow-api-key` in the modal UI, setting these key-values
#   - ROBOFLOW_API_KEY=yourkey
# 
# You're also free to bring your own dataset, with a config in YOLOv10-compatible yaml format.
#
# We'll be training on the medium size model, but you're free to experiment with [other model sizes](https://docs.ultralytics.com/models/yolov10/#model-variants)

# Function for downloading a roboflow dataset
@app.function(
    secrets=[modal.Secret.from_name("roboflow-api-key")] # set up ROBOFLOW_API_KEY=yourkey as a secret in "roboflow-api-key" in the modal UI. this is injected as an env var.
)
def download_dataset(workspace_id: str, project_id: str, version: int, format: str):
    from roboflow import Roboflow
    import os

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(workspace_id).project(project_id).version(version)
    model_id = f"{workspace_id}/{project_id}/{version}"
    dataset_dir = f"{volume_path}/dataset/{model_id}"
    project.download(format, location=dataset_dir)

# After this, we'll have a file tree like this:
#
# https://modal.com/{your-modal-username}/main/storage/yolo-finetune
# - dataset
#   - birds-s35xe
#     - birds-u8mti
#       - 2
#         - data.yaml
#         - train/
#           - images
#           - labels
#         - valid/
#           ...
#         - test/
#           ...
#         ...
#   - bees-tbdsg
#     ...
#
# We'll be training on the `train` and `valid` directories, and running inference on the `test` directory.


# Function for training on a downloaded dataset
# Assumes yaml config file
minute = 60
@app.function(
    gpu='a100',
    timeout=60*minute
)
def train(model_id: str, model_size = "yolov10m.pt"):
    from ultralytics import YOLO
    import os

    volume.reload() # make sure volume is synced
    
    if not os.path.exists(f"{volume_path}/runs/{model_id}"):
        os.makedirs(f"{volume_path}/runs/{model_id}")

    # # Load a pre-trained YOLOv10n model
    model = YOLO(model_size)
    model.train(
        data= f"{volume_path}/dataset/{model_id}/data.yaml",
        epochs=8,
        batch=0.8, # automatic batch size to target 80% util
        workers=4,
        cache=True,
        name=model_id,
        exist_ok=True, # overwrite previous model if it exists
        device=0,
        verbose=True,
        project=f'{volume_path}/runs',
        fraction=0.4,
    )

#
# ## Inference
#
# This example inference pipeline is a bit more complicated than the training function.
#
# We use a class abstraction to load the model only once on container start and reuse it for future inferences.
# We use a generator to stream images to the model.
# 
# The images we use for inference are loaded from the /test set in our Volume.
# Each image read takes ~50ms, and inference can take ~5ms, so the disk read is our biggest bottleneck if we just looped over the images.
# So, we parallelize the disk reads across many workers using Modal's function.map(), and stream the image bytes to the model, shifting the bottleneck to network IO.
# This increases throughput to ~60 images/s, or ~17 milliseconds/image.

# Helper function to read images from the Volume in parallel
@app.function(
    concurrency_limit=20, # prevent spawning too many containers
)
def read_image(image_path: str):
    import cv2
    source = cv2.imread(image_path)
    return cv2.imencode('.jpg', source)[1].tobytes()

# Class-based functions for inferencing
# We use the class abstraction to load the model only once on container start and reuse it for future inferences
@app.cls(
    gpu='a10g',
)
class Inference:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    # A single image prediction method
    # Saves the prediction to the volume
    @modal.method()
    def predict(self, model_id: str, image_path: str):
        self.model.predict(
            image_path, 
            save=True, 
            exist_ok=True,
            project=f"{volume_path}/predictions/{model_id}"
        )
         # you can view the file from the Volumes UI in modal
    
    # Counts the number of objects in a directory of images
    # Does not save the predictions to the volume
    @modal.method()
    def batch_count(self, batch_dir: str, threshold: float|None = None):
        import time, os, cv2
        import numpy as np

        # Input stream
        # Spawn workers to stream images from Volume into a generator
        image_files = os.listdir(batch_dir)
        image_files = [os.path.join(batch_dir, f) for f in image_files]
        def image_generator():
            for image_bytes in read_image.map(image_files):
                # Function.map runs read_image in parallel, yielding images as bytes. 
                # We now decode those bytes into openCV images and yield them into our own generator.
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                yield image

        # image_generator is now a generator which yields openCV images the moment they're available
        # this is to emulate the behavior of a real-time inference pipeline

        count = 0
        completed = 0
        start = time.time()
        for image in image_generator():
            # note that while we call this function "batch", we are not referring to the ML terminology (IE "batch dimension")
            # we're only inferencing on a single image at a time (batch_size=1)
            # 
            # to parallelize inference, one could add logic to accumulate images from the generator across a time window into batches
            # but each single inference is usually done before next images arrives, so there's no benefit to batching inference
            results = self.model.predict(
                image,
                save=False, # don't save to disk, as it slows down the pipeline significantly
            )
            completed += 1
            for res in results:
                for conf in res.boxes.conf:
                    if threshold == None:
                        count += 1
                        continue
                    if conf.item() >= threshold:
                        count += 1

        print("Inferences per second", completed / (time.time() - start))   
        return count
    
# ## Running the example 
# 
# We'll kick off our parallel training jobs and run inference from the command line.
# ```bash
# modal run --detach yolo
# ```

@app.local_entrypoint()
def main():
    # we're going to parallel train and inference on two datasets: 
    # birds
    # and bees
    # (you're probably not familiar with birds and bees, so we'll use the power of AI to help you!)

    datasets = [
        ("birds-s35xe", "birds-u8mti", 2, "yolov9"),  # workspace_id, project_id, version, format
        ("bees-tbdsg", "bee-counting", 11, "yolov9"),
    ]

    # let's download our datasets, in parallel
    # starmap parallelizes the function across the list of input arguments
    results = download_dataset.starmap(datasets)
    list(results) # the result is a generator, so collecting it into a list resolves it

    # let's train our models, in parallel
    model_ids = []
    for dataset in datasets:
        model_id = f"{dataset[0]}/{dataset[1]}/{dataset[2]}"
        model_ids.append(model_id)
    results = train.map(model_ids)
    list(results)

    # # let's run inference!
    # we'll do this serially for variety (since we saved so much time by parallelizing training)
    for model_id in model_ids:
        inference = Inference(f"{volume_path}/runs/{model_id}/weights/best.pt")

        # predict on a single image and save output to the volume
        test_images = volume.listdir(f"/dataset/{model_id}/test/images")
        # inference the first 5 images
        i = 0
        for image in test_images:
            print(f"{model_id}: Single image prediction on image", image.path)
            inference.predict.remote(
                model_id = model_id, 
                image_path = f"{volume_path}/{image.path}"
            )
            i += 1
            if i >= 5:
                break

        # # batch inference on all images in the test set and return the count of detections
        # print(f"{model_id}: Batch inference on all images in the test set...")
        count = inference.batch_count.remote(batch_dir = f"{volume_path}/dataset/{model_id}/test/images")
        print(f"{model_id}: Counted {count} objects!")