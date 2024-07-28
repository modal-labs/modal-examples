import modal
from modal import Image

# create our app
app = modal.App("yolo-finetune")

# build container image
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "roboflow", "opencv-python"])
)

# add persistent volume for storing dataset, trained weights, and inference outputs
volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)

#
# ## Training
#

# function for downloading bees dataset
@app.function(
    image=image,
    volumes={"/root/data/": volume},
    secrets=[modal.Secret.from_name("roboflow-api-key")] # set up ROBOFLOW_API_KEY=yourkey as a secret in "roboflow-api-key" in the modal UI. this is injected as an env var.
)
def download_bees():
    from roboflow import Roboflow
    import os
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("bees-tbdsg").project("bee-counting").version(11)
    project.download("yolov9", location="/root/data/dataset/bees")

# function for training on bees dataset
minute = 60
@app.function(
    image=image,
    volumes={"/root/data": volume},
    gpu='a100',
    timeout=60*minute
)
def train():
    from ultralytics import YOLO
    # # Load a pre-trained YOLOv10n model
    model = YOLO("yolov10n.pt")

    model.train(
        data="/root/data/dataset/bees/data.yaml",
        epochs=8,
        batch=0.8, # automatic batch size to target 80% util
        workers=4,
        cache=True,
        name="bees",
        device=0,
        verbose=True,
        project='/root/data/runs', # saves trained weights to /root/data/runs/bees/weights/best.pt
        fraction=0.4,
    )

#
# ## Inference
#
# This example inference pipeline is a bit more complicated than the training pipeline.
#
# We use a class abstraction to load the model only once on container start and reuse it for future inferences.
# We use a generator to stream images to the model.
# 
# The images we use for inference are loaded from the /test set in our Volume.
# Since each image read from the Volume ends up being slower than the inference itself,
# we spawn up to 20 parallel workers to read images from the Volume and stream their bytes back to the model.

# Helper function to read images from the Volume in parallel
@app.function(
    image=image,
    volumes={"/root/data": volume},
    concurrency_limit=20, # prevent spawning too many containers
)
def read_image(image_path: str):
    import cv2
    source = cv2.imread(image_path)
    return cv2.imencode('.jpg', source)[1].tobytes()

# Class-based functions for inferencing
# We use the class abstraction to load the model only once on container start and reuse it for future inferences
@app.cls(
    image=image,
    volumes={"/root/data/": volume},
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
    @modal.method()
    def predict(self, image_path: str):
        # inference on image and save to /root/data/out/bees/{image_path}
        # you can view the file from the Volumes UI in modal
        self.model.predict(image_path, image_path, save=True, project="/root/data/out", name="bees", exist_ok=True)
    
    @modal.method()
    def batch(self, batch_dir: str, threshold: float|None = None):
        import time, os, cv2
        import numpy as np

        # Input stream
        # Spawn workers to stream images from Volume into a generator
        image_files = os.listdir(batch_dir)
        image_files = [os.path.join(batch_dir, f) for f in image_files]
        print("image_files", len(image_files))
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
            results = self.model.predict(
                image,
                project="/root/data/out", 
                name="bees", 
                exist_ok=True,
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
    

# modal run --detach yolo::demo
# this runs training and inference on a single image, saving output image to the volume.
@app.local_entrypoint()
def demo():
    download_bees.remote()
    train.remote()
    inference.local()

# modal run --detach yolo::inference
# this runs inference on a single image, saving output image to the volume.
@app.local_entrypoint()
def inference():
    inference = Inference("/root/data/runs/bees/weights/best.pt")
    inference.predict.remote(image_path = "/root/data/dataset/bees/test/images/20230228-122000_jpg.rf.5fd572dfd6d555c072d65facfaf897fc.jpg") 

# modal run yolo::batch
# this counts the number of bees in the 1035 image test set
# this is a more realistic inference pipeline, where we stream images from the Volume
@app.local_entrypoint()
def batch():
    inference = Inference("/root/data/runs/bees/weights/best.pt")
    count = inference.batch.remote(batch_dir = "/root/data/dataset/bees/test/images") 
    print(f"Counted {count} bees!")