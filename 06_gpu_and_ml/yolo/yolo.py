import modal
from modal import Image

# create our app
app = modal.App("yolo-finetune")

# build container image
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "roboflow", "opencv-python"])
    .copy_local_file("./demo_images/dog.png", "/demo_images/dog.png")
    .copy_local_file("./demo_images/bees.png", "/demo_images/bees.png")
)

# add persistent volume for storing dataset, trained weights, and inference outputs
volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)

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

# class-based functions for inferencing
# we use the class abstraction to load the model only once on container start and reuse it for future inferences
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
        # load model only once, on container boot
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    @modal.method()
    def predict(self, image_path: str):
        # inference on image and save to /root/data/out/bees/{image_path}
        # you can view the file from the Volumes UI in modal
        self.model.predict(image_path, save=True, project="/root/data/out", name="bees", exist_ok=True)


# modal run --detach yolo::demo
@app.local_entrypoint()
def demo():
    download_bees.remote()
    train.remote()

    inference = Inference("/root/data/runs/bees/weights/best.pt")
    inference.predict.remote(image_path = "/demo_images/bees.png") 
    inference.predict.remote(image_path = "/demo_images/dog.png")


# modal run yolo::inference
@app.local_entrypoint()
def inference():
    inference = Inference("/root/data/runs/bees/weights/best.pt")
    inference.predict.remote(image_path = "/demo_images/bees.png") # first call cold boot (enter) then run the method
    inference.predict.remote(image_path = "/demo_images/dog.png") # subsequent calls are warm and just run the method