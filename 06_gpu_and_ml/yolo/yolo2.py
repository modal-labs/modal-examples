import modal
from modal import Image


app = modal.App("yolo-finetune")
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "roboflow", "opencv-python"])
    .copy_local_file("dog.png", "/dog.png")
    .copy_local_file("bees.png", "/bees.png")
)

volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/root/data/": volume},
    secrets=[modal.Secret.from_name("roboflow-api-key")] # set up ROBOFLOW_API_KEY=yourkey as a secret "roboflow-api-key". this is injected as an env var.
)
def download_bees():
    from roboflow import Roboflow
    import os
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("bees-tbdsg").project("bee-counting").version(11)
    project.download("yolov9", location="/root/data/dataset/bees")


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
        project='/root/data/runs',
        fraction=0.4,
    )

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
        self.model.predict(image_path, save=True, project="/root/data/out", name="bees", exist_ok=True)

# modal run --detach yolo2::demo
@app.local_entrypoint()
def demo():
    download_bees.remote()
    train.remote()

    inference = Inference("/root/data/runs/bees3/weights/best.pt")
    inference.predict.remote(image_path = "/bees.png") 
    inference.predict.remote(image_path = "/dog.png")


# modal run yolo2::inference
@app.local_entrypoint()
def inference():
    inference = Inference("/root/data/runs/bees3/weights/best.pt") # runs init
    inference.predict.remote(image_path = "/bees.png") # runs enter and method (cold boot)
    inference.predict.remote(image_path = "/dog.png") # runs method (warm)