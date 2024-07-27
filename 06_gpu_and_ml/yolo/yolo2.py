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
)
def download_bees():
    from roboflow import Roboflow

    # Bees?
    rf = Roboflow(api_key="Qa6qcF7kZDYza6MUlefE")
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
        res = self.model.predict(image_path)
        res[0].save("/root/data/out/bees")

@app.local_entrypoint()
def predict():
    Inference("/root/data/runs/bees2/weights/best.pt").predict.remote(image_path = "/bees.png")