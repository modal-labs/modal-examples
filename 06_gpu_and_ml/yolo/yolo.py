# ---
# runtimes: ["runc", "gvisor"]
# ---
#
# # Training and Inferencing YOLOv10
#
# Example by [@AnirudhRahul](https://github.com/AnirudhRahul)
#
# [YOLOv10](https://docs.ultralytics.com/models/yolov10/) is the most modern version of the popular "You Only Look Once" ("YOLO") computer vision model, 
# which has been trained to perform bounding box object detection.
#
# In this example, we use the V10 model via the [Ultralytics package](https://docs.ultralytics.com/). We will:
# - Fine tune on a custom dataset, in our case the DoclayNet dataset for object detection within documents.
# - Deploy an inference endpoint.

# For commercial use, sure to consult the [ultralytics software license](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed).

# ## Building Our Image
#
# The example uses the Ultralytics package to run training inference and the Huggingface dataset package to load DoclayNet data.
# 
# All packages are installed into a Pytorch 2.3.1 base image
# using the `pip_install` function.
#
# In addition, we download the base YOLO model weights into the container at build-time
# using the `run_function` function

import modal
import urllib.request, os, yaml, json, shutil, zipfile
from pathlib import Path

def load_yolo_weights():
    from ultralytics import YOLO
    # Choose the large size checkpoint
    # Other model sizes can be found at: https://docs.ultralytics.com/models/yolov10/#performance
    model = YOLO('yolov10l.pt') # This downloads the model's base weights if not already present in the image.
    return model

yolo_image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "datasets", "opencv-python"])
    .run_function(load_yolo_weights)

app = modal.App("train-yolo")

# ## Downloading The Dataset
# 
# Similar to downloading model weights at build-time, it may be tempting to also download the DoclayNet dataset at build-time.
# We instead choose to download the dataset into a Modal Volume, using a Modal Function.
# This approach has two benefits:
# - Reduces the size of the container image, making cold boots much faster. This is especially important for inference when there's no need for having the data in the container.
# - Makes it easier to train on many datasets, storing to create many finetunes usable for inference.

# Define a Volume, which will persist training data and trained weights between invocations 
volume = modal.Volume.from_name("doclaynet-base-yolo", create_if_missing=True)

def download_file(url, output_path):
    from tqdm import tqdm
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def convert_dataset_to_yolo(dataset_path, output_path):
    from tqdm import tqdm
    root_folder = Path(output_path)
    if not root_folder.exists():
        root_folder.mkdir(parents=True)

    category_map = {
                    "0": "Caption", "1": "Footnote", "2": "Formula", "3": "List-item",
                    "4": "Page-footer", "5": "Page-header", "6": "Picture",
                    "7": "Section-header", "8": "Table", "9": "Text", "10": "Title",
                }
    inverse_category_map = {v: k for k, v in category_map.items()}
    with open(root_folder / "data.yaml", "w") as f:
        yaml.dump(
            {
                "path": root_folder.absolute().as_posix(),
                "train": "./images/train",
                "val": "./images/val",
                "test": "./images/test",
                "names": category_map,
            },
            f,
        )
 
    for split in ["train", "val", "test"]:
        print(f"Converting {split} dataset...")
        
        ann_dir = os.path.join(dataset_path, f"base_dataset/{split}/annotations")
        img_dir = os.path.join(dataset_path, f"base_dataset/{split}/images")
        
        os.makedirs(root_folder / "labels" / split, exist_ok=True)
        os.makedirs(root_folder / "images" / split, exist_ok=True)

        for file in tqdm(os.listdir(ann_dir)):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

            image_id = os.path.splitext(file)[0]
            src_image_path = os.path.join(img_dir, f"{image_id}.png")
            dst_image_path = root_folder / "images" / split / f"{image_id}.png"
            # Have to copy this is cross device
            shutil.copy(src_image_path, dst_image_path)

            coco_width, coco_height = data["metadata"]["coco_width"], data["metadata"]["coco_height"]
            
            with open(root_folder / "labels" / split / f"{image_id}.txt", "w") as f:
                seen_annotations = set()
                for item in data["form"]:
                    category_str = item["category"]
                    category = inverse_category_map[category_str]

                    bbox = item["box"]
                    left, top, width, height = bbox
                    left /= coco_width
                    top /= coco_height
                    width /= coco_width
                    height /= coco_height
                    center_x = left + width / 2
                    center_y = top + height / 2

                    annotation = (category, center_x, center_y, width, height)
                    if annotation not in seen_annotations:
                        f.write(f"{category} {center_x} {center_y} {width} {height}\n")
                        seen_annotations.add(annotation)

@app.function(
    image=yolo_image,
    volumes={"/root/data/": volume},
    cpu=4,
)
def download_dataset():
    url = "https://huggingface.co/datasets/pierreguillou/DocLayNet-base/resolve/main/data/dataset_base.zip"
    zip_path = "dataset_base.zip"
    extract_path = "dataset_base"
    
    print("Downloading dataset...")
    download_file(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Convert the dataset to YOLO format
    output_path = "/root/data/doclay-dataset"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    print("Converting dataset to YOLO format...")
    convert_dataset_to_yolo(extract_path, output_path)

# ## Running the Download
# We call our Modal function to download the dataset into a Volume. This happens 100% in the cloud.
# ```bash
# modal run yolo.py::download_dataset
# ```
# Our dataset should now be in a Volume, visible from within the container at /root/data

# ## Training the model
# 
# With the dataset downloaded, we proceed to building our training pipeline.

@app.function(
    image=yolo_image,
    timeout=1500,
    volumes={"/root/data/": volume},
    gpu='a100',
    cpu=2,
)
def train(model_name = "doclaynet-base", data_path = '/root/data/doclay-dataset/data.yaml'):
    img_size = 896
    model = load_yolo_weights()
    model.train(
        data=data_path,
        epochs=8,
        batch=0.8, # automatic batch size to target 80% util
        workers=4,
        cache=True,
        imgsz=img_size,
        name=model_name,
        device=0,
        verbose=True,
        val=True,
        plots=True,
        save_period=1,
        project='/root/data/runs',
        fraction=0.4,
        exist_ok=True, # Replace existing run
    )

# ## Running the Training Job
# We call our Modal train function to start our training job.
# The --detach flag is used so that the job continues in the cloud even if our terminal disconnects.
# ```bash
# modal run --detach yolo.py::train
# ```
# Once successful, Ultralytics will save the trained model in /root/data/runs/doclaynet-base/weights/best.pt
# 
# 
# ### Optional: Running Additional Finetunes
# If you'd like to train new models on other datasets you've downloaded into your Volume, 
# you can reuse the train function as such:
# ```bash
# modal run --detach yolo.py::train --model_name=finetune_1 --data_path=/root/data/finetune1-dataset/data.yaml
# ```

# ## Creating Our Inference Class
# 
# We define a class-based remote handler for inference, allowing us to load our model only once 
# on container boot and reuse it for future inferences.

@app.cls(
    image=yolo_image,
    volumes={"/root/data/": volume},
    gpu='h100',
)
class YOLOInference:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @modal.enter()
    def load_model(self):
        # load model only once, on container boot
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    @modal.method()
    def predict(self, input_dir: str, output_dir: str):
        import os
        from tqdm import tqdm

        # We'll be saving results into 
        os.makedirs(output_dir, exist_ok=True)

        # Get all image files from the input directory
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in tqdm(image_files):
            input_path = os.path.join(input_dir, image_file)
            
            # Perform inference
            results = self.model.predict(input_path)
            
            # Save the results
            output_path = os.path.join(output_dir, os.path.basename(image_file))
            print("SAVING: ", output_path)
            results[0].save(output_path)

        print(f"Inference complete. Results saved to {output_dir}")
    
@app.local_entrypoint()
def infer(model_name = "doclaynet-base", input_dir = "/root/data/doclay-dataset/images/val"):
    from pathlib import Path
    import os

    OUTPUT_DIR = "./val-output" # MUST BE RELATIVE PATH
    weights_path= f"/root/data/runs/{model_name}/weights/best.pt"

    # We'll be using input images from the Volume for simplicity,
    # saving the bounding-box images outputed from inference into the Volume as well.
    YOLOInference(weights_path).predict.remote(
        input_dir = input_dir,
        output_dir = Path("/root/data/") / OUTPUT_DIR,
    )

    # We can download those images from the Volume to local using the modal CLI
    cmd = f"modal volume get doclaynet-base-yolo {OUTPUT_DIR[1:]} --force"
    os.system(cmd)

# ## Running the Inference Task
# We call our Modal infer class to load the model to GPU and perform inference
# ```bash
# modal run yolo.py::infer
# ```
# Once successful, classified output images will be saved to the Volume and downloaded to your local filesystem.
# 
# 
# ### Optional: Inferencing Additional Finetunes
# If you'd like to swap in new finetunes, you can reuse the inference function as such:
# ```bash
# modal run yolo.py::infer --model_name=finetune_1 --input_dir=/root/data/finetune1-dataset/images/val
# ```