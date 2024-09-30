# ---
# cmd: ["modal", "run", "06_gpu_and_ml.sam.app"]
# deploy: true
# ---
import modal

from .helper import show_anns, show_mask, show_points

# # Run Facebook's Segment Anything Model (SAM) on Modal

# This example demonstrates how to deploy Facebook's Segment Anything Model (SAM)
# on Modal. SAM is a powerful, flexible image segmentation model that can be used
# for various computer vision tasks like object detection, instance segmentation,
# and even as a foundation for more complex computer vision applications.
# For more information, see: https://github.com/facebookresearch/segment-anything

# # Setup

# First, we set up the Modal image with the necessary dependencies, including PyTorch,
# OpenCV, and Torchvision. We also install the Segment Anything Model (SAM) library.

MODEL_TYPE = "vit_h"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-glx")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "onnxruntime",
        "onnx",
        "git+https://github.com/facebookresearch/segment-anything.git",
    )
)

app = modal.App("sam-app", image=image)

# # Model definition

# Next, we define the Model class that will handle SAM operations


# We use @modal.build() and @modal.enter() decorators here for optimization:
# @modal.build() ensures this method runs during the container build process,
# downloading the model only once and caching it in the container image.
# @modal.enter() makes sure the method runs only once when a new container starts,
# initializing the model and moving it to GPU.
# The upshot is that model downloading and initialization only happen once upon container startup.
# This significantly reduces cold start times and improves performance.
@app.cls(gpu="any", timeout=600)
class Model:
    def __init__(self, model_type=MODEL_TYPE):
        self.model_type = model_type

    # # Model initialization
    @modal.build()
    @modal.enter()
    def download_model_to_folder(self):
        import subprocess

        from segment_anything import (
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        # SAM offers different model sizes (ViT-H, ViT-L, ViT-B)
        # Here we're downloading the specified model checkpoint
        model_type_string = ""
        if self.model_type == "vit_h":
            model_type_string = "vit_h_4b8939"
        elif self.model_type == "vit_l":
            model_type_string = "vit_l_0b3195"
        elif self.model_type == "vit_b":
            model_type_string = "vit_b_01ec64"
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        # Download the model checkpoint
        subprocess.run(
            [
                "wget",
                f"https://dl.fbaipublicfiles.com/segment_anything/sam_{model_type_string}.pth",
                "-O",
                f"sam_{model_type_string}.pth",
            ]
        )

        print(
            f"Downloaded model {model_type_string} to sam_{model_type_string}.pth"
        )

        # Initialize the SAM model and move it to GPU
        sam = sam_model_registry[MODEL_TYPE](
            checkpoint=f"sam_{model_type_string}.pth"
        )
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    # # Automatic mask generation
    @modal.method()
    def generate_masks_automatically(self, image):
        print(f"image shape: {image.shape}")

        return self.mask_generator.generate(image)

    # # Prompt-based mask generation
    @modal.method()
    def generate_masks_with_prompts(self, prompts, image):
        input_point, input_label = prompts
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return (masks, scores, logits)


# # Local entrypoint


# Finally, We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the training. This local entrypoint runs both:
#  1. Automatic mask generation: Segmenting all objects in an image without prompts
# 2. Prompt-based segmentation: Generating masks based on specific prompts (e.g., points or boxes)
@app.local_entrypoint()
def main():
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # # Image loading
    image = cv2.imread("06_gpu_and_ml/sam/dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Model inference
    # Create a Model instance and run inference
    # This demonstrates how to use SAM for image segmentation tasks
    model = Model(model_type=MODEL_TYPE)
    masks = model.generate_masks_automatically.remote(image=image)

    # # Visualization
    # show the image with the auto-generated masks overlaid
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.show()

    # display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("on")
    plt.show()

    # # Prompt-based segmentation
    # set the input point and label
    input_point = np.array([[300, 250]])
    input_label = np.array([1])

    masks, scores, _ = model.generate_masks_with_prompts.remote(
        prompts=[input_point, input_label], image=image
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
