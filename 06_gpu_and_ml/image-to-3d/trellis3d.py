# ---
# cmd: ["modal", "run", "06_gpu_and_ml/image-to-3d/trellis3d.py", "--image-path", "path/to/image.jpg"]
# ---

import logging
from pathlib import Path

import modal

TRELLIS_DIR = "/root/TRELLIS"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build our image with all dependencies
image = (
    modal.Image.conda()
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
    )
    # First install PyTorch with CUDA 12.4 support
    .conda_install(
        "pytorch",
        "torchvision",
        "pytorch-cuda=12.4",
        "pytorch3d",
        channels=["pytorch3d", "pytorch", "nvidia", "conda-forge"],
    )
    # Install Kaolin after PyTorch
    .pip_install(
        "git+https://github.com/NVIDIAGameWorks/kaolin.git",  # Install from source for CUDA 12.4 support
        "pytorch-lightning==2.1.3",
    )
    # Install TRELLIS and its dependencies
    .pip_install(
        "git+https://github.com/KAIST-Visual-AI-Group/TRELLIS.git",
        "opencv-python==4.8.1.78",
        "safetensors==0.4.1",
        "wandb==0.16.1",
        "spconv-cu124",  # Using CUDA 12.4 version
    )
    .env({"PYTHONPATH": "/root", "CUDA_HOME": "/usr/local/cuda-12.4"})
)

app = modal.App(name="example-trellis-3d")


@app.cls(gpu="A10G", image=image)
class Model:
    def __enter__(self):
        """Load TRELLIS pipeline on container startup."""
        from trellis.pipelines import TrellisImageTo3DPipeline

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "KAIST-Visual-AI-Group/TRELLIS",
            cache_dir="/root/.cache/trellis",
        )
        return self

    def process_image(self, image_path: str) -> bytes:
        """Process an image and return GLB file bytes.

        Args:
            image_path: Path to the input image file

        Returns:
            bytes: The generated GLB file as bytes
        """
        import cv2
        import numpy as np
        from PIL import Image

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate 3D model with optimized parameters
        output = self.pipeline(
            image=image,
            simplify=0.02,
            texture_size=1024,
            sparse_sampling_steps=20,
            sparse_sampling_cfg=7.5,
            slat_sampling_steps=20,
            slat_sampling_cfg=7,
            seed=42,
            output_format="glb",
        )
        return output


@app.local_entrypoint()
def main(
    image_path: str = "path/to/image.jpg",
    output_name: str = "output.glb",
):
    """Generate a 3D model from an image using TRELLIS.

    Args:
        image_path: Path to the input image
        output_name: Name of the output GLB file
    """
    # Create output directory
    output_dir = Path("/tmp/trellis-output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process image and generate 3D model
    model = Model()
    glb_bytes = model.process_image(image_path)

    # Save the GLB file
    output_path = output_dir / output_name
    output_path.write_bytes(glb_bytes)

    print(f"✓ Generated 3D model saved to {output_path}")
    print("You can view the model at https://glb.ee")
