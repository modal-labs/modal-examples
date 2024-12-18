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
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
    )
    # First install PyTorch with CUDA 12.4 support, required by downstream dependencies
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu124",  # Updated to CUDA 12.4
    )
    # Install Kaolin after PyTorch is installed
    .pip_install(
        "git+https://github.com/NVIDIAGameWorks/kaolin.git",  # Install from source for CUDA 12.4 support
        "pytorch-lightning==2.1.3",
        "pytorch3d==0.7.5",
    )
    # Install TRELLIS and its dependencies
    .pip_install(
        "git+https://github.com/KAIST-Visual-AI-Group/TRELLIS.git",
        "opencv-python==4.8.1.78",
        "safetensors==0.4.1",
        "wandb==0.16.1",
        "spconv-cu124",  # Updated to CUDA 12.4 version
    )
    .env({"PYTHONPATH": "/root", "CUDA_HOME": "/usr/local/cuda-12.4"})
)

app = modal.App(name="example-trellis-3d")


@app.cls(gpu="A10G", image=image)
class Model:
    @modal.enter()
    def enter(self):
        from trellis.pipelines import TrellisImageTo3DPipeline

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "KAIST-Visual-AI-Group/TRELLIS",
            cache_dir="/root/.cache/trellis",
        )

    def process_image(self, image_path):
        import cv2
        import numpy as np
        from PIL import Image

        # Load and preprocess image
        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Generate 3D model
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

    @modal.method()
    def generate(self, image_path):
        return self.process_image(image_path)


@app.local_entrypoint()
def main(image_path: str = "path/to/image.jpg"):
    """Generate a 3D model from an input image.

    Args:
        image_path: Path to the input image file.
    """
    # Create output directory in /tmp following Modal examples pattern
    output_dir = Path("/tmp/trellis-3d")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate 3D model
    model = Model()
    output = model.generate.remote(image_path)

    # Save output GLB file using write_bytes
    output_path = output_dir / "output.glb"
    output_path.write_bytes(output)

    print(f"Output saved to {output_path}")
    return str(output_path)
