# ---
# cmd: ["modal", "run", "06_gpu_and_ml/image-to-3d/trellis3d.py", "--image-path", "path/to/image.jpg"]
# ---

import io
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
    # Step 1: Install PyTorch with CUDA 12.4 first
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Step 2: Install Kaolin (requires PyTorch to be installed first)
    .pip_install("kaolin==0.15.0")
    # Step 3: Install TRELLIS dependencies
    .pip_install(
        "numpy",
        "opencv-python",
        "trimesh",
        "matplotlib",
        "scipy",
        "scikit-image",
        "requests",
        "warp-lang",
        "ipyevents",
        "easydict",
        "einops",
        "xatlas",
        "pytorch3d",  # Required by TRELLIS
        "pytorch-lightning",  # Required by TRELLIS
        "wandb",  # Required by TRELLIS
        "tqdm",  # Required by TRELLIS
        "safetensors",  # Required by TRELLIS
        "huggingface-hub",  # Required by TRELLIS
    )
    # Step 4: Clone and install TRELLIS
    .run_commands(
        f"git clone https://github.com/JeffreyXiang/TRELLIS.git {TRELLIS_DIR}",
        f"cd {TRELLIS_DIR} && pip install -e .",
        # Verify installation
        "python3 -c 'from trellis.pipelines import TrellisImageTo3DPipeline'",
    )
)

stub = modal.Stub(name="example-trellis-3d")

@stub.cls(gpu="A10G", image=image)
class Model:
    def __enter__(self):
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

@stub.local_entrypoint()
def main(image_path: str = "path/to/image.jpg"):
    """Generate a 3D model from an input image.

    Args:
        image_path: Path to the input image file.
    """
    model = Model()
    output = model.generate.remote(image_path)

    # Save output to temporary directory following text_to_image.py pattern
    output_dir = Path("/tmp/trellis-3d")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "output.glb"
    output_path.write_bytes(output)

    logger.info(f"Output saved to {output_path}")
    return str(output_path)
