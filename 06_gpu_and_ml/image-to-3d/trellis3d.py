# ---
# cmd: ["modal", "run", "06_gpu_and_ml/image-to-3d/trellis3d.py"]
# ---

import io
import logging
import traceback

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
    # Step 3: Install other dependencies
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
    )
    # Step 4: Clone TRELLIS and install its dependencies
    .run_commands(
        f"git clone https://github.com/JeffreyXiang/TRELLIS.git {TRELLIS_DIR}",
        f"cd {TRELLIS_DIR} && pip install -r requirements.txt",
        f"cd {TRELLIS_DIR} && pip install -e .",
    )
    # Step 5: Set environment variables for TRELLIS
    .env({"PYTHONPATH": TRELLIS_DIR})
)

app = modal.App(name="example-trellis-3d")


@app.cls(
    gpu=modal.gpu.L4(count=1),
    timeout=60 * 60,
    container_idle_timeout=60,
)
class Model:
    @modal.enter()
    def initialize(self):
        import sys

        # Add TRELLIS to Python path
        sys.path.append(TRELLIS_DIR)

        # Import TRELLIS after adding to Python path
        from trellis.pipelines import TrellisImageTo3DPipeline

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "KAIST-Visual-AI-Group/TRELLIS",
            cache_dir="/root/.cache/trellis",
        )

    def process_image(
        self,
        image_url: str,
        simplify: float = 0.02,
        texture_size: int = 1024,
        sparse_sampling_steps: int = 20,
        sparse_sampling_cfg: float = 7.5,
        slat_sampling_steps: int = 20,
        slat_sampling_cfg: int = 7,
        seed: int = 42,
        output_format: str = "glb",
    ):
        import cv2
        import numpy as np
        import requests  # Import requests here after it's installed
        from PIL import Image

        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            output = self.pipeline(
                image=image,
                simplify=simplify,
                texture_size=texture_size,
                sparse_sampling_steps=sparse_sampling_steps,
                sparse_sampling_cfg=sparse_sampling_cfg,
                slat_sampling_steps=slat_sampling_steps,
                slat_sampling_cfg=slat_sampling_cfg,
                seed=seed,
                output_format=output_format,
            )
            return output

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @modal.method()
    def generate(self, image_url: str, output_format: str = "glb") -> bytes:
        return self.process_image(
            image_url=image_url,
            output_format=output_format,
        )


@app.local_entrypoint()
def main(
    image_url: str = "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/astronaut.png",
    output_filename: str = "output.glb",
):
    """Generate a 3D model from an image.

    Args:
        image_url: URL of the input image
        output_filename: Name of the output GLB file
    """
    from pathlib import Path

    import requests

    # Create temporary directories
    output_dir = Path("/tmp/trellis-output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download and process image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {image_url}")

    # Process image and generate 3D model
    model = Model()
    try:
        # Call generate remotely using Modal's remote execution
        glb_bytes = model.generate.remote(response.content)

        # Save output file
        output_path = output_dir / output_filename
        output_path.write_bytes(glb_bytes)
        print(f"Generated 3D model saved to {output_path}")

        # Return file path for convenience
        return str(output_path)
    except Exception as e:
        print(f"Error generating 3D model: {str(e)}")
        print(traceback.format_exc())
        raise
