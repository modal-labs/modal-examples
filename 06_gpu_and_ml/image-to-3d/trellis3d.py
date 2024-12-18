import io
import logging
import traceback
from pathlib import Path

import modal
import requests

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/microsoft/TRELLIS.git"
MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"
TRELLIS_DIR = "/trellis"
MINUTES = 60
HOURS = 60 * MINUTES

cuda_version = "12.4.0"
flavor = "base"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"


def clone_repository():
    import subprocess

    subprocess.run(
        ["git", "clone", "--recurse-submodules", REPO_URL, TRELLIS_DIR],
        check=True,
    )


# Multi-step installation is required due to dependency ordering:
# 1. PyTorch must be installed first as it's required by several ML libraries
# 2. CUDA development tools are needed for building flash-attn
# 3. Kaolin requires a pre-installed PyTorch
# 4. Flash-attention needs special handling due to CUDA compilation
# 5. The rest of the dependencies can be installed together
trellis_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "cmake",
        "clang",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
        "libxrender1",
        "libxext6",
        "ninja-build",
        "cuda-nvcc-12-4",  # Added CUDA compiler
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "NVCC_PATH": "/usr/local/cuda/bin/nvcc",
            "PATH": "/usr/local/cuda/bin:${PATH}",
        }
    )
    # Step 1: Install core Python packages needed for building
    .pip_install(
        "packaging==23.2",
        "wheel==0.42.0",
        "setuptools==69.0.3",
    )
    # Step 2: Install PyTorch first as it's required by several dependencies
    .pip_install(
        "torch==2.1.2",  # Updated to match CUDA 12.4 compatibility
        "torchvision==0.16.2",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    # Step 3: Install Kaolin and its dependencies
    .pip_install(
        "kaolin==0.15.0",
        "warp-lang",  # Required by kaolin physics module
        "ipyevents",  # Optional but removes warnings
        extra_index_url=[
            "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu124/",
        ],
    )
    # Step 4: Install TRELLIS dependencies with spconv-cu118 (latest available version)
    .pip_install(
        "easydict",
        "einops",
        "fastapi[standard]==0.115.6",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "hf_transfer",
        "https://github.com/camenduru/wheels/releases/download/3090/diso-0.1.4-cp310-cp310-linux_x86_64.whl",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
        "igraph",
        "imageio",
        "imageio-ffmpeg",
        "largesteps",
        "numpy",
        "onnxruntime",
        "opencv-python-headless",
        "pillow",
        "pymeshfix",
        "pyvista",
        "rembg",
        "safetensors",
        "scipy",
        "spconv-cu118",  # Latest available version, CUDA 11.8 compatible
        "tqdm",
        "trimesh",
        "xatlas",
        extra_options="--no-build-isolation",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SPCONV_ALGO": "native",
        }
    )
    .run_function(clone_repository)
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
        sys.path.append(TRELLIS_DIR)

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
    image_path: str = "https://raw.githubusercontent.com/sandeeppatra/trellis/main/assets/images/dog.png",
    output_format: str = "glb",
):
    """Generate a 3D model from an input image.

    Args:
        image_path: Path to input image or URL
        output_format: Output format, either 'glb' or 'obj'

    Returns:
        None. Saves the output file to disk and prints its location.
    """
    model = Model()
    output = model.generate.remote(image_path, output_format)

    output_dir = Path("/tmp/trellis3d")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / f"output.{output_format}"
    output_path.write_bytes(output)

    print(f"\nOutput saved to: {output_path}")
    print("You can view the GLB file at https://glb.ee/")
