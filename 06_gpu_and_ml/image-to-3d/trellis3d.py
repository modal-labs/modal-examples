import logging
import traceback
from pathlib import Path

import modal
import requests
from fastapi import HTTPException, Request, Response, status

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
        "torch==2.1.0",  # Pinned to exact version required by torchvision
        "torchvision==0.16.0+cu121",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # Step 3: Install NVIDIA libraries and ML tools
    .pip_install(
        "flash-attn==2.6.3",
        "xformers",
        "warp-lang",  # Required by kaolin physics module
        "ipyevents",  # Optional but removes warnings
        extra_options="--no-build-isolation",
    )
    # Step 4: Install Kaolin after its dependencies
    .pip_install(
        "git+https://github.com/NVIDIAGameWorks/kaolin.git",
    )
    # Step 5: Install the rest of the dependencies
    .pip_install(
        # ML dependencies
        "numpy",
        "pillow",
        "imageio",
        "onnxruntime",
        "trimesh",
        "safetensors",
        "easydict",
        "scipy",
        "tqdm",
        "einops",
        "hf_transfer",
        "opencv-python-headless",
        "largesteps",
        "spconv-cu122",  # Updated to CUDA 12.2
        "rembg",
        "imageio-ffmpeg",
        "xatlas",
        "pyvista",
        "pymeshfix",
        "igraph",
        "fastapi[standard]==0.115.6",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
        "https://github.com/camenduru/wheels/releases/download/3090/diso-0.1.4-cp310-cp310-linux_x86_64.whl",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # For faster model downloads
            "ATTN_BACKEND": "flash-attn",  # Using flash-attn for better performance
            "SPCONV_ALGO": "native",  # For consistent behavior
        }
    )
    .run_function(clone_repository)
)

app = modal.App(name="example-trellis-3d")


@app.cls(
    image=trellis_image,
    gpu=modal.gpu.L4(count=1),
    timeout=1 * HOURS,
    container_idle_timeout=1 * MINUTES,
)
class Model:
    @modal.enter()
    @modal.build()
    def initialize(self):
        import sys

        sys.path.append(TRELLIS_DIR)

        from trellis.pipelines import TrellisImageTo3DPipeline

        try:
            self.pipe = TrellisImageTo3DPipeline.from_pretrained(MODEL_NAME)
            self.pipe.cuda()
            logger.info("TRELLIS model initialized successfully")
        except Exception as e:
            error_msg = f"Error during model initialization: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def process_image(
        self,
        image_url: str,
        simplify: float,
        texture_size: int,
        sparse_sampling_steps: int,
        sparse_sampling_cfg: float,
        slat_sampling_steps: int,
        slat_sampling_cfg: int,
        seed: int,
        output_format: str,
    ):
        import io

        from PIL import Image

        try:
            response = requests.get(image_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to download image from provided URL",
                )

            image = Image.open(io.BytesIO(response.content))

            logger.info("Starting model inference...")
            outputs = self.pipe.run(
                image,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": sparse_sampling_steps,
                    "cfg_strength": sparse_sampling_cfg,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_sampling_cfg,
                },
            )
            logger.info("Model inference completed successfully")

            if output_format == "glb":
                from trellis.utils import postprocessing_utils

                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=simplify,
                    texture_size=texture_size,
                )

                # Create output directory for temporary files
                output_dir = Path("/tmp/trellis-temp")
                output_dir.mkdir(exist_ok=True, parents=True)
                output_path = output_dir / "temp.glb"

                # Export the mesh to GLB format
                glb.export(str(output_path))

                # Read the GLB file
                glb_bytes = output_path.read_bytes()

                # Clean up temporary file
                output_path.unlink()

                # Return the GLB file bytes
                return Response(
                    content=glb_bytes,
                    media_type="model/gltf-binary",
                    headers={
                        "Content-Disposition": "attachment; filename=output.glb"
                    },
                )

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported output format: {output_format}",
                )

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

    @modal.web_endpoint(method="GET", docs=True)
    async def generate(
        self,
        request: Request,
        image_url: str,
        simplify: float = 0.95,
        texture_size: int = 1024,
        sparse_sampling_steps: int = 12,
        sparse_sampling_cfg: float = 7.5,
        slat_sampling_steps: int = 12,
        slat_sampling_cfg: int = 3,
        seed: int = 42,
        output_format: str = "glb",
    ):
        return self.process_image(
            image_url,
            simplify,
            texture_size,
            sparse_sampling_steps,
            sparse_sampling_cfg,
            slat_sampling_steps,
            slat_sampling_cfg,
            seed,
            output_format,
        )


@app.local_entrypoint()
def main(
    image_url: str = "https://raw.githubusercontent.com/KAIST-Visual-AI-Group/TRELLIS/main/assets/demo_images/00.png",
    simplify: float = 0.02,
    texture_size: int = 1024,
    sparse_sampling_steps: int = 20,
    sparse_sampling_cfg: float = 7.5,
    slat_sampling_steps: int = 20,
    slat_sampling_cfg: int = 7,
    seed: int = 42,
    output_format: str = "glb",
):
    """Generate 3D mesh from input image.

    Args:
        image_url: URL of the input image
        simplify: Mesh simplification factor (0-1)
        texture_size: Size of the output texture
        sparse_sampling_steps: Number of steps for sparse structure sampling
        sparse_sampling_cfg: CFG strength for sparse structure sampling
        slat_sampling_steps: Number of steps for SLAT sampling
        slat_sampling_cfg: CFG strength for SLAT sampling
        seed: Random seed for reproducibility
        output_format: Output format (currently only 'glb' is supported)
    """
    # Create output directory
    output_dir = Path("/tmp/trellis-output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize and run model
    model = Model()
    result = model.process_image(
        image_url=image_url,
        simplify=simplify,
        texture_size=texture_size,
        sparse_sampling_steps=sparse_sampling_steps,
        sparse_sampling_cfg=sparse_sampling_cfg,
        slat_sampling_steps=slat_sampling_steps,
        slat_sampling_cfg=slat_sampling_cfg,
        seed=seed,
        output_format=output_format,
    )

    # Save output file
    output_path = output_dir / "output.glb"
    output_path.write_bytes(result.body)
    print(f"Saved output to {output_path}")
