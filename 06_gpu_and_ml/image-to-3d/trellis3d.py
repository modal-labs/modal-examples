import logging
import tempfile
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
    # Step 3: Install Kaolin and its dependencies in the correct order
    .pip_install(
        "pygltflib==1.16.1",  # Required by Kaolin for GLTF file handling
        "usd-core==23.11",  # Required by Kaolin for USD file handling
    )
    # Step 4: Install Kaolin after its dependencies
    .pip_install(
        "git+https://github.com/NVIDIAGameWorks/kaolin.git",
        extra_options="--no-deps",  # Install without dependencies since we installed them above
    )
    # Step 5: Install flash-attention separately with CUDA support
    .pip_install(
        "flash-attn==2.6.3",
        extra_options="--no-build-isolation",
    )
    # Step 6: Install the rest of the dependencies
    .pip_install(
        # ML dependencies
        "xformers==0.0.23.post1",
        "safetensors==0.4.1",
        "huggingface-hub==0.20.3",
        # 3D processing dependencies
        "numpy==1.26.3",
        "pillow==10.2.0",
        "imageio==2.33.1",
        "onnxruntime==1.16.3",
        "trimesh==4.0.5",
        "easydict==1.11",
        "scipy==1.11.4",
        "tqdm==4.66.1",
        "einops==0.7.0",
        "hf_transfer==0.1.4",
        "opencv-python-headless==4.9.0.80",
        "largesteps==0.2.2",  # Updated to latest available version
        "spconv-cu118==2.3.6",  # Keep cu118 as it's not yet available for CUDA 12.4
        "rembg==2.0.50",
        "imageio-ffmpeg==0.4.9",
        "xatlas==0.0.8",
        "pyvista==0.42.3",
        "pymeshfix==0.16.2",
        "igraph==0.11.3",
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
        import os

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

                # Create temporary file for GLB output
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_glb:
                    # Export the mesh to GLB format
                    glb.export(temp_glb.name)
                    temp_glb.flush()

                    # Read the exported GLB file
                    glb_bytes = Path(temp_glb.name).read_bytes()

                    # Clean up the temporary file
                    if os.path.exists(temp_glb.name):
                        os.unlink(temp_glb.name)

                    # Return the GLB file bytes
                    return Response(
                        content=glb_bytes,
                        media_type="model/gltf-binary",
                        headers={"Content-Disposition": "attachment; filename=output.glb"},
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
    image_url: str = "https://raw.githubusercontent.com/microsoft/TRELLIS/main/assets/demo/demo_image.jpg",
    output_filename: str = "output.glb",
):
    """Generate a 3D model from an input image.

    Args:
        image_url: URL of the input image
        output_filename: Name of the output GLB file
    """
    # Create temporary directory for outputs
    output_dir = Path("/tmp/trellis3d")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / output_filename

    try:
        # Download and process image
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content

        # Generate 3D model
        model = Model()
        glb_bytes = model.generate.remote(image_bytes)

        # Save output file
        output_path.write_bytes(glb_bytes)
        logger.info(f"Generated 3D model saved to: {output_path}")

        return str(output_path)
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate 3D model: {str(e)}",
        )
