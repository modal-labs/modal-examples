"This example originally contributed by @sandeeppatra96 and @patraxo on GitHub"
import logging
import tempfile
import traceback

import modal
import requests
from fastapi import HTTPException, Request, Response, status

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


# The specific version of torch==2.4.0 to circumvent the flash attention wheel build error

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
    )
    .pip_install(
        # Core dependencies
        "torch==2.4.0",
        "packaging==23.2",
        "ninja==1.11.1",
        "wheel==0.42.0",
        "setuptools==69.0.3",
        # ML dependencies
        "flash-attn==2.6.3",
        "xformers==0.0.23.post1",
        # 3D processing dependencies
        "numpy==1.26.3",
        "pillow==10.2.0",
        "imageio==2.33.1",
        "onnxruntime==1.16.3",
        "trimesh==4.0.5",
        "safetensors==0.4.1",
        "easydict==1.11",
        "scipy==1.11.4",
        "tqdm==4.66.1",
        "einops==0.7.0",
        "hf_transfer==0.1.4",
        "opencv-python-headless==4.9.0.80",
        "largesteps==0.3.0",
        "spconv-cu118==2.3.6",  # Keep cu118 as it's not yet available for CUDA 12.4
        "rembg==2.0.50",
        "torchvision==0.15.2",
        "imageio-ffmpeg==0.4.9",
        "xatlas==0.0.8",
        "pyvista==0.42.3",
        "pymeshfix==0.16.2",
        "igraph==0.11.3",
        "huggingface-hub==0.20.3",
        "fastapi[standard]==0.115.6",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "git+https://github.com/NVIDIAGameWorks/kaolin.git",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
        "https://github.com/camenduru/wheels/releases/download/3090/diso-0.1.4-cp310-cp310-linux_x86_64.whl",
        "https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
        extra_options="--no-build-isolation",  # Required for flash-attn
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # For faster model downloads
            "ATTN_BACKEND": "flash-attn",  # Using flash-attn for better performance
            "SPCONV_ALGO": "native",  # For consistent behavior
        }
    )
    .entrypoint([])
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

                temp_glb = tempfile.NamedTemporaryFile(
                    suffix=".glb", delete=False
                )
                temp_path = temp_glb.name
                logger.info(f"Exporting mesh to: {temp_path}")
                glb.export(temp_path)
                temp_glb.close()

                try:
                    with open(temp_path, "rb") as file:
                        content = file.read()
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logger.info("Temp file cleaned up")
                        return Response(
                            content=content,
                            media_type="model/gltf-binary",
                            headers={
                                "Content-Disposition": "attachment; filename=output.glb",
                            },
                        )
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e

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

    @modal.method()
    def test(self):
        """Test method for CI to verify the example works."""
        test_url = "https://raw.githubusercontent.com/microsoft/TRELLIS/main/assets/examples/chair.png"
        return self.process_image(
            image_url=test_url,
            simplify=0.95,
            texture_size=512,
            sparse_sampling_steps=2,
            sparse_sampling_cfg=7.5,
            slat_sampling_steps=2,
            slat_sampling_cfg=3,
            seed=42,
            output_format="glb",
        )


@app.command()
def main():
    """Command for testing the example."""
    model = Model()
    model.test.remote()


if __name__ == "__main__":
    main()
