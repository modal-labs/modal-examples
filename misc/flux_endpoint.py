# ---
# lambda-test: false
# ---

# # Serve a fast FLUX.1 [dev] endpoint on Modal

# This example demonstrates how to run a high-performance FLUX.1 image generation endpoint
# on Modal GPUs. FLUX.1 is a state-of-the-art text-to-image model from Black Forest Labs
# that produces high-quality images from text prompts.

# The endpoint supports flexible image generation with various parameters
# and automatically uploads generated images to cloud storage (Cloudflare R2).

# ## Import dependencies and set up paths

# We start by importing the necessary libraries and defining our storage paths.
# We use Modal Volumes for caching model artifacts and Modal CloudBucketMounts for
# storing generated images. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

from __future__ import annotations

from pathlib import Path

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")

# Modal volume for caching compiled model artifacts and other caches across container restarts to reduce cold start times.
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flux_endpoint", create_if_missing=True)

# Configure your Cloudflare R2 bucket details here for image storage
CLOUD_BUCKET_ACCOUNT_ID = "CLOUDFLARE ACCOUNT ID"
CLOUD_BUCKET_NAME = "CLOUDFLARE R2 BUCKET NAME"

# ## Building the container image

# We start with an NVIDIA CUDA base image that includes the necessary GPU drivers
# and development tools.

# Image configuration and setup
cuda_version = "12.6.3"
flavor = "devel"
operating_system = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).entrypoint([])

# We then install all the Python dependencies needed for FLUX.1 inference.

flux_endpoint_image = nvidia_cuda_image.pip_install(
    "accelerate==1.6.0",
    "boto3==1.37.35",
    "diffusers==0.33.1",
    "fastapi[standard]==0.115.12",
    "huggingface-hub[hf_transfer]==0.30.2",
    "numpy==2.2.4",
    "opencv-python-headless==4.11.0.86",
    "para-attn==0.3.32",
    "pydantic==2.11.4",
    "safetensors==0.5.3",
    "sentencepiece==0.2.0",
    "torch==2.7.0",
    "transformers==4.51.3",
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": str(CONTAINER_CACHE_DIR / ".nv_cache"),
        "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".inductor_cache"),
        "TRITON_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".triton_cache"),
    }
)

# ## Creating the Modal app

# We create a Modal App using the defined image and import necessary dependencies
# within the container's runtime environment.

app = modal.App("flux_endpoint", image=flux_endpoint_image)

with flux_endpoint_image.imports():
    import concurrent.futures
    import os
    import time
    import uuid
    from enum import Enum
    from typing import Optional

    import boto3
    import cv2
    import numpy as np
    import torch
    from diffusers import FluxPipeline
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    from pydantic import BaseModel, Field

    # Supported output formats for generated images
    class OutputFormat(Enum):
        PNG = "PNG"
        JPG = "JPG"
        WEBP = "WEBP"

    # ### Defining request/response model

    # We use Pydantic to define a strongly-typed request model. This gives us
    # automatic validation for our API endpoint.

    class InferenceRequest(BaseModel):
        prompt: str
        prompt2: Optional[str] = None
        negative_prompt: Optional[str] = None
        negative_prompt2: Optional[str] = None
        true_cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0, multiple_of=0.1)
        height: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
        width: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
        steps: int = Field(default=28, ge=1, le=50)
        guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0, multiple_of=0.1)
        num_images: int = Field(default=1, ge=1, le=4)
        seed: Optional[int] = None
        output_format: OutputFormat = Field(default=OutputFormat.PNG)
        output_quality: int = Field(default=90, ge=1, le=100)

# ## The FluxService class

# This class handles model loading, optimization, and inference. We use Modal's
# class decorator to control the lifecycle of our cloud container as well as to
# configure auto-scaling parameters, the GPU type, and necessary secrets.


@app.cls(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name(
            "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ),
    ],
    gpu="H100",
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
        CONTAINER_CLOUD_MOUNT_DIR: modal.CloudBucketMount(
            bucket_name=CLOUD_BUCKET_NAME,
            bucket_endpoint_url=f"https://{CLOUD_BUCKET_ACCOUNT_ID}.r2.cloudflarestorage.com",
            secret=modal.Secret.from_name(
                "r2-secret",
                required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
            ),
        ),
    },
    min_containers=1,
    buffer_containers=0,
    scaledown_window=300,  # 5 minutes
    timeout=3600,  # 1 hour
    enable_memory_snapshot=True,
)
class FluxService:
    # ## Model optimization methods

    # These methods apply various optimizations to make model inference faster.
    # The main optimizations are first block cache and torch compile.

    def _optimize(self):
        # apply first block cache, see: [ParaAttention](https://github.com/chengzeyi/ParaAttention)
        apply_cache_on_pipe(
            self.pipe,
            residual_diff_threshold=0.12,  # don't recommend going higher
        )

        # fuse qkv projections
        self.pipe.transformer.fuse_qkv_projections()
        self.pipe.vae.fuse_qkv_projections()

        # use channels last memory format
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        # torch compile configs
        config = torch._inductor.config
        config.conv_1x1_as_mm = True
        config.coordinate_descent_check_all_directions = True
        config.coordinate_descent_tuning = True
        config.disable_progress = False
        config.epilogue_fusion = False
        config.shape_padding = True

        # mark layers for compilation with dynamic shapes enabled
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
        )

        self.pipe.vae.decode = torch.compile(
            self.pipe.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=True
        )

    def _compile(self):
        # monkey-patch torch inductor remove_noop_ops pass for para-attn dynamic compilation
        # swallow AttributeError: 'SymFloat' object has no attribute 'size' and return false
        from torch._inductor.fx_passes import post_grad

        if not hasattr(post_grad, "_orig_same_meta"):
            post_grad._orig_same_meta = post_grad.same_meta

            def _safe_same_meta(node1, node2):
                try:
                    return post_grad._orig_same_meta(node1, node2)
                except AttributeError as e:
                    if "SymFloat" in str(e) and "size" in str(e):
                        # return not the same, instead of crashing
                        return False
                    raise

            post_grad.same_meta = _safe_same_meta

        print("triggering torch compile")
        self.pipe("dummy prompt", height=1024, width=1024, num_images_per_prompt=1)

        # comment this out if you only need num_images_per_prompt=1
        print("recompiling for dynamic batch size")
        self.pipe("dummy prompt", height=1024, width=1024, num_images_per_prompt=2)

    # ## Mega-cache management

    # PyTorch "mega-cache" serializes compiled model artifacts into a blob that
    # can be easily transferred to another machine with the same GPU.

    def _load_mega_cache(self):
        print("loading torch mega-cache")
        try:
            if self.mega_cache_bin_path.exists():
                with open(self.mega_cache_bin_path, "rb") as f:
                    artifact_bytes = f.read()

                if artifact_bytes:
                    torch.compiler.load_cache_artifacts(artifact_bytes)
            else:
                print("torch mega cache not found, regenerating...")
        except Exception as e:
            print(f"error loading torch mega-cache: {e}")

    def _save_mega_cache(self):
        print("saving torch mega-cache")
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            artifact_bytes, _ = artifacts

            with open(self.mega_cache_bin_path, "wb") as f:
                f.write(artifact_bytes)

            # persist changes to volume
            CONTAINER_CACHE_VOLUME.commit()
        except Exception as e:
            print(f"error saving torch mega-cache: {e}")

    # ## Memory Snapshotting

    # We utilize memory snapshotting to avoid reloading model weights into host memory
    # during subsequent container starts.

    @modal.enter(snap=True)
    def load(self):
        print("downloading (if necessary) and loading model")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cpu")

        # Set up mega cache paths
        mega_cache_dir = CONTAINER_CACHE_DIR / ".mega_cache"
        mega_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mega_cache_bin_path = mega_cache_dir / "flux_torch_mega"

    @modal.enter(snap=False)
    def setup(self):
        self.pipe.to("cuda")

        self._load_mega_cache()
        self._optimize()
        self._compile()
        self._save_mega_cache()

        # Initialize S3 client for R2 storage
        try:
            self.s3_client = boto3.client(
                service_name="s3",
                endpoint_url=f"https://{CLOUD_BUCKET_ACCOUNT_ID}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        except Exception as e:
            print(f"Error initiating s3 client: {e}")
            raise

    # ## The main inference endpoint

    # This method handles incoming requests, generates images, and uploads them
    # to cloud storage.

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        generator = (
            torch.Generator("cuda").manual_seed(request.seed)
            if request.seed is not None
            else None
        )

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate images using the FLUX pipeline
        images = self.pipe(
            prompt=request.prompt,
            prompt_2=request.prompt2,
            negative_prompt=request.negative_prompt,
            negative_prompt_2=request.negative_prompt2,
            true_cfg_scale=request.true_cfg_scale,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images,
            generator=generator,
            output_type="np",
        ).images

        torch.cuda.synchronize()
        print(f"inference time: {time.perf_counter() - t0:.2f}s")
        t1 = time.perf_counter()

        # Process and upload images to cloud storage
        image_urls = []
        CONTAINER_CLOUD_MOUNT_DIR.mkdir(parents=True, exist_ok=True)

        # image processing
        def process_image(image):
            # Generate unique filename
            filename = str(uuid.uuid4())
            filename_with_ext = f"{filename}.{request.output_format.value.lower()}"
            output_path = CONTAINER_CLOUD_MOUNT_DIR / filename_with_ext

            # Convert to uint8 and BGR format for OpenCV
            image_np = (image * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Set encoding parameters based on format
            match request.output_format:
                case OutputFormat.JPG:
                    params = [cv2.IMWRITE_JPEG_QUALITY, request.output_quality]
                case OutputFormat.WEBP:
                    params = [cv2.IMWRITE_WEBP_QUALITY, request.output_quality]
                case _:
                    params = []

            # Save image using OpenCV
            cv2.imwrite(str(output_path), image_bgr, params)

            # Generate a signed URL for the uploaded image
            # This allows clients to download the image directly from R2
            signed_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": CLOUD_BUCKET_NAME, "Key": filename_with_ext},
                ExpiresIn=86400,  # 24 hour expiry
            )
            return signed_url

        # process images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            image_urls = list(executor.map(process_image, images))

        torch.cuda.synchronize()
        print(f"image processing and cloud save time: {time.perf_counter() - t1:.2f}s")
        return image_urls
