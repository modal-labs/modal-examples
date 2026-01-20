# ---
# output-directory: "/tmp/ltx2"
# ---

# # High-quality video generation with LTX-2 Two-Stage Pipeline

# This example demonstrates [LTX-2](https://github.com/Lightricks/LTX-2)'s production-quality
# two-stage pipeline for text-to-video and image-to-video generation.

# The two-stage approach generates video at a lower resolution first, then upscales it
# for higher quality output - perfect for production use cases.

# ## First-time setup

# 1. Accept the Gemma license on HuggingFace:
#    Visit https://huggingface.co/google/gemma-3-9b-it and click "Agree and access repository"

# 2. Create a HuggingFace token and add it to Modal:
#    - Go to https://huggingface.co/settings/tokens
#    - Create a token with "Read" access
#    - Add to Modal: https://modal.com/secrets/create?name=huggingface-secret
#    - Use key name: `HUGGING_FACE_HUB_TOKEN`

# 3. Run the example - models will auto-download on first run:

# ```bash
# # Text-to-video
# modal run ltx2_two_stage.py --prompt "A serene mountain landscape at sunrise"
#
# # Image-to-video
# modal run ltx2_two_stage.py \
#   --prompt "Mountains come alive at sunrise" \
#   --image-path /path/to/mountain.jpg
# ```

# Models (~20GB total) will be cached in a Modal Volume after first download.

# ## Setup

import time
from pathlib import Path
from typing import Optional

import modal

app = modal.App("example-ltx2-two-stage")

# ### Container image

# Install LTX-2 packages from GitHub along with dependencies.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "accelerate==1.6.0",
        "transformers==4.51.3",
        "diffusers==0.33.1",
        "sentencepiece==0.2.0",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
        "pillow==11.1.0",
        # "safetensors==0.5.7",÷
        # Install LTX-2 packages
        "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core",
        "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ## Model storage

# We need to download several model files from Hugging Face:
# - Base checkpoint (distilled fp8 version for efficiency)
# - Spatial upsampler (for 2x resolution upscaling)
# - Distilled LoRA (required for two-stage pipeline)
# - Gemma text encoder (automatically downloaded)

# To download the required LTX-2 model files, run:
# ```bash
# huggingface-cli download Lightricks/LTX-2 \
#   ltx-2-19b-distilled-fp8.safetensors \
#   ltx-2-spatial-upscaler-x2-1.0.safetensors \
#   ltx-2-19b-distilled-lora-384.safetensors \
#   --local-dir /path/to/models/ltx2
# ```
# Then upload to Modal Volume:
# ```bash
# modal volume put ltx2-models /path/to/models/ltx2 /models/ltx2
# ```

# Alternatively, you can use the Python API to download them programmatically.

MODEL_VOLUME = "ltx2-models"
model_volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
MODEL_PATH = Path("/models")

OUTPUT_VOLUME = "ltx2-outputs"
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
OUTPUT_PATH = Path("/outputs")

MINUTES = 60

# Model file paths on HuggingFace
HF_REPO = "Lightricks/LTX-2"
CHECKPOINT_FILE = "ltx-2-19b-distilled-fp8.safetensors"
UPSAMPLER_FILE = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA_FILE = "ltx-2-19b-distilled-lora-384.safetensors"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

with image.imports():
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


# ## LTX-2 Two-Stage Pipeline


@app.cls(
    image=image,
    gpu="H100",
    timeout=30 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={
        MODEL_PATH: model_volume,
        OUTPUT_PATH: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTX2TwoStage:
    @modal.enter()
    def setup(self):
        """Initialize the two-stage pipeline."""
        from huggingface_hub import hf_hub_download, snapshot_download

        print("🎬 Loading LTX-2 Two-Stage Pipeline...")
        st = time.perf_counter()

        # Model paths
        checkpoint_path = MODEL_PATH / "ltx2" / CHECKPOINT_FILE
        upsampler_path = MODEL_PATH / "ltx2" / UPSAMPLER_FILE
        distilled_lora_path = MODEL_PATH / "ltx2" / DISTILLED_LORA_FILE
        gemma_path = MODEL_PATH / "gemma"

        # Create directories
        (MODEL_PATH / "ltx2").mkdir(parents=True, exist_ok=True)

        # Download models if not cached
        if not checkpoint_path.exists():
            print(f"📥 Downloading {CHECKPOINT_FILE}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=CHECKPOINT_FILE,
                local_dir=str(MODEL_PATH / "ltx2"),
            )

        if not upsampler_path.exists():
            print(f"📥 Downloading {UPSAMPLER_FILE}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=UPSAMPLER_FILE,
                local_dir=str(MODEL_PATH / "ltx2"),
            )

        if not distilled_lora_path.exists():
            print(f"📥 Downloading {DISTILLED_LORA_FILE}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=DISTILLED_LORA_FILE,
                local_dir=str(MODEL_PATH / "ltx2"),
            )

        if True:  # not gemma_path.exists():
            print("📥 Downloading Gemma text encoder...")
            snapshot_download(
                repo_id=GEMMA_REPO,
                local_dir=str(gemma_path),
            )

        # Commit downloads to volume
        model_volume.commit()

        # Setup distilled LoRA
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                str(distilled_lora_path),
                0.6,  # LoRA strength
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        # Initialize pipeline
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=str(checkpoint_path),
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(upsampler_path),
            gemma_root=str(gemma_path),
            loras=[],
            fp8transformer=True,  # Use FP8 for efficiency
        )

        # Warm up the pipeline to load model shards into memory
        print("🔥 Warming up pipeline (loading model shards)...")
        warmup_start = time.perf_counter()
        try:
            # Trigger model loading with minimal generation
            self.pipeline(
                prompt="warmup",
                negative_prompt="",
                seed=42,
                height=256,
                width=256,
                num_frames=9,  # Minimal frames
                frame_rate=8.0,
                num_inference_steps=1,  # Just 1 step to trigger loading
                cfg_guidance_scale=1.0,
                output_path="/tmp/warmup.mp4",
            )
            # Clean up warmup output
            import os

            if os.path.exists("/tmp/warmup.mp4"):
                os.remove("/tmp/warmup.mp4")
        except Exception as e:
            print(f"⚠️  Warmup generated an error (expected): {e}")

        print(f"✅ Pipeline ready in {time.perf_counter() - st:.2f}s")

    def prompt_to_filename(self, prompt: str, output_dir: Path) -> Path:
        safe_prompt = "".join(c if c.isalnum() or c.isspace() else "-" for c in prompt)
        safe_prompt = safe_prompt[:100].strip().replace(" ", "_")
        return output_dir / f"{int(time.time())}_{safe_prompt}.mp4"

    @modal.method()
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 25.0,
        num_inference_steps: int = 40,
        cfg_guidance_scale: float = 3.0,
        seed: int = 42,
    ) -> str:
        """
        Generate video using two-stage pipeline.

        Args:
            prompt: Text description of the video
            image_path: Optional path to input image for image-to-video
            negative_prompt: Things to avoid
            height: Video height
            width: Video width
            num_frames: Number of frames (~5 seconds at 25fps for 121 frames)
            frame_rate: Output frame rate
            num_inference_steps: Denoising steps (higher = better quality)
            cfg_guidance_scale: Prompt adherence (3-5 typical)
            seed: Random seed for reproducibility

        Returns:
            Filename of generated video
        """
        print(f"🎬 Generating {num_frames} frames ({num_frames / frame_rate:.1f}s)")
        print(f"📝 Prompt: {prompt}")

        # Setup images list if image_path provided
        images = []
        if image_path:
            print(f"🖼️  Using input image: {image_path}")
            images = [(image_path, 0, 1.0)]  # Image at frame 0, strength 1.0

        # Create output filename
        output_file = self.prompt_to_filename(prompt, OUTPUT_PATH)
        start = time.time()

        # Generate video
        self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
            images=images,
        )

        duration = time.time() - start
        print(f"⚡ Generated in {duration:.1f}s")

        output_volume.commit()
        print(f"💾 Saved to {output_file.name}")

        return output_file.name


# ## CLI interface

# Run with:
# ```bash
# modal run ltx2_two_stage.py --prompt "A serene mountain landscape at sunrise"
# ```

# For image-to-video:
# ```bash
# modal run ltx2_two_stage.py \
#   --prompt "The mountain landscape comes alive with morning light" \
#   --image-path /path/to/image.jpg
# ```


@app.local_entrypoint()
def main(
    prompt: str = "A serene mountain landscape with snow-capped peaks, rolling clouds, and golden sunlight",
    image_path: Optional[str] = None,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_frames: int = 121,
    num_inference_steps: int = 40,
    width: int = 768,
    height: int = 512,
    seed: int = 42,
    twice: bool = False,
):
    """Generate high-quality video with LTX-2 two-stage pipeline."""

    # Handle image path if provided
    if image_path:
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        # For now, just use the path - in production you'd upload to volume
        print(f"🖼️  Image-to-video mode with: {image_path}")

    service = LTX2TwoStage()

    def run():
        start = time.time()
        filename = service.generate.remote(
            prompt=prompt,
            image_path=image_path,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        duration = time.time() - start

        print(f"\n⚡ Total time: {duration:.1f}s")
        print(f"💾 Video saved to Modal Volume: {filename}")

        # Download locally
        local_dir = Path("/tmp/ltx2")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / filename

        local_path.write_bytes(b"".join(output_volume.read_file(filename)))
        print(f"📥 Downloaded to: {local_path}\n")

        return duration

    # First run
    first_duration = run()

    if twice:
        print("=" * 60)
        print("🔥 Running again on warm container...")
        print("=" * 60 + "\n")
        second_duration = run()
        speedup = first_duration / second_duration
        print(f"🚀 Warm container was {speedup:.1f}x faster!")


# ## Tips

# **Two-stage benefits:**
# - Higher quality output through upscaling
# - Better temporal coherence
# - Production-ready results

# **Performance:**
# - Uses FP8 for lower memory footprint
# - H100 recommended
# - ~40 steps for good quality (can use less for faster generation)

# **Image-to-video:**
# - Provide an image at frame 0 to guide generation
# - Adjust strength (0-1) to control how much the image influences output
# - Great for animating still images or concept art
