# ---
# output-directory: "/tmp/ltx2"
# ---

# # High-quality video generation with LTX-2 Two-Stage Pipeline (Flash-Attn-3 Test)

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
# modal run ltx2_two_stage_fa3.py --prompt "A serene mountain landscape at sunrise"
#
# # Image-to-video
# modal run ltx2_two_stage_fa3.py \
#   --prompt "Mountains come alive at sunrise" \
#   --image-path /path/to/mountain.jpg
# ```

# Models (~20GB total) will be cached in a Modal Volume after first download.

# ## Setup

import time
from pathlib import Path
from typing import Optional

import modal

# ### Container image

# Install LTX-2 packages from GitHub with Flash-Attention 3 for testing.
# Flash-Attn-3 is a beta release that may provide performance improvements.
# Install torch before LTX-2 to ensure flash-attn-3 compatibility
HF_CACHE_PATH = "/cache"
ltx2_commit = "727c43e998776af554dc502c744ed74b4ba34702"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-core",
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-pipelines",
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-trainer",
        "https://huggingface.co/alexnasa/flash-attn-3/resolve/main/128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",  # Quiet `use_fast` suggestions inside LTX-2
        }
    )
    .entrypoint([])
)

# ## Model storage

# We need to download several model files from Hugging Face:
# - Base checkpoint (distilled fp8 version for efficiency)
# - Spatial upsampler (for 2x resolution upscaling)
# - Distilled LoRA (required for two-stage pipeline)
# - Gemma text encoder (automatically downloaded)
# All of these are cached to `model_volume`, which we mount in the container
# where HuggingFace normally looks for a cache (`/root/.cache/huggingface`).
# We also make a separate volume for output videos. We'll automatically
# save the videos to this volume, and also download them locally.

model_volume = modal.Volume.from_name("ltx2-models", create_if_missing=True)

OUTPUT_VOLUME = "ltx2-outputs"
OUTPUT_PATH = Path("/output-videos")
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


with image.imports():
    # For inference_mode
    import torch

    # Hugging Face
    from huggingface_hub import hf_hub_download, snapshot_download

    # Models
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils import helpers as ltx_helpers

    # For saving LTX-2 video iterators to a video file
    from ltx_pipelines.utils.media_io import encode_video

app = modal.App(
    "example-ltx2-two-stage-fa3",
    image=image,
    volumes={"/root/.cache/huggingface": model_volume, OUTPUT_PATH: output_volume},
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)


# ## LTX-2 Two-Stage Pipeline

MINUTES = 60


@app.cls(
    gpu="H100",
    timeout=30 * MINUTES,
    scaledown_window=15 * MINUTES,
)
class LTX2TwoStage:
    @modal.enter()
    def setup(self):
        """Initialize the two-stage pipeline."""

        print("🎬 Loading LTX-2 Two-Stage Pipeline...")
        st = time.perf_counter()

        # Download models (cached automatically to volume)
        ltx2_repo = "Lightricks/LTX-2"
        checkpoint_path = hf_hub_download(
            repo_id=ltx2_repo,
            filename="ltx-2-19b-distilled-fp8.safetensors",
        )
        upsampler_path = hf_hub_download(
            repo_id=ltx2_repo,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )
        distilled_lora_path = hf_hub_download(
            repo_id=ltx2_repo,
            filename="ltx-2-19b-distilled-lora-384.safetensors",
        )
        gemma_dir = snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
        )

        # Commit downloads to volume
        model_volume.commit()

        # Setup distilled LoRA
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                0.6,  # LoRA strength
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        # Initialize pipeline
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_dir,
            loras=[],
            fp8transformer=True,  # Use FP8 for efficiency
        )

        # Disable memory cleanup between stages for faster inference
        ltx_helpers.cleanup_memory = lambda: None  # Replace with no-op function

        # Warm up the pipeline to load model shards into memory
        print("🔥 Warming up pipeline (loading model shards)...")
        try:
            # Trigger model loading with minimal generation
            with torch.no_grad():
                video_iter, _ = self.pipeline(
                    prompt="warmup",
                    negative_prompt="",
                    seed=42,
                    height=256,
                    width=256,
                    num_frames=9,  # Minimal frames
                    frame_rate=8.0,
                    num_inference_steps=1,  # Just 1 step to trigger loading
                    cfg_guidance_scale=1.0,
                    images=[],
                )
                # Consume the iterator to trigger actual inference
                _ = list(video_iter)
            print("🔥 Warmup complete - models loaded on GPU")
        except Exception as e:
            print(f"⚠️  Warmup error (may be expected): {e}")

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
            image_path: Optional path to input image for image-to-video (used as first frame)
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
        # Generate video - pipeline returns (video_iterator, audio_tensor)
        # Note: video is an Iterator[torch.Tensor] that yields decoded chunks for memory efficiency
        with torch.no_grad():
            video_iterator, audio_tensor = self.pipeline(
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

            encode_video(
                video=video_iterator,
                fps=frame_rate,
                audio=audio_tensor,
                audio_sample_rate=24000,  # LTX-2 audio sample rate
                output_path=str(output_file),
                video_chunks_number=1,  # Single chunk for simple use case
            )

        duration = time.time() - start
        print(f"⚡ Generated in {duration:.1f}s")

        output_volume.commit()
        print(f"💾 Saved to {output_file.name}")

        return output_file.name


# ## CLI interface

# Run with:
# ```bash
# modal run ltx2_two_stage_fa3.py --prompt "A serene mountain landscape at sunrise"
# ```

# For image-to-video:
# ```bash
# modal run ltx2_two_stage_fa3.py \
#   --prompt "The mountain landscape comes alive with morning light" \
#   --image-path /path/to/image.jpg
# ```


@app.local_entrypoint()
def main(
    prompt: str = "A serene mountain landscape with snow-capped peaks, rolling clouds, and golden sunlight; a chickadee swoops into frame whistling a dark and ominous tune",
    image_path: Optional[str] = None,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_frames: int = 121,
    num_inference_steps: int = 40,
    width: int = 768,
    height: int = 512,
    seed: int = 42,
    twice: bool = True,
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
