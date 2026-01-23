# ---
# output-directory: "/tmp/ltx2"
# ---

# # High-quality text-to-video generation with LTX-2

# This example demonstrates [LTX-2](https://github.com/Lightricks/LTX-2)'s two-stage pipeline
# for production-quality text-to-video generation. The two-stage approach generates video at lower resolution first,
# then upscales for higher quality output.

# ## Setup

# Accept the Gemma license on HuggingFace by visiting
# https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
# and clicking "Agree and access repository". Then create a HuggingFace token at
# https://huggingface.co/settings/tokens and add it to Modal at
# https://modal.com/secrets/create?name=huggingface-secret with key name `HF_TOKEN`.

# Run the example in an ephemeral app with:
# ```bash
# modal run ltx2_two_stage.py --prompt "A serene mountain landscape at sunrise"
# ```

# ## Environment setup
import time
from dataclasses import dataclass
from pathlib import Path

import modal

# ## Video output configuration


@dataclass
class VideoOutput:
    """Video output configuration."""

    num_frames: int = 121  # ~5 seconds at 25fps
    width: int = 768
    height: int = 512


# ## Container image

# We install torch by itself first to ensure compatibility with Flash-Attention 3.
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

# ## Model storage and video output settings

# Models are cached to a volume mounted at HuggingFace's default cache location.
# Output videos are saved to a separate volume.
model_volume = modal.Volume.from_name("ltx2-models", create_if_missing=True)
output_volume = modal.Volume.from_name("ltx2-outputs", create_if_missing=True)

OUTPUT_MNT = Path("/output-videos")


with image.imports():
    # For inference_mode
    import torch

    # Hugging Face
    from huggingface_hub import hf_hub_download, snapshot_download

    # Models
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.registry import StateDictRegistry
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    # For saving LTX-2 video iterators to a video file
    from ltx_pipelines.utils.media_io import encode_video

app = modal.App(
    "example-ltx2-two-stage",
    image=image,
    volumes={"/root/.cache/huggingface": model_volume, OUTPUT_MNT: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

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
        LTX2_REPO = "Lightricks/LTX-2"
        checkpoint_path = hf_hub_download(
            repo_id=LTX2_REPO,
            filename="ltx-2-19b-distilled-fp8.safetensors",
        )
        upsampler_path = hf_hub_download(
            repo_id=LTX2_REPO,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )
        distilled_lora_path = hf_hub_download(
            repo_id=LTX2_REPO,
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
            fp8transformer=True,
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
                    negative_prompt="cooldown",
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

    def prompt_to_filename(self, prompt: str) -> Path:
        safe_prompt = "".join(c if c.isalnum() or c.isspace() else "-" for c in prompt)
        safe_prompt = safe_prompt[:100].strip().replace(" ", "_")
        return OUTPUT_MNT / f"{int(time.time())}_{safe_prompt}.mp4"

    @modal.method()
    def generate(
        self,
        prompt: str,
        output: VideoOutput = VideoOutput(),
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    ) -> str:
        """Generate video from text prompt using two-stage pipeline."""
        frame_rate = 25.0
        print(
            f"🎬 Generating {output.num_frames} frames ({output.num_frames / frame_rate:.1f}s)"
        )
        print(f"📝 Prompt: {prompt}")

        output_file = self.prompt_to_filename(prompt)
        start = time.time()

        # Generate video - pipeline returns (video_iterator, audio_tensor)
        with torch.no_grad():
            video_iterator, audio_tensor = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=42,
                height=output.height,
                width=output.width,
                num_frames=output.num_frames,
                frame_rate=frame_rate,
                num_inference_steps=40,
                cfg_guidance_scale=3.0,
                images=[],
            )
            # This saves the video to the output volume
            encode_video(
                video=video_iterator,
                fps=frame_rate,
                audio=audio_tensor,
                audio_sample_rate=24000,  # LTX-2 audio sample rate
                output_path=str(output_file),
                video_chunks_number=1,
            )

        duration = time.time() - start
        print(f"⚡ Generated in {duration:.1f}s")

        output_volume.commit()
        print(f"💾 Saved to {output_file.name}")

        return output_file.name, duration


@app.local_entrypoint()
def main(
    prompt: str = "A busy NYC afternoon, view from across the street toward the glorious Modal HQ building",
    twice: bool = True,
):
    """Generate video from text prompt."""
    service = LTX2TwoStage()

    def run():
        filename, duration = service.generate.remote(prompt=prompt)

        # Download locally
        local_dir = Path("/tmp/ltx2")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / filename

        local_path.write_bytes(b"".join(output_volume.read_file(filename)))
        print(f"📥 Downloaded to: {local_path}\n")

        return duration

    first_duration = run()

    if twice:
        print("=" * 60)
        print("🔥 Running again on warm container...")
        print("=" * 60 + "\n")
        second_duration = run()
        speedup = first_duration / second_duration
        print(f"🚀 Warm container was {speedup:.1f}x faster!")
