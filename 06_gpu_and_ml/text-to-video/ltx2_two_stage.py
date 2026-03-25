# ---
# output-directory: "/tmp/ltx2"
# ---

# # High-quality text-to-video with LTX-2

# [LTX-2](https://github.com/Lightricks/LTX-2) is a 19B-parameter diffusion model
# that generates video with synchronized audio from a text prompt.
# This example runs LTX-2's two-stage production-grade pipeline on Modal: stage 1
# generates video at half resolution, then stage 2 upscales by 2x and refines with
# a distilled LoRA. Output is a 1024x1536 MP4 with audio.

# ## Setup

# The text encoder uses Gemma 3 weights that require accepting a license.
# Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized,
# click "Agree and access repository", then create a token at
# https://huggingface.co/settings/tokens and add it as a
# [Modal Secret](https://modal.com/secrets) named `huggingface-secret`
# with key `HF_TOKEN`.

# Generate a video with:
# ```bash
# modal run ltx2_two_stage.py --prompt "A cathedral made of ice, northern lights overhead"
# ```

# Retrieve the output from the Modal Volume:
# ```bash
# modal volume ls ltx2-outputs
# modal volume get ltx2-outputs <filename>
# ```

# ## Environment setup

import time
from pathlib import Path

import modal

# We pin an LTX-2 commit and install its three subpackages.
# Torch is installed first to ensure compatibility with Flash-Attention 3.

ltx2_commit = "28c3c73"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "transformers>=4.52,<5",
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-core",
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-pipelines",
        f"git+https://github.com/Lightricks/LTX-2.git@{ltx2_commit}#subdirectory=packages/ltx-trainer",
        "https://huggingface.co/alexnasa/flash-attn-3/resolve/main/128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .entrypoint([])
)

# ## Volumes

# Model weights are cached to a Volume at HuggingFace's default cache path.
# Generated videos are saved to a separate output Volume.

model_volume = modal.Volume.from_name("ltx2-models", create_if_missing=True)
output_volume = modal.Volume.from_name("ltx2-outputs", create_if_missing=True)

OUTPUT_DIR = Path("/output-videos")

with image.imports():
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.constants import (
        DEFAULT_AUDIO_GUIDER_PARAMS,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_VIDEO_GUIDER_PARAMS,
    )
    from ltx_pipelines.utils.media_io import encode_video

app = modal.App(
    "example-ltx2-two-stage",
    image=image,
    volumes={
        "/root/.cache/huggingface": model_volume,
        OUTPUT_DIR: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

# ## Inference

NUM_FRAMES = 121  # ~5s at 24 fps
FRAME_RATE = 24
WIDTH = 1536
HEIGHT = 1024


@app.cls(gpu="H200", timeout=30 * 60, scaledown_window=15 * 60)
class LTX2TwoStage:
    @modal.enter()
    def setup(self):
        """Download model weights and initialize the two-stage pipeline."""
        torch.set_float32_matmul_precision("high")

        repo = "Lightricks/LTX-2"
        checkpoint_path = hf_hub_download(repo, "ltx-2-19b-dev.safetensors")
        upsampler_path = hf_hub_download(
            repo, "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        )
        distilled_lora_path = hf_hub_download(
            repo, "ltx-2-19b-distilled-lora-384.safetensors"
        )
        gemma_dir = snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized")
        model_volume.commit()

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ]

        self.tiling_config = TilingConfig.default()
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_dir,
            loras=[],
        )

    @modal.method()
    def generate(self, prompt: str) -> None:
        """Generate a video from a text prompt and save it to the output Volume."""
        print(f"Generating {NUM_FRAMES} frames ({NUM_FRAMES / FRAME_RATE:.0f}s) ...")
        print(f"Prompt: {prompt}")
        start = time.time()

        with torch.no_grad():
            video, audio = self.pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=42,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                frame_rate=FRAME_RATE,
                num_inference_steps=40,
                video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
                audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
                images=[],
                tiling_config=self.tiling_config,
                enhance_prompt=True,
            )
            print(f"Generated in {time.time() - start:.0f}s")

            safe = "".join(c if c.isalnum() or c == " " else "-" for c in prompt)
            filename = f"{int(time.time())}_{safe[:80].strip().replace(' ', '_')}.mp4"
            output_path = OUTPUT_DIR / filename

            encode_video(
                video=video,
                fps=FRAME_RATE,
                audio=audio,
                audio_sample_rate=24_000,
                output_path=str(output_path),
                video_chunks_number=get_video_chunks_number(
                    NUM_FRAMES, self.tiling_config
                ),
            )
        output_volume.commit()
        print(f"Saved to Volume `ltx2-outputs` at {filename}")


@app.local_entrypoint()
def main(
    prompt: str = "A cathedral made of ice, northern lights dancing overhead, camera slowly pushing forward through the nave",
):
    LTX2TwoStage().generate.remote(prompt=prompt)
