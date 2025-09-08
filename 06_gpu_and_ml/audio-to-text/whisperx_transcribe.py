# ---
# lambda-test: false  # requires audio file
# ---
# # WhisperX transcription with word-level timestamps
#
# This example shows how to run [WhisperX](https://github.com/m-bain/whisperX) on
# Modal for accurate, word-level timestamped transcription.
#
# We’ll walk through the following steps:
#
# 1. Defining the container image with CUDA 12.8, cuDNN 8, FFmpeg and Python deps.
# 2. Persisting model weights to a [Modal Volume](https://modal.com/docs/reference/modal.Volume).
# 3. A [Modal Cls](https://modal.com/docs/reference/modal.App#cls) that loads WhisperX once per GPU instance.
# 4. A [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint) that uploads an audio file to the service.
#
# ## Defining image
#
# We start from NVIDIA’s official CUDA 12.8 devel image, add cuDNN, FFmpeg, and
# install the WhisperX Python package plus its numerical deps.
#

import os
import tempfile
from typing import Dict

import modal

MODEL_CACHE_DIR = "/whisperx-cache"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # ── System deps ─────────────────────────────────────────────────────────────
    .apt_install("ffmpeg")  # audio decoding / resampling
    .apt_install("libcudnn8")  # cuDNN runtime
    .apt_install("libcudnn8-dev")  # cuDNN headers (needed by torch wheels)
    # ── Python deps ─────────────────────────────────────────────────────────────
    .pip_install(
        "whisperx==3.4.0",  # our ASR library
        "numpy==2.0.2",
        "scipy==1.15.0",
    )
    # Tell HF & Torch to cache inside our Volume
    .env({"HF_HOME": MODEL_CACHE_DIR})
    .env({"TORCH_HOME": MODEL_CACHE_DIR})
)

# ## Defining the app
#
# Downloaded weights live in a [Modal Volume](https://modal.com/docs/reference/modal.Volume) so subsequent runs reuse them.
app = modal.App("example-whisperx-transcribe", image=image)
models_volume = modal.Volume.from_name("whisperx-models", create_if_missing=True)


# ## Defining the inference service
#
# We wrap WhisperX inference in a Modal Cls.
# A single GPU container can serve multiple concurrent requests.
@app.cls(
    gpu="H100",
    image=image,
    volumes={MODEL_CACHE_DIR: models_volume},
    timeout=30 * 60,
)
class WhisperX:
    """Serverless WhisperX service running on a single GPU."""

    @modal.enter()
    def setup(self):
        print("🔄 Loading WhisperX model …")
        import whisperx

        self.model = whisperx.load_model(
            "large-v2",
            device="cuda",
            compute_type="float16",
            download_root=MODEL_CACHE_DIR,
        )
        print("✅ Model ready!")

    @modal.method()
    def transcribe(self, audio_data: bytes) -> Dict:
        """
        Transcribe an audio file passed in as raw bytes.
        Returns language, per-word segments, and total duration.
        """

        import whisperx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        try:
            audio = whisperx.load_audio(temp_audio_path)
            result = self.model.transcribe(audio, batch_size=16, language="en")

            language = result.get("language", "en")

            if result["segments"]:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self.device,
                        model_dir=MODEL_CACHE_DIR,
                    )
                    result = whisperx.align(
                        result["segments"], align_model, metadata, audio, self.device
                    )
                except Exception as e:
                    print(f"⚠️ Alignment failed: {e} — falling back to segment-level")

            return {
                "language": language,
                "segments": result["segments"],
                "duration": len(audio) / 16_000,  # audio is 16 kHz
            }

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


# ## Command-line usage
#
# We expose a [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint)
# so you can run:
# - using a local audio file
# - using a link to an audio file
#
# ```bash
# modal run whisperx_transcribe.py --audio-file audio.wav # uses a local audio file
# modal run whisperx_transcribe.py --audio-link https://example.com/audio.wav # uses a link to an audio file
# modal run whisperx_transcribe.py # uses a default public audio file
# ```
#
@app.local_entrypoint()
def main(
    audio_file: str = None,
    audio_link: str = None,
):
    import json
    import time

    import requests

    if not audio_file and not audio_link:
        print("No audio file or link provided, using default link")
        audio_link = "https://modal-public-assets.s3.us-east-1.amazonaws.com/erik.wav"

    if audio_file:
        print(f"🔊 Reading {audio_file} …")
        with open(audio_file, "rb") as f:
            audio_data = f.read()
    elif audio_link:
        print(f"🔊 Reading {audio_link} …")
        audio_data = requests.get(audio_link).content

    transcriber = WhisperX()

    print("📝 Transcribing …")
    start = time.time()
    result = transcriber.transcribe.remote(audio_data)
    duration = time.time() - start

    print(f"\n🌐 Detected language: {result['language']}")
    print(f"⏱️  Audio duration:   {result['duration']:.2f} s")
    print(f"🚀 Time taken:        {duration:.2f} s")

    with open("transcription.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n💾 Saved transcription → transcription.json")
