import time
from pathlib import Path

import modal
from common import app, dataset_volume, model_cache
from utils import write_results

MODEL_NAME = "large-v2"

whisperx_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands(
        "uv pip install --system librosa hf_transfer faster-whisper whisperx torchaudio"
    )
    .apt_install("ffmpeg")
    .entrypoint([])
    .add_local_python_source("common", "utils")
)


with whisperx_image.imports():
    import librosa
    import whisperx


@app.cls(
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=whisperx_image,
)
class WhisperX:
    @modal.enter()
    def load(self):
        device = "cuda"
        self.model = whisperx.load_model(MODEL_NAME, device, compute_type="float16")

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        # Convert string back to Path for local usage
        file_path = Path(file)

        # Get audio duration
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        # Time the transcription
        start_time = time.time()
        audio = whisperx.load_audio(file_path, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.time() - start_time

        transcription = " ".join([s["text"] for s in result["segments"]])
        return file_path.name, transcription, transcription_time, duration


@app.local_entrypoint()
def benchmark_whisperx():
    whisperx = WhisperX()
    # Convert paths to strings for serialization
    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    results = list(whisperx.run.map(files))
    results_path = write_results(results, f"whisperx-{MODEL_NAME}")
    with dataset_volume.batch_upload() as batch:
        batch.put_file(results_path, f"/results/{results_path}")
