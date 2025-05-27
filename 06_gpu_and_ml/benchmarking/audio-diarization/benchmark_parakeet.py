import time
from pathlib import Path

import modal
from common import app, dataset_volume, model_cache
from utils import write_results

parakeet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .run_commands(
        "uv pip install --system librosa==0.11.0 hf_transfer huggingface_hub[hf-xet] nemo_toolkit[asr] cuda-python>=12.3",
        "uv pip install --system 'numpy<2.0'",  # downgrade numpy; incompatible current version
    )
    .entrypoint([])
    .add_local_python_source("common", "utils")
)


with parakeet_image.imports():
    import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    gpu="a10g",
    image=parakeet_image,
)
class Parakeet:
    @modal.enter()
    def load(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        import librosa

        # Convert string back to Path for local usage
        file_path = Path(file)

        # Get audio duration
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

        transcription = output[0].text
        return file_path.name, transcription, transcription_time, duration


@app.local_entrypoint()
def benchmark_parakeet():
    parakeet = Parakeet()
    # Convert paths to strings for serialization
    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    results = list(parakeet.run.map(files))

    results_path = write_results(results, MODEL_NAME.replace("/", "-"))
    with dataset_volume.batch_upload() as batch:
        batch.put_file(results_path, f"/results/{results_path}")
