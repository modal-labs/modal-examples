import time
from pathlib import Path

import modal
from common import app, dataset_volume, model_cache
from utils import write_results

MODEL_NAME = "openai/whisper-large-v3-turbo"

whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands("uv pip install --system librosa hf_transfer vllm[audio]")
    .entrypoint([])
    .add_local_python_source("common", "utils")
)


with whisper_image.imports():
    import librosa
    from vllm import LLM, SamplingParams


@app.cls(
    gpu="a10g",
    volumes={
        "/cache": model_cache,
        "/data": dataset_volume,
    },
    image=whisper_image,
)
class Whisper:
    @modal.enter()
    def load(self):
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )

    @modal.method()
    def run(self, file: str):
        # Convert string back to Path for local usage
        file_path = Path(file)

        # Get audio duration
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        # Time the transcription
        start_time = time.time()

        prompts = [
            {
                "prompt": "<|startoftranscript|>",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            }
        ]

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.time() - start_time

        for output in outputs:
            transcription = output.outputs[0].text
            return file, transcription, transcription_time, duration

        return file, "", transcription_time, duration


@app.local_entrypoint()
def benchmark_whisper():
    whisper_instance = Whisper()
    # Convert paths to strings for serialization
    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    results = list(whisper_instance.run.map(files))
    results_path = write_results(results, MODEL_NAME.replace("/", "-"))
    with dataset_volume.batch_upload() as batch:
        batch.put_file(results_path, f"/results/{results_path}")
