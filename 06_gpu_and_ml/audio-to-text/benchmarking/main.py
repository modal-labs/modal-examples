## Benchmarking Audio-to-Text Models - Parakeet, Whisper and WhisperX
import asyncio

from benchmark_parakeet import Parakeet
from benchmark_whisper import Whisper
from benchmark_whisperx import WhisperX
from common import (
    PARAKEET_MODEL_NAME,
    WHISPER_MODEL_NAME,
    WHISPERX_MODEL_NAME,
    app,
    dataset_volume,
)
from download_and_upload_lj_data import (
    download_and_upload_lj_data,
    upload_lj_data_subset,
)
from parse_token_counts import upload_token_counts
from postprocess_results import postprocess_results
from prepare_and_upload_data import process_wav_files
from utils import print_error, print_header, write_results

MODEL_CONFIGS = [
    ("Parakeet", PARAKEET_MODEL_NAME.replace("/", "-"), Parakeet()),
    ("Whisper", WHISPER_MODEL_NAME.replace("/", "-"), Whisper()),
    ("WhisperX", f"whisperx-{WHISPERX_MODEL_NAME}", WhisperX()),
]

# Default behavior downloads the local data subset
# To skip download (on subsequent runs), set REDOWNLOAD_DATA to False
# To use the full dataset, set USE_DATASET_SUBSET to False
REDOWNLOAD_DATA = True
USE_DATASET_SUBSET = True


def run_model_sync(model_name, instance, files):
    results = list(instance.run.map(files))
    results_path = write_results(results, model_name)
    with dataset_volume.batch_upload() as batch:
        batch.put_file(results_path, f"/results/{results_path}")
    print(f"✅ {model_name} results uploaded to /results/{results_path}")
    return model_name, results


@app.local_entrypoint()
async def main():
    from pathlib import Path

    # Download and upload data
    if REDOWNLOAD_DATA:
        if USE_DATASET_SUBSET:
            print_header("🔄 Downloading and uploading LJSpeech data subset...")
            upload_lj_data_subset.remote()
        else:
            print_header("🔄 Downloading and uploading LJSpeech data...")
            download_and_upload_lj_data.remote()
    else:
        print("Skipping data download")
        try:
            dataset_volume.listdir("/raw/wavs")
        except Exception as _:
            print_error(
                "Data not found in volume. Please re-run app.py with REDOWNLOAD_DATA=True. Note that this will take several minutes.",
            )

    # Process data
    print_header("🔄 Processing wav files into appropriate format...")
    process_wav_files.remote()

    # TODO: This should be in the process data step, will add in next PR
    print_header("✨ Parsing metadata to retrieve token counts...")
    upload_token_counts.remote()

    print_header("⚡️ Benchmarking all models in parallel...")
    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]
    print(f"Found {len(files)} files to benchmark")
    tasks = [
        asyncio.get_event_loop().run_in_executor(
            None, run_model_sync, model_name, instance, files
        )
        for _, model_name, instance in MODEL_CONFIGS
    ]
    await asyncio.gather(*tasks)

    print_header("🔮 Postprocessing results...")
    postprocess_results.remote()
