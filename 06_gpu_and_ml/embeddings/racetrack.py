import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterator

import modal

# # Volumes
data_volume = modal.Volume.from_name("example-embedding-data")
results_volume = modal.Volume.from_name("racetrack", create_if_missing=True)
DATA_DIR = Path("/data")
RESULTS_DIR = DATA_DIR / "racetrack-results"
HF_HOME = DATA_DIR / "hf"
TH_CACHE_DIR = DATA_DIR / "model-compile-cache"

# # Images

infinity_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pillow",  # for Infinity input typehint
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "tqdm",  # progress bar for dataset download
            "infinity_emb[all]==0.0.76",  # for Infinity inference lib
            "sentencepiece",  # for this particular chosen model
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            # For fast HuggingFace model and data caching and download in our Volume
            "HF_HOME": HF_HOME.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

th_compile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "tqdm",  # progress bar for dataset download
            "torch",  # torch.compile
            "transformers",  # CLIPVisionModel etc.
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            # For fast HuggingFace model and data caching and download in our Volume
            "HF_HOME": HF_HOME.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Enables speedy caching across containers
            "TORCHINDUCTOR_CACHE_DIR": TH_CACHE_DIR.as_posix(),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        }
    )
)

# # Timer class


class InferenceTimer:
    @modal.enter()
    def check_for_logfile(self):
        self.start_time = time.perf_counter()
        self.n_inferences = 0
        self.end_time = None
        if not hasattr(self, 'logfile'):
            raise ValueError("No `self.logfile` found. " \
            "Can't currently inherit constructors, so " \
            "all child classes of InferenceTimer must " \
            "have their own `logfile` modal.parameter!")
    
    @modal.exit()
    def log_stats(self):
        self.end_time= time.perf_counter()

        with csv. self.logfile
         with open(local_logfile, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    # write header
                    writer.writerow(
                        [
                            "start_time",
                            "concurrency",
                            "max_containers",
                            "gpu",
                            "n_images",
                            "total_time",
                            "total_throughput",
                            "avg_model_throughput",
                        ]
                    )
                # write your row
                writer.writerow(
                    [
                        batch_size,
                        allow_concurrent_inputs,
                        max_containers,
                        gpu,
                        n_ims,
                        total_duration,
                        total_throughput,
                        avg_throughput,
                    ]
                )
