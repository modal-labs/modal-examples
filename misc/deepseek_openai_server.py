# DeepSeek LLM Server with llama.cpp
#
# This implementation provides a FastAPI server running DeepSeek-R1 language model
# using llama.cpp backend. It features:
#
# - GPU-accelerated inference using CUDA
# - API key authentication
# - Automatic model downloading and caching
# - GGUF model file merging
# - Swagger UI documentation
#
# Key Components:
#
# 1. Infrastructure Setup:
#    - Uses Modal for serverless deployment
#    - CUDA 12.4.0 with development toolkit
#    - Python 3.12 environment
#
# 2. Model Configuration:
#    - DeepSeek-R1 model with UD-IQ1_S quantization
#    - Persistent model storage using Modal Volumes
#    - Automatic GGUF file merging for split models
#
# 3. Server Features:
#    - FastAPI-based REST API
#    - API key authentication (X-API-Key header)
#    - Interactive documentation at /docs endpoint
#    - Configurable context length and batch size
#    - Flash attention support
#
# Hardware Requirements:
#    - 5x NVIDIA L40S GPUs
#    - Supports concurrent requests
#
# Usage:
# 1. Set your API key by modifying the TOKEN variable
# 2. Deploy using Modal
# 3. Access the API at http://localhost:8000
# 4. View API documentation at http://localhost:8000/docs
#
# Authentication:
# All API endpoints (except documentation) require the X-API-Key header
# Example:
# curl -H "X-API-Key: your-token" http://localhost:8000/v1/completions
#
# Model Settings:
# - Context length (n_ctx): 8096
# - Batch size (n_batch): 512
# - Thread count (n_threads): 12
# - GPU Layers: All (-1)
# - Flash Attention: Enabled
#
# Note: The server includes automatic redirection from root (/) to documentation (/docs)
# for easier API exploration.

from __future__ import annotations

import glob
import subprocess

# Standard library imports
from pathlib import Path
from typing import Optional

# Third-party imports
import modal

# ## Calling a Modal Function from the command line

# To start, we define our `main` function --
# the Python function that we'll run locally to
# trigger our inference to run on Modal's cloud infrastructure.

# This function, like the others that form our inference service
# running on Modal, is part of a Modal [App](https://modal.com/docs/guide/apps).
# Specifically, it is a `local_entrypoint`.
# Any Python code can call Modal Functions remotely,
# but local entrypoints get a command-line interface for free.

app = modal.App("deepseek-openai-server")

MINUTES = 60

HOURS = 60 * MINUTES

TOKEN = "super-secret-token"
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Combine all apt installations and system dependencies
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "libcurl4-openssl-dev",
        "libopenblas-dev",
        "libomp-dev",
        "clang",
    )
    # Set compiler environment variables
    .run_commands(
        "export CC=clang && export CXX=clang++",
        # Build llama.cpp with CUDA support
        "git clone https://github.com/ggerganov/llama.cpp && "
        "cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON && "
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split && "
        "cp llama.cpp/build/bin/llama-* llama.cpp",
    )
    # Install all Python dependencies at once
    .pip_install(
        [
            "fastapi==0.115.8",
            "sse_starlette==2.2.1",
            "pydantic==2.10.6",
            "uvicorn[standard]==0.34.0",
            "python-multipart==0.0.20",
            "starlette-context==0.3.6",
            "pydantic-settings==2.7.1",
            "ninja==1.11.1.3",
            "packaging==24.2",
            "wheel",
            "torch==2.6.0",
        ],
    )
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python',
        gpu=modal.gpu.L40S(count=1),
    )
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

# To make the model weights available on Modal,
# we download them from Hugging Face.

# Modal is serverless, so disks are by default ephemeral.
# To make sure our weights don't disappear between runs
# and require a long download step, we store them in a
# Modal [Volume](https://modal.com/docs/guide/volumes).
model_cache = modal.Volume.from_name("deepseek", create_if_missing=True)
cache_dir = "/root/.cache/deepseek"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=download_image, volumes={cache_dir: model_cache}, timeout=30 * MINUTES
)
def download_model(repo_id, allow_patterns, revision: Optional[str] = None):
    from huggingface_hub import snapshot_download

    print(f"🦙 downloading model from {repo_id} if not present")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
    )

    model_cache.commit()  # ensure other Modal Functions can see our writes before we quit

    print("🦙 model loaded")


# For more on how to use Modal Volumes to store model weights,
# see [this guide](https://modal.com/docs/guide/model-weights).
N_GPU = 5
MODELS_DIR = "/deepseek"


@app.function(
    image=vllm_image,
    gpu=modal.gpu.L40S(count=N_GPU),
    scaledown_window=5 * MINUTES,
    timeout=15 * MINUTES,
    volumes={MODELS_DIR: model_cache},
    max_containers=1,
)
@modal.asgi_app()
def serve():
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    org_name = "unsloth"
    model_name = "DeepSeek-R1"
    quant = "UD-IQ1_S"
    repo_id = f"{org_name}/{model_name}-GGUF"
    model_pattern = f"*{quant}*"
    download_model.remote(repo_id, [model_pattern])
    model_cache.reload()  # ensure we have the latest version of the weights

    model_entrypoint_file = (
        f"{model_name}-{quant}/DeepSeek-R1-{quant}-00001-of-00003.gguf"
    )
    model_path = MODELS_DIR + "/" + model_entrypoint_file
    # Find and merge GGUF files
    model_dir = f"{MODELS_DIR}/{model_name}-{quant}"
    gguf_files = sorted(glob.glob(f"{model_dir}/*.gguf"))
    if len(gguf_files) > 1:
        print(f"Found {len(gguf_files)} GGUF files to merge")
        output_file = f"{model_dir}/{model_name}-{quant}-merged.gguf"
        if not Path(output_file).exists():
            print(f"🔄 Merging GGUF files to {output_file}")
            merge_command = (
                ["/llama.cpp/llama-gguf-split", "--merge"]
                + [gguf_files[0]]
                + [output_file]
            )
            print(f"Merging files with command: {' '.join(merge_command)}")
            subprocess.run(merge_command, check=True)
            print("🔄 GGUF files merged successfully")
        model_path = output_file
    else:
        model_path = (
            gguf_files[0]
            if gguf_files
            else f"{model_dir}/DeepSeek-R1-{quant}-00001-of-00003.gguf"
        )
    model_cache.reload()  # ensure we have the latest version of the weights
    print(f"🔄 Using model path: {model_path}")
    # Create model settings directly
    model_settings = [
        ModelSettings(
            model=model_path,  # Replace with your model path
            n_gpu_layers=-1,  # Use all GPU layers
            n_ctx=8096,
            n_batch=512,
            n_threads=12,
            verbose=True,
            flash_attn=True,
        )
    ]

    # Create server settings
    server_settings = ServerSettings(host="0.0.0.0", port=8000, api_key=TOKEN)

    # Create the llama.cpp app
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )

    return app
