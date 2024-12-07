# # Run `llama.cpp` on Modal

# [`llama.cpp`](https://github.com/ggerganov/llama.cpp) is a C++ library for running LLM inference.
# It's lightweight, fast, and includes support for exotic quantizations like 5-bit integers.
# This example shows how you can run `llama.cpp` on Modal.

# We start by defining a [container image](https://modal.com/docs/guide/custom-container) with `llama.cpp` installed.

import modal

LLAMA_CPP_RELEASE = "b3472"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl", "unzip"])
    .run_commands(
        [
            f"curl -L -O https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.zip",
            f"unzip llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.zip",
        ]
    )
)

# Next, we download a pre-trained model to run.
# We use a model with 5-bit quantization.
# The model format, `.gguf`, is a custom format used by `llama.cpp`.

ORG_NAME = "bartowski"
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-GGUF"
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
REVISION = "9a8dec50f04fa8fad1dc1e7bc20a84a512e2bb01"


def download_model(repo_id, filename, revision):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir="/",
    )


# We can execute this Python function as part of building our image,
# just as we can install dependencies and set environment variables,
# with the `run_function` method:

image = (
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model, args=(REPO_ID, MODEL_FILE, REVISION))
)


# Now, we're ready to define a serverless function that runs `llama.cpp`!

# We wrap that function with a decorator from a Modal App,
# `@app.function`, specifying the image it should run on
# and setting the maximum number of concurrent replicas
# (here, `100`, which is the default for CPU Functions).


app = modal.App("llama-cpp-modal", image=image)


@app.function(image=image, concurrency_limit=100)
def llama_cpp_inference(
    prompt: str = None,
    num_output_tokens: int = 128,
):
    import subprocess

    if prompt is None:
        prompt = "Write a poem about New York City.\n"
    if num_output_tokens is None:
        num_output_tokens = 128

    subprocess.run(
        [
            "/build/bin/llama-cli",
            "-m",
            f"/{MODEL_FILE}",
            "-n",
            str(num_output_tokens),
            "-p",
            prompt,
        ],
        check=True,
    )


@app.local_entrypoint()
def main(prompt: str = None, num_output_tokens: int = None):
    llama_cpp_inference.remote(prompt, num_output_tokens)
