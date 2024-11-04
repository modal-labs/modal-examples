# ---
# tags: ["use-case-lm-inference"]
# ---
# # Run `llama.cpp` on Modal

# [`llama.cpp`](https://github.com/ggerganov/llama.cpp) is a C++ library for running LLM inference.
# It's lightweight, fast, and includes support for exotic quantizations like 5-bit integers.
# This example shows how you can run `llama.cpp` on Modal.

# We start by defining a container image with `llama.cpp` installed.

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

MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
REVISION = "9a8dec50f04fa8fad1dc1e7bc20a84a512e2bb01"

image = image.run_commands(
    f"curl --fail-with-body -L -O https://huggingface.co/bartowski/{MODEL_NAME}-GGUF/resolve/{REVISION}/{MODEL_FILE}?download=true"
)

# Now, we're ready to define a serverless function that runs `llama.cpp`.
# We wrap that function with a decorator from a Modal App,
# `@app.function` specifying the image it should run on
# and setting the maximum number of concurrent replicas
# (here, `100`, which is the default).

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
        ]
    )


@app.local_entrypoint()
def main(prompt: str = None, num_output_tokens: int = None):
    llama_cpp_inference.remote(prompt, num_output_tokens)
# Comment to force rebuild
