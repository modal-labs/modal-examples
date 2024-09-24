# # Run llama.cpp on Modal

# This example shows how you can run [llama.cpp](https://github.com/ggerganov/llama.cpp) on Modal.


from modal import App, Image

app = App("llama-cpp-modal")


# Change this to the model you want to use
LLAMA_CPP_RELEASE = "b3367"
MODEL_NAME = "Meta-Llama-3-8B-Instruct"
MODEL = "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"
NUM_OUTPUT_TOKENS = 128

llama_cpp_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["curl", "unzip"])
    .run_commands(
        [
            f"curl -L -O https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.zip",
            f"unzip llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.zip",
        ]
    )
)

llama_3_8b_image = llama_cpp_image.run_commands(
    [
        f"curl -L -O https://huggingface.co/bartowski/{MODEL_NAME}-GGUF/resolve/main/{MODEL}",
    ]
)


@app.function(image=llama_3_8b_image)
def llama_cpp_inference():
    import subprocess

    subprocess.run(
        [
            "/build/bin/llama-cli",
            "-m",
            f"/{MODEL}",
            "-n",
            f"{NUM_OUTPUT_TOKENS}",
            "-p",
            "Write a poem about New York City",
        ]
    )


@app.local_entrypoint()
def main():
    llama_cpp_inference.remote()
