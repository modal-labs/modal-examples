# # Installing the CUDA Toolkit on Modal

# This code sample is intended to quickly show how different layers of the CUDA stack are used on Modal.
# For greater detail, see our [guide to using CUDA on Modal](https://modal.com/docs/guide/cuda).

# All Modal Functions with GPUs already have the NVIDIA CUDA drivers,
# NVIDIA System Management Interface, and CUDA Driver API installed.

import modal

app = modal.App("example-install-cuda")


@app.function(gpu="T4")
def nvidia_smi():
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)


# This is enough to install and use many CUDA-dependent libraries, like PyTorch.


@app.function(gpu="T4", image=modal.Image.debian_slim().pip_install("torch"))
def torch_cuda():
    import torch

    print(torch.cuda.get_device_properties("cuda:0"))


# If your application or its dependencies need components of the CUDA toolkit,
# like the `nvcc` compiler driver, installed as system libraries or command-line tools,
# you'll need to install those manually.

# We recommend the official NVIDIA CUDA Docker images from Docker Hub.
# You'll need to add Python 3 and pip with the `add_python` option because the image
# doesn't have these by default.


ctk_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])  # removes chatty prints on entry


@app.function(gpu="T4", image=ctk_image)
def nvcc_version():
    import subprocess

    return subprocess.run(["nvcc", "--version"], check=True)


# You can check that all these functions run by invoking this script with `modal run`.


@app.local_entrypoint()
def main():
    nvidia_smi.remote()
    torch_cuda.remote()
    nvcc_version.remote()
