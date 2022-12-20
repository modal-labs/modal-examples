# # PyTorch with CUDA GPU support
#
# This example shows how you can use CUDA GPUs in Modal, with a minimal PyTorch
# image. You can specify GPU requirements in the `stub.function` decorator.

import time

import modal

stub = modal.Stub(
    "example-import-torch",
    image=modal.Image.debian_slim().pip_install(["torch"], "https://download.pytorch.org/whl/cu116"),
)


@stub.function(gpu="any")
def gpu_function():
    import subprocess

    import torch

    subprocess.run(["nvidia-smi"])
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())


if __name__ == "__main__":
    t0 = time.time()
    with stub.run():
        gpu_function.call()
    print("Full time spent:", time.time() - t0)
