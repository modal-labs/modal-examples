# # Create an image with CUDA
#
# This example shows how you can use the Nvidia CUDA base image from DockerHub.
# We need to add Python 3 and pip with the `add_python` option because the image
# doesn't have these by default.

from modal import App, Image

image = Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
)
app = App(image=image)

# Now, we can create a function with GPU capabilities. Run this file with
# `modal run install_cuda.py`.


@app.function(gpu="T4")
def f():
    import subprocess

    subprocess.run(["nvidia-smi"])
