# # Create an image with CUDA
#
# This example shows how you can create use the Nvidia CUDA base image from DockerHub.
#
# We need to install Python 3 and pip in `setup_dockerfile_commands` because the
# base image doesn't have them by default. The commands to do this are specific
# to Ubuntu 22.04 in this example.

from modal import Image, Stub


stub = Stub()

stub.image = Image.from_dockerhub(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04",
    setup_dockerfile_commands=[
        "RUN apt-get update",
        "RUN apt-get install -y python3 python3-pip python-is-python3",
    ],
)

# Now, we can create a function with GPU capabilities. Run this file with
# `modal run install_cuda.py`.

@stub.function(gpu="T4")
def f():
    import subprocess

    subprocess.run(["nvidia-smi"])
