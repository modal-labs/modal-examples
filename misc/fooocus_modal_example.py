# # Generate: Fooocus
#
# This example demonstrates how to set up and run a web server using the Modal library with Fooocus as the frontend.
# Fooocus provides a beginner-friendly interface to work with the SDXL 1.0 model for image generation tasks.
# The script includes the setup of a Docker image, initialization of Fooocus, and launching a web server with GPU support.
#
# The SDXL model is predownloaded to minimize cold-start times. The script also handles the download of smaller models (like VAE) before starting.
#
# ## Basic setup
#
# The following imports are necessary for the script to interact with the Modal library and define the required configurations.
# `Image` is used to define the Docker image for the container.
# `Stub` is a Modal concept that represents a callable function or a group of functions.
# `gpu` provides GPU configurations, and `web_server` is a decorator to indicate that the function will serve a web application.

from modal import Image, Stub, gpu, web_server

# Define constants for the Docker image, Python version, GPU configuration, and the port number for the web server.
DOCKER_IMAGE = "nvidia/cuda:12.3.1-base-ubuntu22.04"
PYTHON_VER = "3.10"
GPU_CONFIG = gpu.T4()
PORT = 8000

# ## Initialize Fooocus
#
# The `init_Fooocus` function is responsible for setting up the Fooocus environment.
# It installs the required Python packages and downloads the SDXL 1.0 model to the container image.
# This is done to avoid large downloads during cold-starts, which improves the startup time of the application.

def init_Fooocus():
    import os
    import subprocess

    # Change the working directory to the Fooocus directory and install the required Python packages from the requirements file.
    os.chdir(f"/Fooocus")
    os.system(f"pip install -r requirements_versions.txt")

    # Change the directory to the models' checkpoints and download the SDXL 1.0 model using wget.
    os.chdir(f"./models/checkpoints")
    subprocess.run(f"wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'", shell=True)

# ## Define container image
#
# This section defines the container image by installing essential system packages and setting up the Fooocus repository.
# The image is built starting from the specified DOCKER_IMAGE and adding Python, system packages, and the Fooocus application.

image = (
    Image.from_registry(DOCKER_IMAGE, add_python=PYTHON_VER)
    .run_commands("apt update -y")
    .apt_install(
        "software-properties-common",
        "git",
        "git-lfs",
        "coreutils",
        "aria2",
        "libgl1",
        "libglib2.0-0",
        'curl',
        "wget",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
    )
    .run_commands("git clone https://github.com/lllyasviel/Fooocus.git")
    .pip_install("pygit2==1.12.2")
    .run_function(init_Fooocus, gpu=GPU_CONFIG)
)

# ## Run Fooocus
#
# The `run` function is decorated with `stub.function` to set the GPU configuration and function timeout.
# The `web_server` decorator indicates that this function will serve a web application on the specified port with a startup timeout.

stub = Stub("Fooocus", image=image)

@stub.function(gpu=GPU_CONFIG, timeout=60 * 10) # Set GPU configuration and function timeout
@web_server(port=PORT, startup_timeout=180) # Define the web server settings
def run():
    import os
    import subprocess

    # Change the working directory to the Fooocus directory.
    os.chdir(f"/Fooocus")

    # Launch the Fooocus application using a subprocess that listens on the specified port and enables high VRAM usage.
    subprocess.Popen(
        [
            "python",
            "launch.py",
            "--listen",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--always-high-vram"
        ]
    )
