# # Generate: Fooocus
#
# This example demonstrates how to set up and run a web server using the Modal library with Fooocus as the frontend.
# Fooocus provides a beginner-friendly interface to work with the SDXL 1.0 model for image generation tasks.
# The script includes the setup of a Docker image, initialization of Fooocus, and launching a web server with GPU support.
#
# ## Basic setup

import modal

# To create an image that can run Fooocus, we start from an official NVIDIA base image and then add Python
# and a few system packages.
#
# We then download the Fooocus repository.

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.1-base-ubuntu22.04", add_python="3.10")
    .apt_install(
        "software-properties-common",
        "git",
        "git-lfs",
        "coreutils",
        "aria2",
        "libgl1",
        "libglib2.0-0",
        "curl",
        "wget",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
    )
    .run_commands("git clone https://github.com/lllyasviel/Fooocus.git")
)

# ## Initialize Fooocus
#
# We are not limited to running shell commands and package installers in the image setup.
# We can also run Python functions by defining them in our code and passing them to the `run_function` method.
#
# This function installs Fooocus's dependencies and downloads the SDXL 1.0 model to the container image.
#
# This all happens at the time the container image is defined, so that the image is ready to run Fooocus when it is deployed.


def init_Fooocus():
    import os
    import subprocess

    # change the working directory to the Fooocus directory and install the required Python packages from the requirements file.
    os.chdir("/Fooocus")
    os.system("pip install -r requirements_versions.txt")

    # change the directory to the models' checkpoints and download the SDXL 1.0 model using wget.
    os.chdir("./models/checkpoints")
    subprocess.run(
        "wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'",
        shell=True,
    )


GPU_CONFIG = "T4"
image = image.run_function(init_Fooocus, gpu=GPU_CONFIG)

# ## Run Fooocus
#
# The `run` function is decorated with `app.function` to define it as a Modal function.
# The `web_server` decorator indicates that this function will serve a web application on the specified port.
# We increase the startup timeout to three minutes to account for the time it takes to load the model and start the server.

app = modal.App("Fooocus", image=image)

PORT = 8000
MINUTES = 60


@app.function(gpu=GPU_CONFIG, timeout=10 * MINUTES)
@modal.web_server(port=PORT, startup_timeout=3 * MINUTES)
def run():
    import os
    import subprocess

    # change the working directory to the Fooocus directory.
    os.chdir("/Fooocus")

    # launch the Fooocus application using a subprocess that listens on the specified port
    subprocess.Popen(
        [
            "python",
            "launch.py",
            "--listen",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--always-high-vram",
        ]
    )
