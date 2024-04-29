import modal
import subprocess  # Import subprocess to fix the NameError

gpu_device = "a100"

# Define the container image using Conda for package management
image = (
    modal.Image.conda()
    .apt_install("git")
    .run_commands(
        "apt-get update",
        "apt-get install -y wget build-essential git",
        "pip install --upgrade pip",
    )
    .conda_install(
        "nvidia::cuda-toolkit=12.4.1",
        channels=["nvidia"],
        gpu=gpu_device,
    )
    .conda_install(
        "nvidia::cuda-nvcc=12.4.131",
        channels=["nvidia"],
        gpu=gpu_device,
    )
    .run_commands(
        "which nvidia-smi",
        "which nvcc",
        "ls /usr/local/",
        # "ls /usr/local/cuda-12.4.1",
    )
    # .env(
    #     {
    #         "CUDA_HOME": "/usr/local/cuda-12.4.1",
    #     },
    # )
    .pip_install(
        "torch",
        "transformers",
        "huggingface-hub",
        gpu=gpu_device,
    )
    .run_commands(
        "pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
        "git clone https://github.com/nerfstudio-project/nerfstudio.git",
        "cd nerfstudio && pip install -e .",
        gpu=gpu_device,
    )
    .pip_install(
        "httpx",
        "requests",
        "tqdm",
    )
    .pip_install("kplanes-nerfstudio")
)

app = modal.App(name="nerfstudio-on-modal", image=image)


@app.function(
    gpu=gpu_device,
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    keep_warm=1,
    timeout=1800,
)
@modal.web_server(7007, startup_timeout=600)
def nerfstudio_web():
    import subprocess

    cmd = "cd nerfstudio && python -m nerfstudio.server --port 7007"
    subprocess.Popen(cmd, shell=True)
