from pathlib import Path

import modal

app = modal.App("examples-ncu-profiling")
here = Path(__file__).parent

image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt update",
        "apt install -y --no-install-recommends gnupg wget software-properties-common",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "add-apt-repository contrib",
        "apt-get update",
        "apt-get -y install cuda-nsight-systems-12-4 cuda-toolkit-12-4",
    )
    .env({"PATH": "/usr/local/cuda-12/bin:${PATH}"})
)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(image=image, gpu="a10g")
def compile_and_profile(code: str):
    import subprocess

    Path("kernel.cu").write_text(code)

    subprocess.run(["nvcc", "-arch=sm_70", "kernel.cu"], check=True)

    subprocess.run(
        ["ncu", "-o", "profile.ncu", "./a.out"],
        check=True,
    )

    return Path("profile.ncu.ncu-rep").read_bytes()


@app.local_entrypoint()
def main(path: str = None):
    if not path:
        path = here / "kernel.cu"
    code = Path(path).read_text()

    profile = compile_and_profile.remote(code)

    (path := Path("/tmp/profile.ncu.ncu-rep")).write_bytes(profile)

    print(f"profile saved at {path}")
