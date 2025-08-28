# ---
# runtimes: ["runc"]
# ---

# # Trace and profile GPU-accelerated applications with NSight Systems

# This example demonstrates how to use
# NVIDIA's [NSight Systems](https://developer.nvidia.com/nsight-systems)
# profiling tool on Modal.

# NSight Systems traces and profiles GPU-accelerated applications at the _systems_ level --
# that is, it correlates events across the host and the device(s), aka the CPU(s) and GPU(s).

# To run NSight Systems, you will need to use a different version of Modal's Function runtime
# that allows user code to perform additional syscalls.
# This is made available to select users on Modal's [Enterprise Plan](https://modal.com/pricing).
# Users on that plan can request access by contacting Modal Support.

# Note that the PyTorch profiler captures similar metrics to NSight Systems but
# does not require elevated permissions. You can find sample code for that [here](https://modal.com/docs/examples/torch_profiling).

# ## Install NSight Systems and the CUDA Toolkit

# First, we need to install the software into our
# [container Image](https://modal.com/docs/guide/images).

from pathlib import Path

import modal

app = modal.App("example-nsys")
here = Path(__file__).parent  # directory of this script

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        # install system packages required to install NSight Systems
        "apt update",
        "apt install -y --no-install-recommends gnupg wget software-properties-common",
        # add NVIDIA's GPG keys
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        # add the contrib APT repository that distributes CUDA software on Debian Linux
        "add-apt-repository contrib",
        # install NSight Systems and the CUDA Toolkit
        "apt update",
        "apt install -y cuda-nsight-systems-12-8 cuda-toolkit-12-8",
    )
    # update the PATH so that the CUDA software can be found
    .env({"PATH": "/usr/local/cuda-12/bin:${PATH}"})
    # add local files for use in `modal shell`, see discussion below
    .add_local_file(here / "toy.cu", remote_path="/root/toy.cu")
)

# ## Run NSight Systems on Modal

# Now we can use the `nsys` command-line tool on Modal.

# As a simple demonstration, we can pass in code directly as a string,
# compile it with `nvcc`, and then profile it with `nsys`.

# We both return the generated profile to the caller
# and persist the profile to a [Modal Volume](https://modal.com/docs/guide/volumes).

profile_volume = modal.Volume.from_name("example-nsys-traces", create_if_missing=True)


@app.function(image=image, gpu="A10", volumes={"/traces": profile_volume})
def compile_and_profile(code: str, output_path: str = "profile.nsys-rep"):
    import subprocess

    Path("kernel.cu").write_text(code)

    subprocess.run(["nvcc", "-arch=sm_75", "kernel.cu"], check=True)

    output_path = f"/traces/{output_path}"

    subprocess.run(["nsys", "profile", "--output", output_path, "./a.out"], check=True)

    return Path(output_path).read_bytes()


# To confirm everything's working, we can pass the code directly to our Modal Function
# from the terminal and write the response into a local file:

# ```bash
# MODAL_FUNCTION_RUNTIME=runc modal run -w profile.nsys-rep nsys.py::compile_and_profile --code $'#include <iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }'
# ```

# If you [install the NSight Systems GUI on your local machine](https://developer.nvidia.com/nsight-systems/get-started),
# you can view the trace and profiler report -- no NVIDIA GPUs or CUDA Toolkit required.

# The `local_entrypoint` below shows a slightly more realistic pattern,
# based on passing in a path to a CUDA program.

# ```bash
# MODAL_FUNCTION_RUNTIME=runc modal run nsys.py --input-path toy.cu
# ```


@app.local_entrypoint()
def main(input_path: str | None = None, output_path: str | None = None):
    if not input_path:
        input_path = here / "toy.cu"
    code = Path(input_path).read_text()

    profile = compile_and_profile.remote(code)

    (path := Path(output_path or here / "profile.nsys-rep")).write_bytes(profile)

    print(f"profile saved at {path}")


# For multi-file compilation and profiling, we recommend using
# [`Image.add_local_file`](https://modal.com/docs/reference/modal.Image#add_local_file) and
# [`Image.add_local_dir`](https://modal.com/docs/reference/modal.Image#add_local_dir)
# to add your source code to the Image and then either editing the `nvcc` command directly
# or profiling in an interactive shell with

# ```bash
# MODAL_FUNCTION_RUNTIME=runc modal shell nsys.py
# ```

# Profiles saved to `/traces/` will be persisted in a Modal Volume.
# See [the guide](https://modal.com/docs/guide/volumes)
# for details on how to retrieve them for local review.
