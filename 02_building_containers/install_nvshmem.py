# # Install NVSHMEM for multi-GPU communication

# This example demonstrates how to install and use NVSHMEM (NVIDIA Shared Memory)
# for direct GPU-to-GPU communication on a single node with multiple GPUs.

# NVSHMEM provides a Partitioned Global Address Space (PGAS) programming model
# where each GPU can directly read from and write to memory on other GPUs.
# This is useful for tightly-coupled parallel applications that need low-latency
# GPU-to-GPU communication without going through the CPU.

# This example shows how to use NVSHMEM with MPI for multi-GPU coordination
# and demonstrates a simple reduction operation across GPUs.

import subprocess
from pathlib import Path

import modal

here = Path(__file__).parent

# Build a custom container image with NVSHMEM and its dependencies.
# We start from NVIDIA's CUDA development image which includes the CUDA toolkit.

nvshmem_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])  # Remove chatty NVIDIA container entry messages
    .apt_install(
        "libopenmpi-dev",  # OpenMPI development libraries
        "openmpi-bin",  # OpenMPI binaries
    )
    .pip_install(
        "mpi4py",  # Python MPI bindings
        "nvshmem4py-cu12",  # NVSHMEM Python bindings
        "nvidia-nvshmem-cu12",  # NVSHMEM C/C++ libraries
        "cupy-cuda12x",  # GPU array library
        "cuda-python",  # NVIDIA CUDA Python bindings
    )
    .add_local_file(
        here / "nvshmem_reduction.py",
        "/root/nvshmem_reduction.py",
    )
)

app = modal.App("example-install-nvshmem")


@app.function(gpu="H100:4", image=nvshmem_image, timeout=600)
def run_nvshmem_example():
    """
    Run NVSHMEM example using MPI to coordinate multiple GPUs.

    This uses mpirun to launch one MPI process per GPU, where each process
    initializes NVSHMEM and performs collective operations.
    """
    # Determine number of GPUs automatically
    import cupy as cp

    n_gpus = cp.cuda.runtime.getDeviceCount()

    script_path = "/root/nvshmem_reduction.py"

    print(f"Running NVSHMEM example with {n_gpus} GPUs using MPI\n")
    print("=" * 60)

    # Run the script with mpirun
    # -np: number of processes
    # --allow-run-as-root: needed in container environments
    # --bind-to none: don't bind processes to specific cores
    cmd = [
        "mpirun",
        "-np",
        str(n_gpus),
        "--allow-run-as-root",
        "--bind-to",
        "none",
        "python",
        script_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print("=" * 60)
        print("\n✓ NVSHMEM multi-GPU example completed successfully!")
        print(f"✓ {n_gpus} GPUs coordinated via NVSHMEM")
        print("✓ Collective reduction operation verified")

    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"MPI process failed with code {e.returncode}")


# Also provide a simpler check function


@app.function(gpu="H100:4", image=nvshmem_image)
def check_gpu_configuration():
    """Check GPU configuration and NVSHMEM package installation."""
    import cupy as cp
    import nvshmem.core

    print("NVSHMEM Package Information:")
    print(f"  - nvshmem.core module: {nvshmem.core.__file__}")

    n_gpus = cp.cuda.runtime.getDeviceCount()
    print("\nGPU Configuration:")
    print(f"  - Number of GPUs detected: {n_gpus}")

    for i in range(n_gpus):
        props = cp.cuda.runtime.getDeviceProperties(i)
        print(f"  - GPU {i}: {props['name'].decode()}")

    # Check physical topology with nvidia-smi
    print("\n" + "=" * 60)
    print("Physical GPU Topology (nvidia-smi topo -m)")
    print("=" * 60)
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
    )
    print(result.stdout)

    # Check P2P capabilities and try to enable
    print("\n" + "=" * 60)
    print("Testing P2P Access Capabilities")
    print("=" * 60)
    for i in range(n_gpus):
        cp.cuda.Device(i).use()  # Set as current device

        for j in range(n_gpus):
            if i != j:
                can_access = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                print(f"\nGPU {i} -> GPU {j}:")
                print(f"  deviceCanAccessPeer: {can_access}")

                if can_access:
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                        print("  ✓ Successfully enabled P2P access")
                    except Exception as e:
                        print(f"  ✗ Failed to enable: {e}")
                else:
                    print("  ✗ P2P not supported")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    if any(
        [
            cp.cuda.runtime.deviceCanAccessPeer(i, j)
            for i in range(n_gpus)
            for j in range(n_gpus)
            if i != j
        ]
    ):
        print("✓ GPU P2P access is available via NVLink/PCIe")
        print("  NVSHMEM will use direct GPU-to-GPU communication")
    else:
        print("✗ GPU P2P access is not available in this environment")
        print("  (This is common in containerized/virtualized environments)")
        print("  NVSHMEM will use system memory for GPU communication")
        print("  (Performance impact, but functionally equivalent)")
    print("=" * 60)


# You can run this example with:
# modal run 02_building_containers/install_nvshmem.py


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Running NVSHMEM Multi-GPU Example")
    print("=" * 60)
    run_nvshmem_example.remote()
