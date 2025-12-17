# # Benchmark NVSHMEM with and without P2P

# This example benchmarks NVSHMEM reduction operations with P2P enabled and disabled
# to measure the performance impact of direct GPU-to-GPU communication.

import subprocess
from pathlib import Path

import modal

here = Path(__file__).parent

# Build a custom container image with NVSHMEM and its dependencies
nvshmem_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install(
        "libopenmpi-dev",
        "openmpi-bin",
    )
    .pip_install(
        "mpi4py",
        "nvshmem4py-cu12",
        "nvidia-nvshmem-cu12",
        "cupy-cuda12x",
        "cuda-python",
    )
    .add_local_file(
        here / "nvshmem_reduction_benchmark.py",
        "/root/nvshmem_reduction_benchmark.py",
    )
)

app = modal.App("benchmark-nvshmem-p2p")


def run_benchmark(disable_p2p: bool, n_gpus: int):
    """Helper function to run a single benchmark configuration."""
    script_path = "/root/nvshmem_reduction_benchmark.py"

    env = {"DISABLE_P2P": "1" if disable_p2p else "0"}

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

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
    )

    if result.returncode != 0:
        print(f"BENCHMARK FAILED (exit code {result.returncode})")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        raise RuntimeError(f"Benchmark failed with exit code {result.returncode}")

    return result.stdout


@app.function(gpu="H100:4", image=nvshmem_image, timeout=600)
def benchmark_p2p_comparison():
    """
    Run NVSHMEM benchmarks with P2P enabled and disabled to compare performance.
    """
    import cupy as cp

    n_gpus = cp.cuda.runtime.getDeviceCount()

    print("=" * 80)
    print("NVSHMEM P2P PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    # Run with P2P enabled
    print("=" * 80)
    print("BENCHMARK 1: P2P ENABLED")
    print("=" * 80)
    print()

    p2p_enabled_output = run_benchmark(disable_p2p=False, n_gpus=n_gpus)
    print(p2p_enabled_output)

    print()
    print()

    # Run with P2P disabled
    print("=" * 80)
    print("BENCHMARK 2: P2P DISABLED")
    print("=" * 80)
    print()

    p2p_disabled_output = run_benchmark(disable_p2p=True, n_gpus=n_gpus)
    print(p2p_disabled_output)

    print()
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Extract timing information from outputs
    def extract_mean_time(output: str) -> float:
        """Extract mean time from benchmark output."""
        for line in output.split("\n"):
            if "Mean:" in line:
                # Extract number like "Mean:   123.456 ms"
                parts = line.split("Mean:")[1].strip().split()
                return float(parts[0])
        return 0.0

    def extract_bandwidth(output: str) -> float:
        """Extract bandwidth from benchmark output."""
        for line in output.split("\n"):
            if "Effective Bandwidth:" in line:
                # Extract number like "Effective Bandwidth: 12.34 GB/s"
                parts = line.split(":")[1].strip().split()
                return float(parts[0])
        return 0.0

    p2p_enabled_time = extract_mean_time(p2p_enabled_output)
    p2p_disabled_time = extract_mean_time(p2p_disabled_output)
    p2p_enabled_bw = extract_bandwidth(p2p_enabled_output)
    p2p_disabled_bw = extract_bandwidth(p2p_disabled_output)

    if p2p_enabled_time > 0 and p2p_disabled_time > 0:
        speedup = p2p_disabled_time / p2p_enabled_time
        slowdown_pct = ((p2p_disabled_time - p2p_enabled_time) / p2p_enabled_time) * 100

        print(f"P2P Enabled:  {p2p_enabled_time:.3f} ms  ({p2p_enabled_bw:.2f} GB/s)")
        print(f"P2P Disabled: {p2p_disabled_time:.3f} ms  ({p2p_disabled_bw:.2f} GB/s)")
        print()
        print(f"Speedup with P2P: {speedup:.2f}x")
        print(f"Performance loss without P2P: {slowdown_pct:.1f}%")
        print()

        if speedup > 1.5:
            print("✓ P2P provides significant performance benefit")
        elif speedup > 1.1:
            print("✓ P2P provides moderate performance benefit")
        else:
            print("⚠ P2P shows minimal performance difference")
            print(
                "  (This may indicate P2P is not available or data size is too small)"
            )
    else:
        print("⚠ Could not extract timing information from benchmark outputs")

    print("=" * 80)


@app.local_entrypoint()
def main():
    benchmark_p2p_comparison.remote()
