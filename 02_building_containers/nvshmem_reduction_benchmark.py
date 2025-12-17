import os
import re
import subprocess
import time

import cupy as cp
import mpi4py.MPI as MPI
import nvshmem.core as nvshmem
from cuda.core.experimental import Device, system


def get_nvlink_counters(gpu_id):
    """Read NVLink TX/RX byte counters for a specific GPU."""
    try:
        # Query NVLink counters using nvidia-smi
        # Use -i to specify GPU ID and -gt d to get data transfer counters
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-i", str(gpu_id), "-gt", "d"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        # Parse output to extract TX/RX bytes
        # Output format:
        # GPU 0: NVIDIA H100 80GB HBM3 (UUID: ...)
        #      Link 0: Data Tx: 123456 KiB
        #      Link 0: Data Rx: 654321 KiB
        tx_kib = 0
        rx_kib = 0

        for line in result.stdout.split("\n"):
            if "Data Tx:" in line:
                # Extract the number in KiB
                match = re.search(r"Data Tx:\s+(\d+)\s+KiB", line)
                if match:
                    tx_kib += int(match.group(1))
            elif "Data Rx:" in line:
                # Extract the number in KiB
                match = re.search(r"Data Rx:\s+(\d+)\s+KiB", line)
                if match:
                    rx_kib += int(match.group(1))

        # Convert KiB to bytes
        return {"tx_bytes": tx_kib * 1024, "rx_bytes": rx_kib * 1024}
    except Exception:
        return None


def print_nvlink_stats(label, counters_before, counters_after):
    """Print NVLink statistics comparing before/after counters."""
    print(f"\n{label}", flush=True)
    print("=" * 60, flush=True)

    if counters_before is None or counters_after is None:
        print("NVLink counters not available", flush=True)
        return

    tx_before_gb = counters_before["tx_bytes"] / (1024**3)
    rx_before_gb = counters_before["rx_bytes"] / (1024**3)
    tx_after_gb = counters_after["tx_bytes"] / (1024**3)
    rx_after_gb = counters_after["rx_bytes"] / (1024**3)

    tx_diff = counters_after["tx_bytes"] - counters_before["tx_bytes"]
    rx_diff = counters_after["rx_bytes"] - counters_before["rx_bytes"]

    print("Before benchmark:", flush=True)
    print(f"  TX: {tx_before_gb:.2f} GB, RX: {rx_before_gb:.2f} GB", flush=True)
    print("After benchmark:", flush=True)
    print(f"  TX: {tx_after_gb:.2f} GB, RX: {rx_after_gb:.2f} GB", flush=True)
    print("Delta during benchmark:", flush=True)
    print(f"  NVLink TX: {tx_diff / (1024**3):.2f} GB", flush=True)
    print(f"  NVLink RX: {rx_diff / (1024**3):.2f} GB", flush=True)
    print(f"  Total NVLink data: {(tx_diff + rx_diff) / (1024**3):.2f} GB", flush=True)
    print("=" * 60, flush=True)


# Get MPI communicator and rank info
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

# Configuration
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 20  # More iterations for better statistics
DISABLE_P2P = os.environ.get("DISABLE_P2P", "0") == "1"

if world_rank == 0:
    print("Benchmark Configuration:", flush=True)
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}", flush=True)
    print(f"  Benchmark iterations: {BENCHMARK_ITERATIONS}", flush=True)
    print(f"  P2P disabled: {DISABLE_P2P}", flush=True)
    print("=" * 60, flush=True)

# Find a unique device for each rank
local_rank_per_node = world_rank % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()
stream = dev.create_stream()

print(f"Rank {world_rank} using GPU {local_rank_per_node}", flush=True)

# Optionally disable P2P
if DISABLE_P2P:
    for i in range(system.num_devices):
        if i != local_rank_per_node:
            try:
                cp.cuda.runtime.deviceDisablePeerAccess(i)
                if world_rank == 0 and i == 1:  # Only log once
                    print("Disabled P2P access between GPUs", flush=True)
            except Exception:
                pass  # Already disabled or not enabled

# Initialize NVSHMEM with MPI
nvshmem.init(device=dev, mpi_comm=comm, initializer_method="mpi")

my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

print(f"PE {my_pe}/{n_pes}: NVSHMEM initialized successfully", flush=True)

# Allocate symmetric arrays for a simple reduction
# Use a large array to generate measurable NVLink traffic
# H100 has 80GB HBM3, use ~12GB per array to leave headroom for NVSHMEM/NCCL overhead
size = 3_221_225_472  # 3B elements = 12 GiB per array (24 GiB total per GPU)
src_array = nvshmem.array((size,), dtype="float32")
dst_array = nvshmem.array((size,), dtype="float32")

# Initialize source with PE number, destination with zeros
src_array[:] = float(my_pe + 1)
dst_array[:] = 0.0

data_size_mb = len(src_array) * 4 / 1024 / 1024
if my_pe == 0:
    print(
        f"Array size: {len(src_array)} elements ({data_size_mb:.1f} MB per array)",
        flush=True,
    )
    print("=" * 60, flush=True)

# Warmup iterations
if my_pe == 0:
    print(f"Running {WARMUP_ITERATIONS} warmup iterations...", flush=True)

for i in range(WARMUP_ITERATIONS):
    dst_array[:] = 0.0
    nvshmem.reduce(nvshmem.Teams.TEAM_WORLD, dst_array, src_array, "sum", stream=stream)
    stream.sync()

comm.Barrier()  # Synchronize all ranks after warmup

if my_pe == 0:
    print(
        f"Warmup complete. Running {BENCHMARK_ITERATIONS} timed iterations...",
        flush=True,
    )

# Capture NVLink counters before benchmark
nvlink_before = get_nvlink_counters(local_rank_per_node) if my_pe == 0 else None

# Benchmark iterations
timings = []
for i in range(BENCHMARK_ITERATIONS):
    dst_array[:] = 0.0

    # Synchronize before timing
    stream.sync()
    comm.Barrier()

    start = time.perf_counter()
    nvshmem.reduce(nvshmem.Teams.TEAM_WORLD, dst_array, src_array, "sum", stream=stream)
    stream.sync()
    end = time.perf_counter()

    elapsed = end - start
    timings.append(elapsed)

    if my_pe == 0:
        print(f"  Iteration {i + 1}: {elapsed * 1000:.3f} ms", flush=True)

# Capture NVLink counters after benchmark
nvlink_after = get_nvlink_counters(local_rank_per_node) if my_pe == 0 else None

# Gather all timings to rank 0
all_timings = comm.gather(timings, root=0)

if my_pe == 0:
    print("=" * 60, flush=True)
    print("RESULTS:", flush=True)
    print("=" * 60, flush=True)

    # Compute statistics (use rank 0 timings as representative)
    import statistics

    my_timings = all_timings[0]

    mean_time = statistics.mean(my_timings)
    median_time = statistics.median(my_timings)
    min_time = min(my_timings)
    max_time = max(my_timings)
    stdev_time = statistics.stdev(my_timings) if len(my_timings) > 1 else 0.0

    # Calculate effective bandwidth
    # Each reduction moves data across GPUs
    total_data_gb = data_size_mb * n_pes / 1024
    bandwidth_gbps = total_data_gb / mean_time

    print(f"P2P Status: {'DISABLED' if DISABLE_P2P else 'ENABLED'}", flush=True)
    print(f"Number of GPUs: {n_pes}", flush=True)
    print(f"Array size per GPU: {data_size_mb:.1f} MB", flush=True)
    print(f"Total data: {total_data_gb:.2f} GB", flush=True)
    print("", flush=True)
    print(f"Timing Statistics ({BENCHMARK_ITERATIONS} iterations):", flush=True)
    print(f"  Mean:   {mean_time * 1000:.3f} ms", flush=True)
    print(f"  Median: {median_time * 1000:.3f} ms", flush=True)
    print(f"  Min:    {min_time * 1000:.3f} ms", flush=True)
    print(f"  Max:    {max_time * 1000:.3f} ms", flush=True)
    print(f"  Stdev:  {stdev_time * 1000:.3f} ms", flush=True)
    print("", flush=True)
    print(f"Effective Bandwidth: {bandwidth_gbps:.2f} GB/s", flush=True)
    print("=" * 60, flush=True)

    # Verify correctness
    expected_sum = sum(range(1, n_pes + 1))
    actual_sum = float(dst_array[0])
    if abs(actual_sum - expected_sum) < 0.001:
        print("✓ Reduction correctness verified", flush=True)
    else:
        print(
            f"✗ Reduction failed: expected {expected_sum}, got {actual_sum}", flush=True
        )

    # Print NVLink statistics
    print_nvlink_stats("NVLink Data Transfer (GPU 0)", nvlink_before, nvlink_after)

# Clean up symmetric memory before finalizing
nvshmem.free_array(src_array)
nvshmem.free_array(dst_array)

nvshmem.finalize()
