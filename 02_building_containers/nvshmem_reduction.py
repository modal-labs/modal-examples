import mpi4py.MPI as MPI
import nvshmem.core as nvshmem
from cuda.core.experimental import Device, system

# Get MPI communicator and rank info
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

print(f"MPI rank {world_rank}/{world_size} starting", flush=True)

# Find a unique device for each rank
local_rank_per_node = world_rank % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()
stream = dev.create_stream()

print(f"Rank {world_rank} using GPU {local_rank_per_node}", flush=True)

# Initialize NVSHMEM with MPI
nvshmem.init(device=dev, mpi_comm=comm, initializer_method="mpi")

my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

print(f"PE {my_pe}/{n_pes}: NVSHMEM initialized successfully", flush=True)

# Allocate symmetric arrays for a simple reduction
# Use a large array to generate measurable NVLink traffic
size = 268_435_456  # 256M elements = 1 GiB per array
src_array = nvshmem.array((size,), dtype="float32")
dst_array = nvshmem.array((size,), dtype="float32")

# Initialize source with PE number, destination with zeros
src_array[:] = float(my_pe + 1)  # PE 0 -> [1,1,1,1], PE 1 -> [2,2,2,2]
dst_array[:] = 0.0

print(
    f"PE {my_pe}: src_array[0:4] = {src_array[0:4]} (size={len(src_array)} elements, {len(src_array) * 4 / 1024 / 1024:.1f} MB)",
    flush=True,
)

# Perform a sum reduction across all PEs
nvshmem.reduce(nvshmem.Teams.TEAM_WORLD, dst_array, src_array, "sum", stream=stream)
stream.sync()

print(f"PE {my_pe}: After reduction, dst_array[0:4] = {dst_array[0:4]}", flush=True)

# Verify the result (sum of 1+2+...+n_pes for each element)
expected_sum = sum(range(1, n_pes + 1))
if my_pe == 0:  # Root PE gets the reduction result
    actual_sum = float(dst_array[0])
    print(
        f"PE {my_pe}: Expected sum = {expected_sum}, Actual = {actual_sum}", flush=True
    )
    if abs(actual_sum - expected_sum) < 0.001:
        print(f"PE {my_pe}: ✓ Reduction successful!", flush=True)
    else:
        print(f"PE {my_pe}: ✗ Reduction failed!", flush=True)

# Clean up symmetric memory before finalizing
nvshmem.free_array(src_array)
nvshmem.free_array(dst_array)

nvshmem.finalize()
print(f"PE {my_pe}: Finalized", flush=True)
