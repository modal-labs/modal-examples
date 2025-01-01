import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.run.
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])
# The master (or leader) rank is always 0 with torch.distributed.run.
MASTER_RANK = 0

TRACE_DIR = Path("/traces")

# This `run` function performs a simple distributed data transfer between containers
# using the specified distributed communication backend.
#
# An example topology on the cluster when WORLD_SIZE=4 is shown below:
#
#        +---------+
#        | Master  |
#        | Rank 0  |
#        +----+----+
#             |
#             |
#    +--------+--------+
#    |        |        |
#    |        |        |
# +--+--+  +--+--+  +--+--+
# |Rank 1| |Rank 2| |Rank 3|
# +-----+  +-----+  +-----+
#
# A broadcast operation (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast)
# is performed between the master container (rank 0) and all other containers.
#
# The master container (rank 0) sends a tensor to all other containers.
# Each container then receives that tensor from the master container.

def run(backend):
    # Helper function providing a vanity name for each container based on its world (i.e. global) rank.
    def container_name(wrld_rank: int) -> str:
        return f"container-{wrld_rank} (master)" if wrld_rank == 0 else f"container-{wrld_rank}"

    tensor = torch.zeros(1)

    # Need to put tensor on a GPU device for NCCL backend.
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == MASTER_RANK:
        print(f"{container_name(WORLD_RANK)} sending data to all other containers...\n")
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print(f"{container_name(WORLD_RANK)} sent data to {container_name(rank_recv)}\n")
    else:
        dist.recv(tensor=tensor, src=MASTER_RANK)
        print(f"{container_name(WORLD_RANK)} has received data from {container_name(MASTER_RANK)}\n")

# In order for the broadcast operation to happen across the cluster, we need to have the master container (rank 0)
# learn the network addresses of all other containers.
#
# This is done by calling `dist.init_process_group` with the specified backend.
#
# See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group for more details.
def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

if __name__ == "__main__":
    # This is a minimal CLI interface adhering to the requirements of torch.distributed.run (torchrun).
    #
    # Our Modal Function will use torch.distributed.run to launch this script.
    #
    # See https://pytorch.org/docs/stable/elastic/run.html for more details on the CLI interface.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--backend", type=str, default="gloo", choices=["nccl", "gloo"]
    )
    args = parser.parse_args()

    init_processes(backend=args.backend)

    from uuid import uuid4

    import torch

    function_name = "demo"
    label = None
    steps = 3
    schedule = None
    record_shapes = False
    profile_memory = False
    with_stack = False
    print_rows = 10

    output_dir = (
        TRACE_DIR
        / (function_name + (f"_{label}" if label else ""))
        / str(uuid4())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if schedule is None:
        if steps < 3:
            raise ValueError(
                "Steps must be at least 3 when using default schedule"
            )
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}

    schedule = torch.profiler.schedule(**schedule)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(steps):
            run(backend=args.backend)
            prof.step()

    if print_rows:
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=print_rows
            )
        )

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    print(f"trace saved to {trace_path.relative_to(TRACE_DIR)}")
