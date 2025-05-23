# ---
# lambda-test: false  # auxiliary-file
# pytest: false
# ---
import argparse
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.run.
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])
# The master (or leader) rank is always 0 with torch.distributed.run.
MASTER_RANK = 0

# This `run` function performs a simple distributed data transfer between containers
# using the specified distributed communication backend.

# An example topology of the cluster when WORLD_SIZE=4 is shown below:
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

# A broadcast operation (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast)
# is performed between the master container (rank 0) and all other containers.

# The master container (rank 0) sends a tensor to all other containers.
# Each container then receives that tensor from the master container.


def run(backend):
    # Helper function providing a vanity name for each container based on its world (i.e. global) rank.
    def container_name(wrld_rank: int) -> str:
        return (
            f"container-{wrld_rank} (main)"
            if wrld_rank == 0
            else f"container-{wrld_rank}"
        )

    tensor = torch.zeros(1)

    # Need to put tensor on a GPU device for NCCL backend.
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == MASTER_RANK:
        print(f"{container_name(WORLD_RANK)} sending data to all other containers...\n")
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print(
                f"{container_name(WORLD_RANK)} sent data to {container_name(rank_recv)}\n"
            )
    else:
        dist.recv(tensor=tensor, src=MASTER_RANK)
        print(
            f"{container_name(WORLD_RANK)} has received data from {container_name(MASTER_RANK)}\n"
        )


# In order for the broadcast operation to happen across the cluster, we need to have the master container (rank 0)
# learn the network addresses of all other containers.

# This is done by calling `dist.init_process_group` with the specified backend.

# See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group for more details.


@contextmanager
def init_processes(backend):
    try:
        dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
        yield
    finally:
        dist.barrier()  # ensure any async work is done before cleaning up
        # Remove this if it causes program to hang. ref: https://github.com/pytorch/pytorch/issues/75097.
        dist.destroy_process_group()


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
    parser.add_argument("--backend", type=str, default="gloo", choices=["nccl", "gloo"])
    args = parser.parse_args()

    with init_processes(backend=args.backend):
        run(backend=args.backend)
