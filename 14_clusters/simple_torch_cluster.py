# ---
# lambda-test: false
# ---
# # Simple PyTorch cluster

# This example shows how you can performance distributed computation with PyTorch.
# It is a kind of 'hello world' example for distributed ML training, setting up a cluster
# a performing a trivial broadcast operation to share a single tensor.

# ## Basic setup
# Let's get the imports out of the way and define an [`App`](https://modal.com/docs/reference/modal.App).

import os

import modal
import modal.experimental

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.5.1", "numpy~=2.2.1")
    # Mount the script that performs the actual distributed computation.
    # Our modal.Function is merely a 'launcher' that sets up the distributed
    # cluster environment and then calls torch.distributed.run with desired arguments.
    .add_local_file(
        "simple_torch_cluster_script.py", remote_path="/root/script.py"
    )
)
app = modal.App("example-simple-torch-cluster", image=image)

# Some basic configuration allows for demoing either a CPU-only cluster or a GPU-enabled cluster
# with one GPU per container. These cluster configurations are helpful for testing, but typically
# you'll want to run a cluster with 8 GPUs per container, each GPU serving its own local 'worker' process.

# https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
backend = "nccl"
# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 4
# Typically this matches the number of GPUs per container.
n_proc_per_node = 1


@app.function(gpu=modal.gpu.H100())
@modal.experimental.clustered(size=n_nodes)
def demo():
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()
    container_rank = cluster_info.rank
    main_addr = cluster_info.container_ips[0]
    world_size = len(cluster_info.container_ips)
    task_id = os.environ["MODAL_TASK_ID"]
    print(
        f"hello from {container_rank=}, {main_addr=}, {world_size=}, {task_id=}"
    )

    run(
        parse_args(
            [
                f"--nnodes={n_nodes}",
                f"--node_rank={cluster_info.rank}",
                f"--master_addr={cluster_info.container_ips[0]}",
                f"--nproc-per-node={n_proc_per_node}",
                "--master_port=1234",
                "/root/script.py",
                "--backend",
                backend,
            ]
        )
    )
