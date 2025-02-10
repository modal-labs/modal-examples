# # Simple PyTorch cluster

# This example shows how you can perform distributed computation with PyTorch.
# It is a kind of 'hello world' example for distributed ML training: setting up a cluster
# and executing a broadcast operation to share a single tensor.

# ## Basic setup: Imports, dependencies, and a script

# Let's get the imports out of the way first.
# We need to import `modal.experimental` to use this feature, since it's still under development.
# Let us know if you run into any issues!

import os
from pathlib import Path

import modal
import modal.experimental

# Communicating between nodes in a cluster requires communication libraries.
# We'll use `torch`, so we add it to our container's [Image](https://modal.com/docs/guide/images) here.

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch~=2.5.1", "numpy~=2.2.1"
)

# The approach we're going to take is to use a Modal [Function](https://modal.com/docs/reference/modal.Function)
# to launch the underlying script we want to distribute over the cluster nodes.
# The script is located in another file in the same directory
# of [our examples repo](https://github.com/modal-labs/modal-examples/).
# In order to use it in our remote Modal Function,
# we need to duplicate it remotely, which we do with `add_local_file`.

this_directory = Path(__file__).parent

image = image.add_local_file(
    this_directory / "simple_torch_cluster_script.py",
    remote_path="/root/script.py",
)

app = modal.App("example-simple-torch-cluster", image=image)

# ## Configuring a test cluster

# First, we set the size of the cluster in containers/nodes. This can be between 1 and 8.
# This is part of our Modal configuration, since Modal is responsible for spinning up our cluster.

n_nodes = 4

# Next, we set the number of processes we run per node.
# The usual practice is to run one process per GPU,
# so we set those two values to be equal.
# Note that `N_GPU` is Modal configuration ("how many GPUs should we spin up for you?")
# while `nproc_per_node` is `torch.distributed` configuration ("how many processes should we spawn for you?").

n_proc_per_node = N_GPU = 1
GPU_CONFIG = f"H100:{N_GPU}"

# Lastly, we need to select our communications library: the software that will handle
# sending messages between nodes in our cluster.
# Since we are running on GPUs, we use the
# [NVIDIA Collective Communications Library](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
# (`nccl`, pronounced "nickle").

# This is part of `torch.distributed` configuration --
# Modal handles the networking infrastructure but not the communication protocol.

backend = "nccl"  # or "gloo" on CPU, see https://pytorch.org/docs/stable/distributed.html#which-backend-to-use

# This cluster configurations is nice for testing, but typically
# you'll want to run a cluster with the maximum number of GPUs per container --
# 8 if you're running on H100s, the beefiest GPUs we offer on Modal.

# ## Launching the script

# Our Modal Function is merely a 'launcher' that sets up the distributed
# cluster environment and then calls `torch.distributed.run`,
# the underlying Python code exposed by the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html)
# command line tool.

# So executing this distributed job is easy! Just run

# ```bash
# modal run simple_torch_cluster.py
# ```

# in your terminal.

# In addition to the values set in code above, you can pass additional arguments to `torch.distributed.run`
# via the command line:

# ```bash
# modal run simple_torch_cluster.py --max-restarts=1
# ```


@app.function(gpu=GPU_CONFIG)
@modal.experimental.clustered(size=n_nodes)
def dist_run_script(*args):
    from torch.distributed.run import parse_args, run

    cluster_info = (  # we populate this data for you
        modal.experimental.get_cluster_info()
    )
    # which container am I?
    container_rank = cluster_info.rank
    # how many containers are in this cluster?
    world_size = len(cluster_info.container_ips)
    # what's the leader/master/main container's address?
    main_addr = cluster_info.container_ips[0]
    # what's the identifier of this cluster task in Modal?
    task_id = os.environ["MODAL_TASK_ID"]
    print(f"hello from {container_rank=}")
    if container_rank == 0:
        print(
            f"reporting cluster state from rank0/main: {main_addr=}, {world_size=}, {task_id=}"
        )

    run(
        parse_args(
            [
                f"--nnodes={n_nodes}",
                f"--node_rank={cluster_info.rank}",
                f"--master_addr={main_addr}",
                f"--nproc-per-node={n_proc_per_node}",
                "--master_port=1234",
            ]
            + list(args)
            + ["/root/script.py", "--backend", backend]
        )
    )
