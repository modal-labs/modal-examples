import os

import modal
import modal.experimental

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
app = modal.App("example-simple-torch-cluster", image=image)

gpu = False
backend = "gloo" if not gpu else "nccl"
n_nodes = 4
n_proc_per_node = 1

@app.function(
    gpu="any" if gpu else None,
    mounts=[modal.Mount.from_local_file("simple_torch_cluster_script.py", remote_path="/root/script.py")],
)
@modal.experimental.clustered(size=n_nodes)
def main():
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()
    container_rank = cluster_info.rank
    main_addr = cluster_info.container_ips[0]
    world_size = len(cluster_info.container_ips)
    task_id = os.environ["MODAL_TASK_ID"]
    print(f"{container_rank=}, {main_addr=}, {world_size=}, {task_id=}")

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
