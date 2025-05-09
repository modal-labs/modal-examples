import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterator, List, Sequence, Tuple

import modal

n_gpus = 2
vol_name = "nuscenes"
vol_mnt = Path("/data")
dataset_path = (vol_mnt / "nuscenes-download/dataset").as_posix()

TH_CACHE_DIR = vol_mnt / "model-compile-cache"
th_compile_kwargs = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}

data_volume = modal.Volume.from_name(vol_name, create_if_missing=True)

repo_dst = "/DSVT"

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu20.04", add_python="3.10")
    .env(
        {
            # env vars needed to compile the project
            "REPO_DIR": f"{repo_dst}",
            "DEBIAN_FRONTEND": "noninteractive",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
            "CXX": "g++",
            "CC": "gcc",
        }
    )
    .apt_install(["git", "build-essential", "ninja-build", "clang"])
    .pip_install(["uv"])
    .pip_install("uv")
    .run_commands(
        "pip3 install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    .run_commands(
        "uv pip install --system spconv-cu118 torch-scatter "
        "-f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
    )
    .run_commands(f"git clone https://github.com/beijbom/DSVT.git {repo_dst}")
    .run_commands(f"cd {repo_dst}  && python setup.py develop")
    # Stuff that setup.py fails to install:
    .run_commands("uv pip install --system tensorrt onnx")
    .run_commands("uv pip install --system pyyaml")
    .env(
        {
            # env vars used within the program
            "NGPUS": f"{n_gpus}",
            "LOCAL_RANK": "0",
        }
    )
    .entrypoint([])
)

# Initialize the app
app = modal.App(
    "train-nuscenes",
    image=image,
    volumes={vol_mnt: data_volume},
)


def add_fake_info(yaml_config):
    data_path = Path(dataset_path)
    for idx, item in enumerate(yaml_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"]):
        if item["NAME"] == "gt_sampling":
            new = []
            for path in yaml_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][idx][
                "DB_INFO_PATH"
            ]:
                actual_path = data_path / path
                if not actual_path.exists():
                    raise ValueError(f"Path ain't exist~\n\t{actual_path}")
                else:
                    new.append((dataset_path / x).as_posix())
                yaml_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][idx] = new

    #         yaml_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][idx]["BACKUP_DB_INFO"] = {
    #             "DB_INFO_PATH": "a",
    #             "DB_DATA_PATH": "b",
    #             "NUM_POINT_FEATURES": 23,
    #         }
    return yaml_config


# Imports inside the container
# with th_compile_image.imports():
@app.function(
    image=image,
    volumes={vol_mnt: data_volume},
    gpu=f"A100:{n_gpus}",
    # buffer_containers=1,
    max_containers=1,
)
def do(data_config):
    from yaml import safe_dump, safe_load

    # --------------------------------
    # 0) Paths
    dsvt = Path(os.environ["REPO_DIR"])
    tools = dsvt / "tools"
    script = (tools / "train.py").as_posix()
    cfg_path = tools / "cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences.yaml"
    tmp_datacfg_path = tools / "cfgs/dataset_configs/ben-nuscenes.yaml"

    # 1) Load main model config
    with open(cfg_path, "r") as f:
        cfg = safe_load(f)

    # 2) Inject our temporary data‐config path
    cfg["DATA_CONFIG"]["_BASE_CONFIG_"] = tmp_datacfg_path.as_posix()
    cfg["DATA_CONFIG"] = add_fake_info(cfg["DATA_CONFIG"])
    data_config = add_fake_info(data_config)

    # 3) Write the updated main config back
    with open(cfg_path, "w") as f:
        safe_dump(cfg, f)

    # 4) Write the user‐passed data_config dict to the tmp file
    with open(tmp_datacfg_path, "w") as f:
        safe_dump(data_config, f)

    # Execute
    os.chdir(tools)
    os.system(
        f"torchrun "
        f"--nproc_per_node={n_gpus} "
        f"{script} --launcher pytorch --cfg_file {cfg_path} "
    )


# other example args:
# port = 29500
# f"--rdzv_backend=c10d "
# f"--rdzv_endpoint=localhost:{port} "


@app.local_entrypoint()
def main():
    # TODO:
    # add dataset /volume setup script with NuScenes username/pw secret

    data_config_path = Path(
        "~/github/DSVT/tools/cfgs/dataset_configs/nuscenes_dataset.yaml"
    ).expanduser()

    from yaml import safe_load

    # Load local YAML config
    with open(data_config_path, "r") as f:
        data_cfg = safe_load(f)

    # Send to container
    do.remote(data_cfg)
