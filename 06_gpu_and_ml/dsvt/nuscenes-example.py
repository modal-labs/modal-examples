# # Training DSVT on NuScenes with Modal

# ## Setup

from pathlib import Path, PurePosixPath

import modal


# ### NuScenes Dataset Setup
# In this demo we are only setting up the `v1.0-mini` version of the dataset.
# To download NuScenes you need to make an account here: https://www.nuscenes.org/sign-up?prevpath=nuscenes&prevhash=download
# Then upload your username and password as a [Modal Secret](https://modal.com/secrets/). Click the Custom secret
# button, name the secret `nuscenes`, then enter two key/value pairs: [NUSCENES_USERNAME, NUSCENES_PASSWORD].
# Then the following line will import these values as environment variables:
nuscenes_secret = modal.Secret.from_name("nuscenes")

# We will download the data into a Modal.Volume with this name:
vol_name = "example-nuscenes2"
# This is the location within the container where this Volume will be mounted:
vol_mnt = Path("/data")
vol_data_subdir = "nuscenes"  # data subdir within the volume
nuscenes_posix = (vol_mnt / vol_data_subdir).as_posix()

# Create (or ID) the Volume object:
nuscenes_volume = modal.Volume.from_name(vol_name, create_if_missing=True)

# ### Define the image
nuscenes_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu20.04", add_python="3.9")
    .env(
        {  # Some environment variable needed to compile the libs in the image
            "DEBIAN_FRONTEND": "noninteractive",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
            "CXX": "g++",
            "CC": "gcc",
        }
    )
    .apt_install(["git", "python3-opencv", "build-essential", "ninja-build", "clang"])
    .run_commands("pip install --upgrade pip")
    .pip_install(["uv"])
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "'numpy==1.23.5' 'scikit-image<=0.21.0' "
        "'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' 'torchaudio==2.0.2+cu118' "
        "--index-url https://download.pytorch.org/whl/cu118 "
        "--extra-index-url https://pypi.org/simple"
    )
    .run_commands(
        "uv pip install --system --no-build-isolation spconv-cu118 torch-scatter "
        "-f https://data.pyg.org/whl/torch-2.0.1+cu118.html"
    )
    .run_commands(
        "uv pip install --system tensorrt onnx pyyaml 'nuscenes-devkit==1.0.5'"
    )
    # NOTE: You could instead import from local with add_local_dir:
    # https://modal.com/docs/guide/images#add-local-files-with-add_local_dir-and-add_local_file
    #
    .run_commands("git clone https://github.com/beijbom/DSVT.git")
    .run_commands("uv pip install --system --no-build-isolation -e DSVT")
    .run_commands("uv pip install --system 'mmcv>=1.4.0,<2.0.0'")
    # NOTE: might be possible to use DSVT's internal copy of pcdet?
    .run_commands("git clone https://github.com/open-mmlab/OpenPCDet.git")
    .run_commands("uv pip install --system --no-build-isolation -e OpenPCDet")
    .run_commands("uv pip install --system 'av2==0.2.0' 'kornia<0.7'")
    .entrypoint([])
)

# Initialize the app
app = modal.App(
    "train-nuscenes",
    image=nuscenes_image,
    volumes={vol_mnt: nuscenes_volume},
)

# ## NuScenes Automated Downloading + Preprocessing
# This function automagically downloads and preprocesses the NuScenes dataset.
# Tested with the v1.0-mini partition only -- the massive v1.0-train will take a
# long time (you may want to make a different Modal Volume for each subset).


# We cap max_containers with 1 here, but you could distribute the download
# and preprocessing over subsets of v1.0-train and v1.0-test.
@app.function(
    image=nuscenes_image,
    secrets=[nuscenes_secret],
    volumes={vol_mnt: nuscenes_volume},
    timeout=3 * 60 * 60,  # processing v1.0-mini takes 0.5-1.5 hours
    gpu="A10G",
    max_containers=1,
)
def download_nuscenes(
    volume_subdir: str,
    region: str = "us",  # or "asia"
    dataset_version: str = "v1.0-mini",
):
    """
    Automated download inspired by:
    https://github.com/li-xl/nuscenes-download/blob/master/download_nuscenes.py
    """
    import json
    import os
    import subprocess
    import sys
    import tarfile

    import requests
    from tqdm import tqdm

    download_dir = vol_mnt / volume_subdir
    tgz_file = download_dir / f"{dataset_version}.tgz"
    extract_dir = download_dir / dataset_version
    info_prefix = "nuscenes"

    # (0) Download .tgz from AWS:
    if not tgz_file.is_file():
        download_dir.mkdir(parents=True, exist_ok=True)
        # (1a) Get login from Modal Secret
        if not os.getenv("NUSCENES_USERNAME") or not os.getenv("NUSCENES_PASSWORD"):
            print(
                "Error: set NUSCENES_USERNAME and NUSCENES_PASSWORD in your env",
                file=sys.stderr,
            )
            sys.exit(1)

        # (1b) Log in via Cognito to get a bearer token
        print(f"Setting up Nuscenes mini dataset in `{vol_name}/{volume_subdir}`.")
        resp = requests.post(
            "https://cognito-idp.us-east-1.amazonaws.com/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            data=json.dumps(
                {
                    "AuthFlow": "USER_PASSWORD_AUTH",
                    "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
                    "AuthParameters": {
                        "USERNAME": os.getenv("NUSCENES_USERNAME"),
                        "PASSWORD": os.getenv("NUSCENES_PASSWORD"),
                    },
                }
            ),
        )
        resp.raise_for_status()
        token = resp.json()["AuthenticationResult"]["IdToken"]
        print("\tLogged in successfully")

        # (1c) Fetch the mini archive URL
        api = (
            f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1"
            f"/archives/v1.0/{tgz_file.name}?region={region}&project=nuScenes"
        )
        resp = requests.get(api, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        download_url = resp.json()["url"]
        print(f"\tGot download URL for {tgz_file.name}")

        # (1d) Download into download_dir
        os.makedirs(download_dir, exist_ok=True)

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(tgz_file, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\tDownloaded to {tgz_file}")
    else:
        print(f"\t.tgz archive found at: {tgz_file}")

    # (2) Extract the archive in-place
    n_files = len(list(extract_dir.glob("*")))
    if n_files < 3:
        print(f"\tExtracting to {extract_dir}")
        with tarfile.open(tgz_file, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting", unit="file"):
                tar.extract(member, path=extract_dir)  # NOW extract
        print("\tExtraction complete")
    else:
        print(
            f"\t{n_files} found at {extract_dir}, assuming archive already extracted...!"
        )

    # (3) Generate metadata with nuScenes devkit
    pickles = [
        extract_dir / f"{info_prefix}_infos_10sweeps_train.pkl",
        extract_dir / f"{info_prefix}_dbinfos_10sweeps_withvelo.pkl",
        extract_dir / f"{info_prefix}_infos_10sweeps_val.pkl",
    ]
    if all([x.is_file() for x in pickles]):
        print("\tpickle files found!")
    else:
        print("\tGenerating devkit metadata pickles...")
        os.chdir("/OpenPCDet")  # TODO: try using DSVT's copy inside its repo..
        cmd = [
            sys.executable,
            "-m",
            "pcdet.datasets.nuscenes.nuscenes_dataset",
            "--func",
            "create_nuscenes_infos",
            "--cfg_file",
            "tools/cfgs/dataset_configs/nuscenes_dataset.yaml",
            "--version",
            f"{dataset_version}",
            "--with_cam",
        ]
        subprocess.run(cmd, check=True)

    print(f"\tNuScenes {dataset_version} is ready in: {extract_dir}")


# Train:
@app.cls(
    image=nuscenes_image,
    volumes={vol_mnt: nuscenes_volume},
    timeout=24 * 60 * 60,
    max_containers=1,
)
class DSVTTrainer:
    @modal.method()
    def default_nuscenes_config_setup(
        self,
        tag: str = "",
        data_ver: str = "v1.0-mini",
        model_name: str = "dsvt_plain_1f_onestage_nusences",
        data_name: str = "nuscenes_dataset",
        config_save_dir="saved-configs",
    ):
        # Data catalogs
        from yaml import safe_dump, safe_load

        # Directories
        tools = Path("/DSVT") / "tools"
        model_configs = tools / "cfgs/dsvt_models"
        data_configs = tools / "cfgs/dataset_configs"

        # Template config defaults:
        template_model_path = model_configs / f"{model_name}.yaml"
        template_data_path = data_configs / f"{data_name}.yaml"

        # Configs we'll create & use
        savedir = vol_mnt / config_save_dir
        savedir.mkdir(exist_ok=True, parents=True)
        output_data_path = savedir / f"{tag}-data-config.yaml"
        output_model_path = savedir / f"{tag}-model-config.yaml"

        # Data catalogs
        data_dir = vol_mnt / vol_data_subdir / data_ver
        train_data = data_dir / "nuscenes_infos_10sweeps_train.pkl"
        val_data = data_dir / "nuscenes_infos_10sweeps_val.pkl"
        velo_data = data_dir / "nuscenes_dbinfos_10sweeps_withvelo.pkl"

        ########################################################
        # (1) Edit the data config

        with open(template_data_path, "r") as f:
            data_config = safe_load(f)

        data_config["DATA_PATH"] = (vol_mnt / vol_data_subdir).as_posix()
        data_config["VERSION"] = data_ver
        data_config["INFO_PATH"] = {
            "train": [train_data.as_posix()],
            "test": [val_data.as_posix()],
        }

        data_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][0]["DB_INFO_PATH"] = [
            velo_data.as_posix()
        ]

        with open(output_data_path, "w") as f:
            safe_dump(data_config, f)

        ########################################################
        # (2) Edit model config
        with open(template_model_path, "r") as f:
            model_config = safe_load(f)
        model_config["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][0][
            "DB_INFO_PATH"
        ] = [velo_data.as_posix()]
        # Point model config to our new data config
        model_config["DATA_CONFIG"]["_BASE_CONFIG_"] = output_data_path.as_posix()

        with open(output_model_path, "w") as f:
            safe_dump(model_config, f)

        print(
            f"Using configs:"
            f"\n\tmodel: {output_model_path} (exists: {output_data_path.is_file()})"
            f"\n\tdata: {output_data_path} (exist: {output_data_path.is_file()})"
        )
        return output_model_path.as_posix()

    @modal.method()
    def train(
        self,
        exp_name: str,
        model_config_path: str,
        params: dict = {},
    ):
        import os

        import torch

        # Prepare inputs
        flags = " ".join([f"--{arg} {val}" for arg, val in params.items()])
        cmd = (
            f"torchrun "
            f"--nproc_per_node={torch.cuda.device_count()} "
            f"/DSVT/tools/train.py --launcher none "
            f"--cfg_file {model_config_path} " + flags
        )
        if exp_name:
            print(f"Running exp {exp_name} with command:\n\t{cmd}")
        os.chdir("/DSVT/tools")
        os.system(cmd)


@app.local_entrypoint()
def main():
    # Add as CLI:
    n_gpus = 1
    gpu = "A100"
    data_ver = "v1.0-mini"

    # (0) Check for data before firing up the downloader/preprocessor container
    # Check if the necessary pickle files exist:
    pickles = [
        "nuscenes_infos_10sweeps_train.pkl",
        "nuscenes_infos_10sweeps_val.pkl",
        "nuscenes_dbinfos_10sweeps_withvelo.pkl",
    ]

    # TODO: better way to check if subdir exists?
    run_downloader = False
    try:
        paths = [
            Path(x.path).name
            for x in nuscenes_volume.listdir(f"{vol_data_subdir}/{data_ver}")
            if x.path.endswith("pkl")
        ]
    except Exception as err:
        # Error if listdir called on non-existent vol
        run_downloader = True

    for p in pickles:
        if p not in paths:
            run_downloader = True

    if run_downloader:
        download_nuscenes.remote(vol_data_subdir)
    else:
        print("Dataset pickles found, skipping download etc.")

    # (1) Data identified! Call trainer.
    trainer = DSVTTrainer.with_options(gpu=f"{gpu}:{n_gpus}")()
    # Replace this with your custom config setup:
    exp_name = "single-gpu-demo"
    exp_config = trainer.default_nuscenes_config_setup.remote(
        tag=exp_name, data_ver=data_ver
    )
    trainer.train.remote(
        exp_name=exp_name, model_config_path=exp_config, params={"epochs", 1}
    )
