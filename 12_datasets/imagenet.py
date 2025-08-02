# ---
# lambda-test: false  # long-running
# ---
#
# This scripts demonstrates how to ingest the famous ImageNet (https://www.image-net.org/)
# dataset into a mounted volume.
#
# It requires a Kaggle account's API token stored as a modal.Secret in order to download part
# of the dataset from Kaggle's servers using the `kaggle` CLI.
#
# It is recommended to iterate on this code from a modal.Function running Jupyter server.
# This better supports experimentation and maintains state in the face of errors:
# 11_notebooks/jupyter_inside_modal.py
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import modal

bucket_creds = modal.Secret.from_name(
    "aws-s3-modal-examples-datasets", environment_name="main"
)
bucket_name = "modal-examples-datasets"
volume = modal.CloudBucketMount(
    bucket_name,
    secret=bucket_creds,
)
image = modal.Image.debian_slim().apt_install("tree").pip_install("kaggle", "tqdm")
app = modal.App(
    "example-imagenet",
    image=image,
    secrets=[modal.Secret.from_name("kaggle-api-token")],
)


def start_monitoring_disk_space(interval: int = 30) -> None:
    """Start monitoring the disk space in a separate thread."""
    task_id = os.environ["MODAL_TASK_ID"]

    def log_disk_space(interval: int) -> None:
        while True:
            statvfs = os.statvfs("/")
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(
                f"{task_id} free disk space: {free_space / (1024**3):.2f} GB",
                file=sys.stderr,
            )
            time.sleep(interval)

    monitoring_thread = threading.Thread(target=log_disk_space, args=(interval,))
    monitoring_thread.daemon = True
    monitoring_thread.start()


def copy_concurrent(src: pathlib.Path, dest: pathlib.Path) -> None:
    """
    A modified shutil.copytree which copies in parallel to increase bandwidth
    and compensate for the increased IO latency of volume mounts.
    """
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)
            self.copy_jobs = []

        def copy(self, source, dest):
            res = self.pool.apply_async(
                shutil.copy2,
                args=(source, dest),
                callback=lambda r: print(f"{source} copied to {dest}"),
                # NOTE: this should `raise` an exception for proper reliability.
                error_callback=lambda exc: print(
                    f"{source} failed: {exc}", file=sys.stderr
                ),
            )
            self.copy_jobs.append(res)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    with MultithreadedCopier(max_threads=24) as copier:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True)


def extractall(fzip, dest, desc="Extracting"):
    from tqdm.auto import tqdm
    from tqdm.utils import CallbackIOWrapper

    dest = pathlib.Path(dest).expanduser()
    with (
        zipfile.ZipFile(fzip) as zipf,
        tqdm(
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
        ) as pbar,
    ):
        for i in zipf.infolist():
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, os.fspath(dest))
            else:
                full_path = dest / i.filename
                full_path.parent.mkdir(exist_ok=True, parents=True)
                with zipf.open(i) as fi, open(full_path, "wb") as fo:
                    shutil.copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)


@app.function(
    volumes={"/mnt/": volume},
    timeout=60 * 60 * 8,  # 8 hours,
    ephemeral_disk=1000 * 1024,  # 1TB
)
def import_transform_load() -> None:
    start_monitoring_disk_space()
    kaggle_api_token_data = os.environ["KAGGLE_API_TOKEN"]
    kaggle_token_filepath = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    kaggle_token_filepath.parent.mkdir(exist_ok=True)
    kaggle_token_filepath.write_text(kaggle_api_token_data)

    tmp_path = pathlib.Path("/tmp/imagenet/")
    vol_path = pathlib.Path("/mnt/imagenet/")
    filename = "imagenet-object-localization-challenge.zip"
    dataset_path = vol_path / filename
    if dataset_path.exists():
        dataset_size = dataset_path.stat().st_size
        if dataset_size < (150 * 1024 * 1024 * 1024):
            dataset_size_gib = dataset_size / (1024 * 1024 * 1024)
            raise RuntimeError(
                f"Partial download of dataset .zip. It is {dataset_size_gib}GiB but should be > 150GiB"
            )
    else:
        subprocess.run(
            f"kaggle competitions download -c imagenet-object-localization-challenge --path {tmp_path}",
            shell=True,
            check=True,
        )
        vol_path.mkdir(exist_ok=True)
        shutil.copy(tmp_path / filename, dataset_path)

    # Extract dataset
    extracted_dataset_path = tmp_path / "extracted"
    extracted_dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"Extracting .zip into {extracted_dataset_path}...")
    extractall(dataset_path, extracted_dataset_path)
    print(f"Extracted {dataset_path} to {extracted_dataset_path}")
    subprocess.run(f"tree -L 3 {extracted_dataset_path}", shell=True, check=True)

    final_dataset_path = vol_path / "extracted"
    final_dataset_path.mkdir(exist_ok=True)
    copy_concurrent(extracted_dataset_path, final_dataset_path)
    subprocess.run(f"tree -L 3 {final_dataset_path}", shell=True, check=True)
    print("Dataset is loaded ✅")
