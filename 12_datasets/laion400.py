# ---
# deploy: true
# lambda-test: false
# ---
#
# https://laion.ai/blog/laion-400-open-dataset/
#
# LAION-400 is a large dataset of 400M English (image, text) pairs.
#
# As described on the dataset's homepage, it consists of 32 .parquet files
# containing dataset metadata *but not* the image data itself.
#
# After downloading the .parquet files, this script fans out 32 worker jobs
# to process a single .parquet file. Processing involves fetch and transform
# of image data into 256 * 256 square JPEGs.
#
# This script is loosely based off the following instructions:
# https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md
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

import modal

bucket_creds = modal.Secret.from_name(
    "aws-s3-modal-examples-datasets", environment_name="main"
)

bucket_name = "modal-examples-datasets"

volume = modal.CloudBucketMount(
    bucket_name,
    secret=bucket_creds,
)

image = (
    modal.Image.debian_slim()
    .apt_install("wget")
    .pip_install("img2dataset~=1.45.0")
)

app = modal.App("example-laion400-dataset-import", image=image)


def start_monitoring_disk_space(interval: int = 30) -> None:
    """Start monitoring the disk space in a separate thread, printing info to stdout"""
    task_id = os.environ["MODAL_TASK_ID"]

    def log_disk_space(interval: int) -> None:
        while True:
            statvfs = os.statvfs("/")
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(
                f"{task_id} free disk space: {free_space / (1024 ** 3):.2f} GB",
                file=sys.stderr,
            )
            time.sleep(interval)

    monitoring_thread = threading.Thread(
        target=log_disk_space, args=(interval,)
    )
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
        shutil.copytree(
            src, dest, copy_function=copier.copy, dirs_exist_ok=True
        )


@app.function(
    volumes={"/mnt": volume},
    # 20 hours — img2dataset is extremely slow to work through all images.
    timeout=60 * 60 * 20,
    ephemeral_disk=512 * 1024,
)
def run_img2dataset_on_part(
    i: int,
    partfile: str,
) -> None:
    start_monitoring_disk_space(interval=60)
    while not pathlib.Path(partfile).exists():
        print(f"{partfile} not yet visible...", file=sys.stderr)
        time.sleep(1)
    # Each part works in its own subdirectory because img2dataset creates a working
    # tmpdir at <output_folder>/_tmp and we don't want consistency issues caused by
    # all concurrently processing parts read/writing from the same temp directory.
    tmp_laion400m_data_path = pathlib.Path(f"/tmp/laion400/laion400m-data/{i}/")
    tmp_laion400m_data_path.mkdir(exist_ok=True, parents=True)
    # Increasing retries comes at a *large* performance cost.
    retries = 0
    # TODO: Support --incremental mode. https://github.com/rom1504/img2dataset?tab=readme-ov-file#incremental-mode
    command = (
        f'img2dataset --url_list {partfile} --input_format "parquet" '
        '--url_col "URL" --caption_col "TEXT" --output_format webdataset '
        f"--output_folder {tmp_laion400m_data_path} --processes_count 16 --thread_count 128 --image_size 256 "
        f'--retries={retries} --save_additional_columns \'["NSFW","similarity","LICENSE"]\' --enable_wandb False'
    )
    print(f"Running img2dataset command: \n\n{command}")
    subprocess.run(command, shell=True, check=True)
    print("Completed img2dataset, copying into mounted volume...")
    laion400m_data_path = pathlib.Path("/mnt/laion400/laion400m-data/")
    copy_concurrent(tmp_laion400m_data_path, laion400m_data_path)


@app.function(
    volumes={"/mnt": volume},
    timeout=60 * 60 * 16,  # 16 hours
)
def import_transform_load() -> None:
    start_monitoring_disk_space()
    # We initially download into a tmp directory outside of the volume to avoid
    # any filesystem incompatibilities between the `wget` application and the bucket
    # filesystem mount.
    tmp_laion400m_meta_path = pathlib.Path("/tmp/laion400/laion400m-meta")
    laion400m_meta_path = pathlib.Path("/mnt/laion400/laion400m-meta")
    if not laion400m_meta_path.exists():
        laion400m_meta_path.mkdir(parents=True, exist_ok=True)
        # WARNING: We skip the certificate check for the-eye.eu because its TLS certificate expired as of mid-May 2024.
        subprocess.run(
            f"wget -l1 -r --no-check-certificate --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ -P {tmp_laion400m_meta_path}",
            shell=True,
            check=True,
        )

        parquet_files = list(tmp_laion400m_meta_path.glob("**/*.parquet"))
        print(
            f"Downloaded {len(parquet_files)} parquet files into {tmp_laion400m_meta_path}."
        )
        # Perform a simple copy operation to move the data into the bucket.
        copy_concurrent(tmp_laion400m_meta_path, laion400m_meta_path)

    parquet_files = list(laion400m_meta_path.glob("**/*.parquet"))
    print(
        f"Stored {len(parquet_files)} parquet files into {laion400m_meta_path}."
    )
    print(f"Spawning {len(parquet_files)} to enrich dataset...")
    list(
        run_img2dataset_on_part.starmap(
            (i, f) for i, f in enumerate(parquet_files)
        )
    )
