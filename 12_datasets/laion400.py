import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import modal

bucket_creds = modal.Secret.from_name("aws-s3-modal-examples-datasets", environment_name="main")
bucket_name = "modal-examples-datasets"
volume = modal.CloudBucketMount(
    bucket_name,
    secret=bucket_creds,
)
image = modal.Image.debian_slim().apt_install("wget").pip_install("img2dataset~=1.45.0")
app = modal.App("example-laoin400-dataset-import", image=image)

# This script is base off the following instructions:
# https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md

def start_monitoring_disk_space(interval: int = 30) -> None:
    """Start monitoring the disk space in a separate thread."""
    task_id = os.environ["MODAL_TASK_ID"]
    def log_disk_space(interval: int) -> None:
        while True:
            statvfs = os.statvfs('/')
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(f"{task_id} free disk space: {free_space / (1024 ** 3):.2f} GB", file=sys.stderr)
            time.sleep(interval)

    monitoring_thread = threading.Thread(target=log_disk_space, args=(interval,))
    monitoring_thread.daemon = True
    monitoring_thread.start()

@app.function(
    volumes={"/mnt": volume},
    timeout=60 * 60 * 12,  # 12 hours
)
def run_img2dataset_on_part(
    i: int,
    partfile: str,
) -> None:
    start_monitoring_disk_space()
    # Each part works in its own subdirectory because img2dataset creates a working
    # tmpdir at <output_folder>/_tmp and we don't want consistency issues caused by
    # all concurrently processing parts read/writing from the same temp directory.
    laion400m_data_path = pathlib.Path(f"/mnt/laion400/laion400m-data/{i}/")
    command = (
        f'img2dataset --url_list {partfile} --input_format "parquet" '
        '--url_col "URL" --caption_col "TEXT" --output_format webdataset '
        f'--output_folder {laion400m_data_path} --processes_count 16 --thread_count 128 --image_size 256 '
        '--save_additional_columns \'["NSFW","similarity","LICENSE"]\' --enable_wandb False'
    )
    print(f"Running img2dataset command: \n\n{command}")
    subprocess.run(command, shell=True, check=True)


@app.function(
    volumes={"/mnt": volume},
    timeout=60 * 60 * 16,  # 16 hours
)
def import_transform_load() -> None:
    start_monitoring_disk_space()
    # We initially download into a tmp directory outside of the volume to avoid
    # any filesystem incompatibilities between the `wget` application and the bucket mount
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

        # Perform a simple copy operation to move the data into the bucket.
        shutil.copytree(tmp_laion400m_meta_path, laion400m_meta_path)

    parquet_files = list(laion400m_meta_path.glob("**/*.parquet"))
    print(f"Downloaded {len(parquet_files)} parquet files.")
    print(f"Spawning {len(parquet_files)} to enrich dataset...")
    list(run_img2dataset_on_part.starmap((i, f) for i, f in enumerate(parquet_files)))
