# ---
# lambda-test: false  # long-running
# ---
#
# This script demonstrates ingestion of the [COCO](https://cocodataset.org/#download) (Common Objects in Context)
# dataset.
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
image = modal.Image.debian_slim().apt_install("wget").pip_install("tqdm")
app = modal.App(
    "example-coco",
    image=image,
    secrets=[],
)


def start_monitoring_disk_space(interval: int = 120) -> None:
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


def copy_concurrent(src: pathlib.Path, dest: pathlib.Path) -> None:
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)

        def copy(self, source, dest):
            self.pool.apply_async(shutil.copy2, args=(source, dest))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    with MultithreadedCopier(max_threads=48) as copier:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True)


# This script uses wget to download ZIP files over HTTP because while the official
# website recommends using gsutil to download from a bucket (https://cocodataset.org/#download)
# that bucket no longer exists.


@app.function(
    volumes={"/vol/": volume},
    timeout=60 * 60 * 5,  # 5 hours
    ephemeral_disk=600 * 1024,  # 600 GiB,
)
def _do_part(url: str) -> None:
    start_monitoring_disk_space()
    part = url.replace("http://images.cocodataset.org/", "")
    name = pathlib.Path(part).name.replace(".zip", "")
    zip_path = pathlib.Path("/tmp/") / pathlib.Path(part).name
    extract_tmp_path = pathlib.Path("/tmp", name)
    dest_path = pathlib.Path("/vol/coco/", name)

    print(f"Downloading {name} from {url}")
    command = f"wget {url} -O {zip_path}"
    subprocess.run(command, shell=True, check=True)
    print(f"Download of {name} completed successfully.")
    extract_tmp_path.mkdir()
    extractall(
        zip_path, extract_tmp_path, desc=f"Extracting {name}"
    )  # extract into /tmp/
    zip_path.unlink()  # free up disk space by deleting the zip
    print(f"Copying extract {name} data to volume.")
    copy_concurrent(extract_tmp_path, dest_path)  # copy from /tmp/ into mounted volume


# We can process each part of the dataset in parallel, using a 'parent' Function just to execute
# the map and wait on completion of all children.


@app.function(
    timeout=60 * 60 * 5,  # 5 hours
)
def import_transform_load() -> None:
    print("Starting import, transform, and load of COCO dataset")
    list(
        _do_part.map(
            [
                "http://images.cocodataset.org/zips/train2017.zip",
                "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/zips/test2017.zip",
                "http://images.cocodataset.org/zips/unlabeled2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/image_info_test2017.zip",
                "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
            ]
        )
    )
    print("✅ Done")
