# ---
# lambda-test: false  # long-running
# ---
#
# This script demonstrated how to ingest the https://github.com/RosettaCommons/RoseTTAFold protein-folding
# model's dataset into a mounted volume.

# The dataset is over 2 TiB when decompressed to the runtime of this script is quite long.
# ref: https://github.com/RosettaCommons/RoseTTAFold/issues/132.
#
# It is recommended to iterate on this code from a modal.Function running Jupyter server.
# This better supports experimentation and maintains state in the face of errors:
# 11_notebooks/jupyter_inside_modal.py
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
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
image = modal.Image.debian_slim().apt_install("wget")
app = modal.App("example-rosettafold", image=image)


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


def decompress_tar_gz(file_path: pathlib.Path, extract_dir: pathlib.Path) -> None:
    print(f"Decompressing {file_path} into {extract_dir}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        print(f"Decompressed {file_path} to {extract_dir}")


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


@app.function(
    volumes={"/mnt/": volume},
    timeout=60 * 60 * 24,
    ephemeral_disk=2560 * 1024,
)
def _do_part(url: str) -> None:
    name = url.split("/")[-1].replace(".tar.gz", "")
    print(f"Downloading {name}")
    compressed = pathlib.Path("/tmp", name)
    cmd = f"wget {url} -O {compressed}"
    p = subprocess.Popen(cmd, shell=True)
    returncode = p.wait()
    if returncode != 0:
        raise RuntimeError(f"Error in downloading. {p.args!r} failed {returncode=}")
    decompressed = pathlib.Path("/tmp/rosettafold/", name)

    # Decompression is much faster against the container's local SSD disk
    # compared with against the mounted volume. So we first compress into /tmp/.
    print(f"Decompressing {compressed} into {decompressed}.")
    decompress_tar_gz(compressed, decompressed)
    print(
        f"✅ Decompressed {compressed} into {decompressed}. Now deleting it to free up disk.."
    )
    compressed.unlink()  # delete compressed file to free up disk

    # Finally, we move the decompressed data from /tmp/ into the mounted volume.
    # There are a large mount of files to copy so this step takes a while.
    dest = pathlib.Path("/mnt/rosettafold/")
    copy_concurrent(decompressed, dest)
    shutil.rmtree(decompressed, ignore_errors=True)  # free up disk
    print(f"Dataset part {url} is loaded ✅")


@app.function(
    volumes={"/mnt/": volume},
    # Timeout for this Function is set at the maximum, 24 hours,
    # because downloading, decompressing and storing almost 2 TiB of
    # files takes a long time.
    timeout=60 * 60 * 24,
)
def import_transform_load() -> None:
    # NOTE:
    # The mmseq.com server upload speed is quite slow so this download takes a while.
    # The download speed is also quite variable, sometimes taking over 5 hours.
    list(
        _do_part.map(
            [
                "http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz",
                "https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz",
                "https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz",
            ]
        )
    )
    print("Dataset is loaded ✅")
