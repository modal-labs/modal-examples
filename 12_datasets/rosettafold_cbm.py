import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import modal
from multiprocessing import Process
from multiprocessing.pool import ThreadPool


bucket_creds = modal.Secret.from_name(
    "aws-s3-modal-examples-datasets", environment_name="main"
)
bucket_name = "modal-examples-datasets"
volume = modal.CloudBucketMount(
    bucket_name,
    secret=bucket_creds,
)
image = modal.Image.debian_slim().apt_install("wget")
app = modal.App("example-rosettafold-dataset-import", image=image)


def start_monitoring_disk_space(interval: int = 30) -> None:
    """Start monitoring the disk space in a separate thread."""
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


def decompress_tar_gz(
    file_path: pathlib.Path, extract_dir: pathlib.Path
) -> None:
    print(f"Decompressing {file_path} into {extract_dir}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        print(f"Decompressed {file_path} to {extract_dir}")


def copy_concurrent(src: pathlib.Path, dest: pathlib.Path) -> None:
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
        shutil.copytree(
            src, dest, copy_function=copier.copy, dirs_exist_ok=True
        )


@app.function(
    volumes={"/mnt/": volume},
    timeout=60 * 60 * 9,  # 9 hours,
    ephemeral_disk=1000 * 1024,
)
def import_transform_load() -> None:
    start_monitoring_disk_space()
    uniref30 = pathlib.Path("/tmp/UniRef30_2020_06_hhsuite.tar.gz")
    bfd_dataset = pathlib.Path(
        "/tmp/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
    )
    structure_templates = pathlib.Path("/tmp/pdb100_2021Mar03.tar.gz")
    commands = []
    if not uniref30.exists():
        print("Downloading uniref30 [46G]")
        commands.append(
            f"wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz -O {uniref30}"
        )

    if not bfd_dataset.exists():
        print("Downloading BFD [272G]")
        # NOTE: the mmseq.com server upload speed is quite slow so this download takes a while.
        commands.append(
            f"wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -O {bfd_dataset}"
        )

    if not structure_templates.exists():
        print(
            "Downloading structure templates (including *_a3m.ffdata, *_a3m.ffindex)"
        )
        commands.append(
            f"wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz -O {structure_templates}"
        )

    # Start all downloads in parallel
    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

    # Wait for all downloads to complete
    errors = []
    for p in processes:
        returncode = p.wait()
        if returncode == 0:
            print("Download completed successfully.")
        else:
            errors.append(
                f"Error in downloading. {p.args} failed {returncode=}"
            )
    if errors:
        raise RuntimeError(errors)

    uniref30_decompressed = pathlib.Path(
        "/tmp/rosettafold/UniRef30_2020_06_hhsuite"
    )
    bfd_dataset_decompressed = pathlib.Path(
        "/tmp/rosettafold/bfd_metaclust_clu_complete_id30_c90_final_seq"
    )
    structure_templates_decompressed = pathlib.Path(
        "/tmp/rosettafold/pdb100_2021Mar03/"
    )
    decompression_jobs = {
        (uniref30, uniref30_decompressed),
        (bfd_dataset, bfd_dataset_decompressed),
        (structure_templates, structure_templates_decompressed),
    }
    processes = []
    # Create and start a process for each decompression task
    for file_path, extract_dir in decompression_jobs:
        decompress_tar_gz(file_path, extract_dir)
        file_path.unlink()

    print("All decompression tasks completed.")

    dest = pathlib.Path("/mnt/rosettafold/")
    copy_concurrent(uniref30_decompressed, dest)
    shutil.rmtree(uniref30_decompressed, ignore_errors=True)
    copy_concurrent(bfd_dataset_decompressed, dest)
    shutil.rmtree(bfd_dataset_decompressed, ignore_errors=True)
    copy_concurrent(structure_templates_decompressed, dest)
    shutil.rmtree(structure_templates_decompressed, ignore_errors=True)
    print("Dataset is loaded ✅")
