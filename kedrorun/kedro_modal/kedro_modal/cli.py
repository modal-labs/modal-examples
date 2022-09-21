import os
import shutil
import subprocess
import sys
import warnings
import click
from pathlib import Path
from modal import Image, Mount, SharedVolume, Stub


@click.group(name="Kedro-Modal")
def commands():
    """Kedro plugin for running kedro pipelines on Modal"""
    pass




@commands.group(
    name="modal", context_settings=dict(help_option_names=["-h", "--help"])
)
def modal_group():
    """Interact with Kedro pipelines run on Modal"""

def run_kedro(project_path):
    os.chdir(project_path)
    subprocess.call(["kedro", "run"])


def sync_data(source: Path, destination: Path, reset: bool=False):
    """Sync a local data directory *to* a shared volume"""
    if not destination.exists() or reset:
        shutil.copytree(source, destination)


@modal_group.command
@click.pass_obj
def run(metadata):
    source_path = metadata.source_dir
    requirements_txt = (source_path / "requirements.txt")

    image = Image.debian_slim()
    if requirements_txt.exists():
        image = image.pip_install_from_requirements(requirements_txt)
    else:
        warnings.warn("No requirements.txt in kedro src dir - attaching no dependencies")
        image = image.pip_install("kedro")

    stub = Stub(name=f"kedro::{metadata.project_name}", image=image)
    stub.kedro_volume = SharedVolume()

    remote_project_mount_point = Path(f"/kedro-project/{metadata.package_name}")
    kedro_proj_mount = Mount(remote_dir=remote_project_mount_point, local_dir=metadata.project_path)
    modal_run_kedro = stub.function(mounts=[kedro_proj_mount])(run_kedro)
    modal_sync_data = stub.function(
        mounts=[kedro_proj_mount],
        shared_volumes={"/kedro-data": stub.kedro_volume}
    )(sync_data)

    with stub.run():
        modal_sync_data(remote_project_mount_point / "data", Path("/kedro-storage/data"), overwrite=False)
        modal_run_kedro(remote_project_mount_point)
