from contextlib import contextmanager
import os
import shutil
import subprocess
import sys
from typing import Iterator
import warnings
import click
from pathlib import Path
from modal import App, Image, Mount, SharedVolume, Stub, lookup


@click.group(name="Kedro-Modal")
def commands():
    """Kedro plugin for running kedro pipelines on Modal"""
    pass


@commands.group(
    name="modal", context_settings=dict(help_option_names=["-h", "--help"])
)
def modal_group():
    """Interact with Kedro pipelines run on Modal"""

def run_kedro(project_path: Path, data_path: Path):
    shutil.rmtree(project_path / "data")  # replace mount
    (project_path / "data").symlink_to(data_path)
    os.chdir(project_path)
    subprocess.call(["kedro", "run"])


def sync_data(source: Path, destination: Path, reset: bool=False):
    """Sync a local data directory *to* a shared volume"""

    # TODO: only sync raw data - no intermediates etc?

    if not destination.exists() or reset:
        if reset:
            shutil.rmtree(destination)
        shutil.copytree(source, destination)



def _modal_stub(metadata) -> Stub:
    source_path = metadata.source_dir
    requirements_txt = (source_path / "requirements.txt")

    image = Image.debian_slim()
    if requirements_txt.exists():
        image = image.pip_install_from_requirements(requirements_txt)
    else:
        warnings.warn("No requirements.txt in kedro src dir - attaching no dependencies")
        image = image.pip_install("kedro")

    stub = Stub(image=image)
    volume_name = f"kedro.{metadata.project_name}.storage"
    data_volume = SharedVolume().persist(volume_name)

    remote_project_mount_point = Path(f"/kedro-project/{metadata.package_name}")
    kedro_proj_mount = Mount(remote_dir=remote_project_mount_point, local_dir=metadata.project_path)
    stub.function(mounts=[kedro_proj_mount], shared_volumes={"/kedro-storage": data_volume})(run_kedro)
    stub.function(
        mounts=[kedro_proj_mount],
        shared_volumes={"/kedro-storage": data_volume}
    )(sync_data)
    remote_data_path = Path("/kedro-storage/data")
    return stub, remote_project_mount_point, remote_data_path


@modal_group.command
@click.pass_obj
def run(metadata):
    stub, remote_project_mount_point, remote_data_path = _modal_stub(metadata)
    with stub.run() as app:
        app.sync_data(remote_project_mount_point / "data", remote_data_path, reset=False)
        app.run_kedro(remote_project_mount_point, remote_data_path)


@modal_group.command
@click.pass_obj
def deploy(metadata):
    stub, remote_project_mount_point, remote_data_path = _modal_stub(metadata)
    name=f"kedro.{metadata.project_name}"
    stub.deploy(name)
    sync_data = lookup(name, "sync_data")  # use the deployed function
    sync_data(remote_project_mount_point / "data", remote_data_path)


@modal_group.command
@click.pass_obj
def reset(metadata):
    stub = Stub(name=f"kedro.{metadata.project_name}")
    volume_name = f"kedro.{metadata.project_name}.storage"
    data_volume = SharedVolume().persist(volume_name)

    remote_kedro_data_mount_path = Path(f"/kedro-project/{metadata.package_name}/data")
    kedro_data_mount = Mount(remote_dir=remote_kedro_data_mount_path, local_dir=metadata.project_path / "data")
    modal_sync_data = stub.function(
        mounts=[kedro_data_mount],
        shared_volumes={"/kedro-storage": data_volume}
    )(sync_data)

    remote_data_path = Path("/kedro-storage/data")

    with stub.run():
        modal_sync_data(remote_kedro_data_mount_path, remote_data_path, reset=True)
