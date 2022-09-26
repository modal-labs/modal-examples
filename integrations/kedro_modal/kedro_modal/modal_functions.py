import os
import shutil
import subprocess
import warnings
from pathlib import Path

from modal import Image, Mount, SharedVolume, Stub, create_package_mounts

package_mounts = create_package_mounts(["kedro_modal"])


def run_kedro(project_path: Path, data_path: Path):
    shutil.rmtree(project_path / "data")  # replace project mounted data dir
    (project_path / "data").symlink_to(data_path)
    os.chdir(project_path)
    subprocess.call(["kedro", "run"])


def sync_data(source: Path, destination: Path, reset: bool = False):
    """Sync a local data directory *to* a shared volume"""

    # TODO: only sync raw data - no intermediates etc?
    if destination.exists() and reset:
        shutil.rmtree(destination)
    if not destination.exists():
        shutil.copytree(source, destination)


def non_hidden_files(project_path: Path):
    def condition(path):
        rel = Path(path).relative_to(project_path)
        return not any(
            part != ".gitkeep" and part.startswith(".") for part in rel.parts
        )

    return condition


def main_stub(project_path, project_name, package_name) -> Stub:
    requirements_txt = project_path / "src" / "requirements.txt"

    image = Image.debian_slim()
    if requirements_txt.exists():
        image = image.pip_install_from_requirements(requirements_txt)
    else:
        warnings.warn(
            "No requirements.txt in kedro src dir - attaching no dependencies"
        )
        image = image.pip_install("kedro")

    remote_project_mount_point = Path(f"/kedro-project/{package_name}")
    kedro_proj_mount = Mount(
        remote_dir=remote_project_mount_point,
        local_dir=project_path,
        condition=non_hidden_files(project_path),
    )
    stub = Stub(
        f"kedro-run.{project_name}",
        image=image,
        mounts=[kedro_proj_mount] + package_mounts,
    )
    volume_name = f"kedro.{project_name}.storage"
    data_volume = SharedVolume().persist(volume_name)

    stub.function(shared_volumes={"/kedro-storage": data_volume})(run_kedro)
    stub.function(shared_volumes={"/kedro-storage": data_volume})(sync_data)
    remote_data_path = Path("/kedro-storage/data")
    return stub, remote_project_mount_point, remote_data_path


def sync_stub(project_path, project_name):
    # slimmer sync stub that only mounts the data dir in order to upload raw data
    stub = Stub(f"kedro-data-sync.{project_name}")
    volume_name = f"kedro.{project_name}.storage"
    data_volume = SharedVolume().persist(volume_name)

    remote_source_path = Path("/source-data")
    source_mount = Mount(
        remote_dir=remote_source_path,
        local_dir=project_path / "data",
        condition=non_hidden_files(project_path),
    )
    stub.function(
        mounts=[source_mount] + package_mounts,
        shared_volumes={"/kedro-storage": data_volume},
    )(sync_data)
    remote_destination_path = Path("/kedro-storage/data")
    return stub, remote_source_path, remote_destination_path
