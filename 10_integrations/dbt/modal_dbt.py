import os
import subprocess
import typing
from pathlib import Path

import modal

image = modal.Image.debian_slim().pip_install("dbt-postgres").run_commands("apt-get install -y git")

stub = modal.Stub(image=image, secrets=[modal.Secret.from_name("postgres-secret")])

local_project_path = Path(__file__).parent / "sample_proj"

project_mount = modal.Mount(local_dir=local_project_path, remote_dir="/sample_proj")


def _dbt_cli(subcommand: typing.List):
    os.chdir("/sample_proj")
    subprocess.check_call(["dbt"] + subcommand)


@stub.function(mounts=[project_mount])
def run():
    _dbt_cli(["run"])


@stub.function(mounts=[project_mount])
def test():
    _dbt_cli(["test"])
