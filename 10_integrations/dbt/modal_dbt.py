import os
import subprocess
import typing
from pathlib import Path

import modal

LOCAL_DBT_PROJECT = Path(__file__).parent / "sample_proj"
REMOTE_DBT_PROJECT = "/sample_proj"

image = modal.Image.debian_slim().pip_install("dbt-sqlite").run_commands("apt-get install -y git")

# raw data loaded by meltano, see the meltano example in 10_integrations/meltano
raw_volume = modal.SharedVolume.from_name("meltano_volume")

# output schemas
db_volume = modal.SharedVolume().persist("dbt_dbs")
project_mount = modal.Mount(local_dir=LOCAL_DBT_PROJECT, remote_dir=REMOTE_DBT_PROJECT)
stub = modal.Stub(image=image, mounts=[project_mount])


@stub.function(shared_volumes={"/raw": raw_volume, "/db": db_volume})
def dbt_cli(subcommand: typing.List):
    os.chdir(REMOTE_DBT_PROJECT)
    subprocess.check_call(["dbt"] + subcommand)


@stub.function
def run():
    dbt_cli.call(["run"])


@stub.function
def debug():
    dbt_cli.call(["debug"])
