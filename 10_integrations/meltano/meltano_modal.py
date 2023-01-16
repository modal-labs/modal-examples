import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL_PROJECT_ROOT = Path(__file__).parent / "meltano_project"
REMOTE_PROJECT_ROOT = "/meltano_project"
PERSISTED_VOLUME_PATH = "/persisted"
REMOTE_DB_PATH = Path(f"{PERSISTED_VOLUME_PATH}/meltano.db")
REMOTE_LOGS_PATH = Path(f"{REMOTE_PROJECT_ROOT}/.meltano/logs")
PERSISTED_LOGS_DIR = Path(f"{PERSISTED_VOLUME_PATH}/logs")

meltano_source_mount = modal.Mount(
    local_dir=LOCAL_PROJECT_ROOT,
    remote_dir=REMOTE_PROJECT_ROOT,
    condition=lambda path: not any(p.startswith(".") for p in Path(path).parts),
)

storage = modal.SharedVolume().persist("meltano_volume")

meltano_conf = modal.Secret(
    {
        "MELTANO_PROJECT_ROOT": REMOTE_PROJECT_ROOT,
        "MELTANO_DATABASE_URI": f"sqlite:///{REMOTE_DB_PATH}",
        "SQLITE_WAREHOUSE": f"{PERSISTED_VOLUME_PATH}/warehouse",
        "MELTANO_ENVIRONMENT": "modal",
    }
)


def install_project_deps():
    os.environ["MELTANO_DATABASE_URI"] = "sqlite:////.empty_meltano.db"  # dummy during installation
    subprocess.check_call(["meltano", "install"])
    # delete empty logs dir, so running containers can add a symlink instead
    shutil.rmtree(REMOTE_LOGS_PATH, ignore_errors=True)


meltano_img = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("meltano")
    .copy(meltano_source_mount)
    .run_function(install_project_deps)
)


stub = modal.Stub(
    image=meltano_img,
    secrets=[modal.Secret.from_name("meltano-secrets"), meltano_conf],
)


class MeltanoContainer:
    def __enter__(self):
        # symlink logs so they end up in persisted shared volume
        PERSISTED_LOGS_DIR.mkdir(exist_ok=True, parents=True)
        REMOTE_LOGS_PATH.symlink_to(PERSISTED_LOGS_DIR)

    @stub.wsgi(shared_volumes={PERSISTED_VOLUME_PATH: storage})
    def meltano_ui(self):
        # This serves the deprecated meltano UI as a webhook
        import meltano.api.app

        return meltano.api.app.create_app()

    @stub.function(shared_volumes={PERSISTED_VOLUME_PATH: storage})
    def github_to_jsonl(self):
        subprocess.call(["meltano", "run", "github-to-jsonl"])

    @stub.function(shared_volumes={PERSISTED_VOLUME_PATH: storage})
    def elt(self):
        subprocess.call(["meltano", "run", "download_sample_data", "tap-csv", "target-sqlite"])


@stub.function(schedule=modal.Period(days=1))
def scheduled_runs():
    MeltanoContainer().elt.call()
    MeltanoContainer().github_to_jsonl.call()


@stub.local_entrypoint
def github_json_example():
    MeltanoContainer().github_to_jsonl.call()


# Run this example using `modal run meltano_modal.py`
@stub.local_entrypoint
def main():
    MeltanoContainer().elt.call()
