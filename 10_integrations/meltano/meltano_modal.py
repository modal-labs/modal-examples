import os
import shutil
import subprocess
from pathlib import Path

import modal

local_project_root = Path(__file__).parent / "meltano_project"

meltano_source_mount = modal.Mount(
    local_dir=local_project_root,
    remote_dir="/project",
    condition=lambda path: not any(p.startswith(".") for p in Path(path).parts),
)

storage = modal.SharedVolume().persist("meltano_volume")
db_path = Path("/persisted/meltano.db")
remote_project_root = "/meltano/project"

meltano_conf = modal.Secret(
    {
        "MELTANO_PROJECT_ROOT": remote_project_root,
        "MELTANO_DATABASE_URI": f"sqlite:///{db_path}",
    }
)
default_logs_dir = Path(f"{remote_project_root}/.meltano/logs")


def install_project_deps():
    os.environ["MELTANO_PROJECT_ROOT"] = "/meltano/project"
    subprocess.check_call(["meltano", "install"])
    # delete logs, so they can easily be symlinked from running containrs
    shutil.rmtree(default_logs_dir, ignore_errors=True)


meltano_img = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("meltano")
    .copy(meltano_source_mount, "/meltano")
    .run_function(install_project_deps)
)


stub = modal.Stub(image=meltano_img)


class MeltanoContainer:
    def __enter__(self):
        if not db_path.exists():
            # copy the clean default db if there is none
            db_path.write_bytes(Path("/meltano_project/.meltano/meltano.db").read_bytes())

        # symlink logs so they end up in persisted shared volume
        persisted_logs_dir = Path("/persisted/logs")
        persisted_logs_dir.mkdir(exist_ok=True)
        default_logs_dir.symlink_to(persisted_logs_dir)

    @stub.wsgi(
        image=meltano_img,
        shared_volumes={"/persisted": storage},
        secrets=[modal.Secret.from_name("meltano-secrets"), meltano_conf],
    )
    def meltano_ui(self):
        # This serves the deprecated meltano UI as a webhook. Note that triggering long running
        # tasks from the UI would not work well since it spawns the tasks in
        # a thread within the potentially short-lived container
        import meltano.api.app

        return meltano.api.app.create_app()

    @stub.function(
        image=meltano_img,
        shared_volumes={"/persisted": storage},
        secrets=[modal.Secret.from_name("meltano-secrets"), meltano_conf],
    )
    def daily_ingest(self):
        subprocess.call(["meltano", "run", "github-to-jsonl"])


@stub.function(schedule=modal.Period(days=1))
def scheduled_runs():
    MeltanoContainer().daily_ingest.call()


if __name__ == "__main__":
    with stub.run():
        MeltanoContainer().daily_ingest.call()

    # serve the deprecated meltano UI:
    # stub.serve()
