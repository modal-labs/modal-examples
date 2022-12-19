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
db_path = Path("/meltano_db_volume/meltano.db")

meltano_conf = modal.Secret(
    {
        "MELTANO_PROJECT_ROOT": "/meltano/project",
        "MELTANO_DATABASE_URI": f"sqlite:///{db_path}",
    }
)


def install_project_deps():
    os.environ["MELTANO_PROJECT_ROOT"] = "/meltano/project"
    subprocess.check_call(["meltano", "install"])


meltano_img = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install(["meltano"])
    .copy(meltano_source_mount, "/meltano")
    .run_function(install_project_deps)
)


stub = modal.Stub(image=meltano_img)


class MeltanoContainer:
    def __enter__(self):
        # init database if it doesn't exist
        if not db_path.exists():
            db_path.write_bytes(Path("/meltano_project/.meltano/meltano.db").read_bytes())

        # symlink logs so they end up in persisted shared volume
        from_path = Path("/meltano_project/.meltano/logs")
        to_path = Path("/meltano_db_volume/logs")
        if from_path.exists() and not from_path.is_symlink():
            shutil.rmtree(from_path)
        if not from_path.exists():
            to_path.mkdir(exist_ok=True)
            from_path.symlink_to(to_path)

    @stub.wsgi(
        image=meltano_img,
        shared_volumes={"/meltano_db_volume": storage},
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
        shared_volumes={"/meltano_db_volume": storage},
        secrets=[modal.Secret.from_name("meltano-secrets"), meltano_conf],
    )
    def daily_ingest(self):
        subprocess.call(["meltano", "run", "github-to-jsonl"])


@stub.function(schedule=modal.Period(days=1))
def scheduled_runs():
    MeltanoContainer().daily_ingest.call()


if __name__ == "__main__":
    stub.serve()
