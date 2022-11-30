import shutil
import subprocess

import modal
import os
from pathlib import Path


def meltano_install():
    shutil.copytree("/meltano_source", "/meltano_project")
    os.environ["MELTANO_PROJECT_ROOT"] = "/meltano_project"
    subprocess.call(["meltano", "install"])

# TODO: easier way to add build context from local directory, which also invalidates on file changes
meltano_source_mount = modal.Mount(
    local_dir=Path(__file__).parent / "meltano_project",
    remote_dir="/meltano_source",
    condition=lambda path: not any(p.startswith(".") for p in Path(path).parts)
)

meltano_img = modal.Image.debian_slim() \
    .apt_install(["git"]) \
    .pip_install(["meltano"]) \
    .run_function(meltano_install, mounts=[meltano_source_mount])


stub = modal.Stub(image=meltano_img)


db_volume = modal.SharedVolume().persist("meltano_db")
db_path = Path("/meltano_db_volume/meltano.db")

@stub.wsgi(image=meltano_img, shared_volumes={"/meltano_db_volume": db_volume}, secrets=[modal.Secret.from_name("meltano-secrets")])
def meltano_ui():
    # init database if it doesn't exist
    if not db_path.exists():
        db_path.write_bytes(Path("/meltano_project/.meltano/meltano.db").read_bytes())
    # symlink logs so they end up in persisted shared volume
    log_output_path = Path("/meltano_project/.meltano/logs")
    if log_output_path.exists() and not log_output_path.is_symlink():
        log_output_path.unlink()
    if not log_output_path.exists():
        log_storage = Path("/meltano_db_volume/logs")
        log_storage.mkdir(exist_ok=True)
        log_output_path.symlink_to(log_storage)

    os.environ["MELTANO_PROJECT_ROOT"] = "/meltano_project"
    os.environ["MELTANO_PROJECT_READONLY"] = "true"
    os.environ["MELTANO_DATABASE_URI"] = f"sqlite:///{db_path}"
    import meltano.api.app
    return meltano.api.app.create_app()


@stub.function(image=meltano_img, schedule=modal.Period(days=1), shared_volumes={"/meltano_db_volume": db_volume}, secrets=[modal.Secret.from_name("meltano-secrets")])
def daily_ingest():
    os.environ["MELTANO_PROJECT_ROOT"] = "/meltano_project"
    os.environ["MELTANO_PROJECT_READONLY"] = "true"
    os.environ["MELTANO_DATABASE_URI"] = f"sqlite:///{db_path}"
    subprocess.call(["meltano", "run", "github-to-jsonl"])


if __name__ == "__main__":
    stub.serve()
