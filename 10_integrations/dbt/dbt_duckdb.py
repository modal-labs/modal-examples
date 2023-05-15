import json
from pathlib import Path

import modal


def create_inline_dbt_profile_mount(remote_path: Path) -> modal.Mount:
    """
    Writes an inline DBT profile to a temporary file that's
    return as a modal.Mount.
    """
    import tempfile
    dbt_profile = {
        "sample_proj": {
            "target": "modal",
            "outputs": {
                "modal": {
                    "type": "duckdb",
                    "path": "/tmp/dbt.duckdb",
                    "extensions": ["httpfs", "parquet"],
                    "settings": {
                        "s3_region": "us-east-1",
                        "s3_access_key_id": "{{ env_var('AWS_ACCESS_KEY_ID') }}",
                        "s3_secret_access_key": "{{ env_var('AWS_SECRET_ACCESS_KEY') }}",
                    }
                }
            }
        }
    }
    temp_profile_file = tempfile.NamedTemporaryFile(delete=False)
    temp_profile_file.write(json.dumps(dbt_profile).encode())
    return modal.Mount.from_local_file(local_path=temp_profile_file.name, remote_path=Path(remote_path, "profiles.yml"))



LOCAL_DBT_PROJECT = Path(__file__).parent / "sample_proj_duckdb_s3"
PROJ_PATH = "/root/dbt"
PROFILES_PATH = "/root/dbt_profile"
TARGET_PATH = "/root/target"
dbt_image = (
    modal.Image.debian_slim()
    .pip_install("boto", "dbt-duckdb>=1.5.1")
    .env(
        {
            "DBT_PROJECT_DIR": PROJ_PATH,
            "DBT_PROFILES_DIR": PROFILES_PATH,
            "DBT_TARGET_PATH": TARGET_PATH,
        }
    )
)
stub = modal.Stub(name="example-dbt-duckdb-s3", image=dbt_image)
dbt_project = modal.Mount.from_local_dir(LOCAL_DBT_PROJECT, remote_path=PROJ_PATH)
dbt_target = modal.SharedVolume().persist("dbt-target")
s3_secret = modal.Secret.from_name("aws-secret")


@stub.function(
    schedule=modal.Period(days=1),
    secrets=[s3_secret],
    mounts=[dbt_project, create_inline_dbt_profile_mount(remote_path=PROFILES_PATH)],
    shared_volumes={TARGET_PATH: dbt_target},
)
def run_dbt() -> None:
    from dbt.cli.main import dbtRunner

    res = dbtRunner().invoke(["build"])
    if res.exception:
        print(res.exception)
