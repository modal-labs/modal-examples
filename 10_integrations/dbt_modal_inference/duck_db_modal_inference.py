import pathlib

import modal

LOCAL_DBT_PROJECT = pathlib.Path(__file__).parent / "dbt_modal_inference_proj"
PROJ_PATH = "/root/dbt"
VOL_PATH = "/root/vol"
DB_PATH = f"{VOL_PATH}/db"
PROFILES_PATH = "/root/dbt_profile"
TARGET_PATH = f"{VOL_PATH}/target"
dbt_image = (
    modal.Image.debian_slim()
    .pip_install(
        "dbt-duckdb==1.8.1",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "requests==2.32.3",
    )
    .env(
        {
            "DBT_PROJECT_DIR": PROJ_PATH,
            "DBT_PROFILES_DIR": PROFILES_PATH,
            "DBT_TARGET_PATH": TARGET_PATH,
            "DB_PATH": DB_PATH,
        }
    )
)

app = modal.App("duckdb-dbt-inference", image=dbt_image)

dbt_project = modal.Mount.from_local_dir(
    LOCAL_DBT_PROJECT, remote_path=PROJ_PATH
)
dbt_profiles = modal.Mount.from_local_file(
    local_path=LOCAL_DBT_PROJECT / "profiles.yml",
    remote_path=pathlib.Path(PROFILES_PATH, "profiles.yml"),
)
dbt_vol = modal.Volume.from_name("dbt-inference-vol", create_if_missing=True)


@app.function(
    mounts=[dbt_project, dbt_profiles],
    volumes={VOL_PATH: dbt_vol},
)
def dbt_run() -> None:
    import os

    import duckdb
    from dbt.cli.main import dbtRunner

    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(TARGET_PATH, exist_ok=True)

    ref = modal.Function.lookup(
        "example-trtllm-Meta-Llama-3-8B-Instruct", "generate_web"
    )

    res = dbtRunner().invoke(
        ["run", "--vars", f"{{'inference_url': '{ref.web_url}'}}"]
    )
    if res.exception:
        print(res.exception)

    duckdb.sql(
        f"select * from '{DB_PATH}/product_reviews_sentiment_agg.parquet';"
    ).show()
