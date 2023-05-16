from pathlib import Path

import modal

# this must match the bucket source in `sample_proj_duckdb_s3/sources.yml`
BUCKET_NAME = "example-dbt-duckdb-s3"
LOCAL_DBT_PROJECT = Path(__file__).parent / "sample_proj_duckdb_s3"
PROJ_PATH = "/root/dbt"
PROFILES_PATH = "/root/dbt_profile"
TARGET_PATH = "/root/target"
dbt_image = (
    modal.Image.debian_slim()
    .pip_install(
        "boto3",
        "dbt-duckdb>=1.5.1",
        "pandas",
        "pyarrow",
    )
    .env(
        {
            "DBT_PROJECT_DIR": PROJ_PATH,
            "DBT_PROFILES_DIR": PROFILES_PATH,
            "DBT_TARGET_PATH": TARGET_PATH,
        }
    )
)
stub = modal.Stub(name="example-dbt-duckdb-s3", image=dbt_image)
dbt_project = modal.Mount.from_local_dir(
    LOCAL_DBT_PROJECT, remote_path=PROJ_PATH
)
dbt_profiles = modal.Mount.from_local_file(
    local_path=LOCAL_DBT_PROJECT / "profiles.yml",
    remote_path=Path(PROFILES_PATH, "profiles.yml"),
)
dbt_target = modal.SharedVolume().persist("dbt-target")
# Create this secret using the "AWS" template at https://modal.com/secrets/create.
# Be sure that the AWS user you provide credentials for has permission to
# create S3 buckets and read/write data from them.
#
# Below we will use a Modal function to create an S3 bucket and populate it with
# .parquet data.
s3_secret = modal.Secret.from_name("personal-aws-user")

# ## Seed data
#
# In order to provide source data for DBT to ingest and transform,
# we have this `seed` function which creates an AWS S3 bucket and
# populates it with .parquet files based of CSV data in the seeds/ directory.
#
# This is not the typical way that seeds/ data is used, but it is fine for this
# demonstration example. See https://docs.getdbt.com/docs/build/seeds for more info.


@stub.function(
    mounts=[dbt_project],
    secrets=[s3_secret],
)
def seed():
    import boto3
    import pandas as pd

    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=BUCKET_NAME)

    for seed_csv_path in Path(PROJ_PATH, "seeds").glob("*.csv"):
        print(f"found seed file {seed_csv_path}")
        name = seed_csv_path.stem
        df = pd.read_csv(seed_csv_path)
        parquet_filename = f"{name}.parquet"
        df.to_parquet(parquet_filename)

        object_key = f"sources/{parquet_filename}"
        print(f"uploading {object_key=} to S3 bucket '{BUCKET_NAME}'")
        s3_client.upload_file(parquet_filename, BUCKET_NAME, object_key)


# This `daily_build` function runs on a schedule to keep the DuckDB data warehouse
# up-to-date. Currently, the source data for this warehouse is static, so the updates
# don't really update anything, just re-build. But this example could be extended
# to have sources which continually provide new data across time.


@stub.function(
    schedule=modal.Period(days=1),
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    shared_volumes={TARGET_PATH: dbt_target},
)
def daily_build() -> None:
    run("build")


# `modal run dbt_duckdb.py::run --command run`
#
# A successful run will log something a lot like the following:
#
# ```
# 03:41:04  Running with dbt=1.5.0
# 03:41:05  Found 5 models, 8 tests, 0 snapshots, 0 analyses, 313 macros, 0 operations, 3 seed files, 3 sources, 0 exposures, 0 metrics, 0 groups
# 03:41:05
# 03:41:06  Concurrency: 1 threads (target='modal')
# 03:41:06
# 03:41:06  1 of 5 START sql table model main.stg_customers ................................ [RUN]
# 03:41:06  1 of 5 OK created sql table model main.stg_customers ........................... [OK in 0.45s]
# 03:41:06  2 of 5 START sql table model main.stg_orders ................................... [RUN]
# 03:41:06  2 of 5 OK created sql table model main.stg_orders .............................. [OK in 0.34s]
# 03:41:06  3 of 5 START sql table model main.stg_payments ................................. [RUN]
# 03:41:07  3 of 5 OK created sql table model main.stg_payments ............................ [OK in 0.36s]
# 03:41:07  4 of 5 START sql external model main.customers ................................. [RUN]
# 03:41:07  4 of 5 OK created sql external model main.customers ............................ [OK in 0.72s]
# 03:41:07  5 of 5 START sql table model main.orders ....................................... [RUN]
# 03:41:08  5 of 5 OK created sql table model main.orders .................................. [OK in 0.22s]
# 03:41:08
# 03:41:08  Finished running 4 table models, 1 external model in 0 hours 0 minutes and 3.15 seconds (3.15s).
# 03:41:08  Completed successfully
# 03:41:08
# 03:41:08  Done. PASS=5 WARN=0 ERROR=0 SKIP=0 TOTAL=5
# ```
#


@stub.function(
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    shared_volumes={TARGET_PATH: dbt_target},
)
def run(command: str) -> None:
    from dbt.cli.main import dbtRunner

    res = dbtRunner().invoke([command])
    if res.exception:
        print(res.exception)


# Look for the "'materialized='external'" DBT config in the SQL templates
# to see how `dbt-duckdb` is able to write back the transformed data to AWS S3!
#
# After running the 'run' command and seeing it succeed, check what's contained
# under the bucket's `out/` key prefix. You'll see that DBT has run the transformations
# defined in `sample_proj_duckdb_s3/models/` and produced output .parquet files.
