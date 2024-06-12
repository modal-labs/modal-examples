# ---
# cmd: ["modal", "run", "10_integrations/dbt/dbt_duckdb.py::run", "--command", "run"]
# ---
#
# This example contains a minimal but capable cloud data warehouse.
# It's comprised of the following:
#
# - [DuckDB](https://duckdb.org) as the warehouse's OLAP database engine
# - AWS S3 as the data storage provider
# - [DBT](https://docs.getdbt.com/docs/introduction) as the data transformation tool
#
# Meet your new cloud data warehouse.

from pathlib import Path

import modal

# ## Bucket name configuration
#
# The only thing in the source code that you must update is the S3 bucket name.
# AWS S3 bucket names are globally unique, and the one in this source is used by Modal.
#
# Update the `BUCKET_NAME` variable below and also any references to the original value
# within `sample_proj_duckdb_s3/models/`. The AWS IAM policy below also includes the bucket
# name and that must be updated.

BUCKET_NAME = "modal-example-dbt-duckdb-s3"
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
app = modal.App(name="example-dbt-duckdb-s3", image=dbt_image)

# ## DBT Configuration
#
# Most of the DBT code and configuration is taken directly from the
# https://github.com/dbt-labs/jaffle_shop demo and modified to support
# using dbt-duckdb with an S3 bucket.
#
# The DBT profiles.yml configuration is taken from
# https://github.com/jwills/dbt-duckdb#configuring-your-profile.
#
# Here we mount all this local code and configuration into the Modal function
# so that it will be available when we run DBT in the Modal cloud.

dbt_project = modal.Mount.from_local_dir(
    LOCAL_DBT_PROJECT, remote_path=PROJ_PATH
)
dbt_profiles = modal.Mount.from_local_file(
    local_path=LOCAL_DBT_PROJECT / "profiles.yml",
    remote_path=Path(PROFILES_PATH, "profiles.yml"),
)
dbt_target = modal.NetworkFileSystem.from_name(
    "dbt-target", create_if_missing=True
)
# Create this secret using the "AWS" template at https://modal.com/secrets/create.
# Be sure that the AWS user you provide credentials for has permission to
# create S3 buckets and read/write data from them.
#
# The policy required for this example is the following.
# Not that you *must* update the bucket name listed in the policy to your
# own bucket name.
#
# ```json
# {
#     "Statement": [
#         {
#             "Action": "s3:*",
#             "Effect": "Allow",
#             "Resource": [
#                 "arn:aws:s3:::modal-example-dbt-duckdb-s3/*",
#                 "arn:aws:s3:::modal-example-dbt-duckdb-s3"
#             ],
#             "Sid": "duckdbs3access"
#         }
#     ],
#     "Version": "2012-10-17"
# }
# ```
#
# Below we will use this user in a Modal function to create an S3 bucket and
# populate it with .parquet data.
s3_secret = modal.Secret.from_name("modal-examples-aws-user")

# ## Seed data
#
# In order to provide source data for DBT to ingest and transform,
# we have this `create_source_data` function which creates an AWS S3 bucket and
# populates it with .parquet files based of CSV data in the seeds/ directory.
#
# This is not the typical way that seeds/ data is used, but it is fine for this
# demonstration example. See https://docs.getdbt.com/docs/build/seeds for more info.


@app.function(
    mounts=[dbt_project],
    secrets=[s3_secret],
)
def create_source_data():
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


@app.function(
    schedule=modal.Period(days=1),
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    network_file_systems={TARGET_PATH: dbt_target},
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


@app.function(
    secrets=[s3_secret],
    mounts=[dbt_project, dbt_profiles],
    network_file_systems={TARGET_PATH: dbt_target},
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
