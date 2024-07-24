# ---
# cmd: ["modal", "run", "10_integrations/dbt_modal_inference/duck_db_modal_inference.py"]
# ---
# # dbt python model with llm inference
#
# In this example we demonstrate how you could combine dbt python models with llm
# inference models powered by modal, allowing you to run serverless gpu workloads within dbt.
#
# ## Overview
#
# This example runs [dbt](https://docs.getdbt.com/docs/introduction) with a [duckdb](https://duckdb.org)
# backend directly on top of modal, but could be translated to run on any dbt-compatible
# database that supports python models. Similarly you could make these requests from UDFs
# directly in SQL instead if you don't want to use python models.
#
# In this example we use an llm deployed in a previous example: [Serverless TensorRT-LLM (LLaMA 3 8B)](https://modal.com/docs/examples/trtllm_llama)
# but you could easily swap this for whichever modal function you wish. We use this to classify the sentiment
# for free-text product reviews and aggregate them in subsequent dbt sql models.
#
#
# ## dbt & image configuration
#
# We set up the environment variables necessary for dbt and
# create a slim debian and install the packages necessary to run.

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

# We mount the local code and configuration into the modal function
# so that it will be available when we run dbt
# and create a volume so that we can persist our data.

dbt_project = modal.Mount.from_local_dir(
    LOCAL_DBT_PROJECT, remote_path=PROJ_PATH
)
dbt_profiles = modal.Mount.from_local_file(
    local_path=LOCAL_DBT_PROJECT / "profiles.yml",
    remote_path=pathlib.Path(PROFILES_PATH, "profiles.yml"),
)
dbt_vol = modal.Volume.from_name("dbt-inference-vol", create_if_missing=True)

# ## Modal function

# Using the above configuration we can invoke dbt from modal
# and use this to run transformations in our warehouse
# The `dbt_run` function does a few things
# 1. It creates the directories for storing the duckdb database and dbt target files
# 2. It gets a reference to a deployed modal function that serves an llm inference endpoint
# 3. It runs dbt with a variable for the inference url
# 4. It prints the output of the final dbt table in the duckdb parquet output


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

    # Remember to either deploy this yourself in your environment
    # or change to another web endpoint you have
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


# Running the function:
# `modal run 10_integrations/dbt_modal_inference/duck_db_modal_inference.py`
# will result in something like:
#
# ```
# 21:25:21  Running with dbt=1.8.4
# 21:25:21  Registered adapter: duckdb=1.8.1
# 21:25:23  Found 5 models, 2 seeds, 6 data tests, 2 sources, 408 macros
# 21:25:23
# 21:25:23  Concurrency: 1 threads (target='dev')
# 21:25:23
# 21:25:23  1 of 5 START sql table model main.stg_products ................................. [RUN]
# 21:25:23  1 of 5 OK created sql table model main.stg_products ............................ [OK in 0.22s]
# 21:25:23  2 of 5 START sql table model main.stg_reviews .................................. [RUN]
# 21:25:23  2 of 5 OK created sql table model main.stg_reviews ............................. [OK in 0.17s]
# 21:25:23  3 of 5 START sql table model main.product_reviews .............................. [RUN]
# 21:25:23  3 of 5 OK created sql table model main.product_reviews ......................... [OK in 0.17s]
# 21:25:23  4 of 5 START python external model main.product_reviews_sentiment .............. [RUN]
# 21:25:32  4 of 5 OK created python external model main.product_reviews_sentiment ......... [OK in 8.83s]
# 21:25:32  5 of 5 START sql external model main.product_reviews_sentiment_agg ............. [RUN]
# 21:25:32  5 of 5 OK created sql external model main.product_reviews_sentiment_agg ........ [OK in 0.16s]
# 21:25:32
# 21:25:32  Finished running 3 table models, 2 external models in 0 hours 0 minutes and 9.76 seconds (9.76s).
# 21:25:33
# 21:25:33  Completed successfully
# 21:25:33
# 21:25:33  Done. PASS=5 WARN=0 ERROR=0 SKIP=0 TOTAL=5
# ┌──────────────┬──────────────────┬─────────────────┬──────────────────┐
# │ product_name │ positive_reviews │ neutral_reviews │ negative_reviews │
# │   varchar    │      int64       │      int64      │      int64       │
# ├──────────────┼──────────────────┼─────────────────┼──────────────────┤
# │ Splishy      │                3 │               0 │                1 │
# │ Blerp        │                3 │               1 │                1 │
# │ Zinga        │                2 │               0 │                0 │
# │ Jinkle       │                2 │               1 │                1 │
# │ Flish        │                2 │               2 │                1 │
# │ Kablooie     │                2 │               1 │                1 │
# │ Wizzle       │                2 │               1 │                0 │
# │ Snurfle      │                2 │               1 │                0 │
# │ Glint        │                2 │               0 │                0 │
# │ Flumplenook  │                2 │               1 │                1 │
# │ Whirlybird   │                2 │               0 │                1 │
# ├──────────────┴──────────────────┴─────────────────┴──────────────────┤
# │ 11 rows                                                    4 columns │
# └──────────────────────────────────────────────────────────────────────┘
# ```
#
# Here we can see that the llm classified the results into three different categories
# that we could then aggregate in a subsequent sql model!
#
# Since we're using a volume for storing our dbt target results
# and our duckdb parquet files
# you can view the results and use them outside the function too.
#
# View the target directory by:
# ```sh
# modal volume ls dbt-inference-vol target/
#            Directory listing of 'target/' in 'dbt-inference-vol'
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Filename                      ┃ Type ┃ Created/Modified      ┃ Size      ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
# │ target/run                    │ dir  │ 2024-07-19 22:59 CEST │ 14 B      │
# │ target/compiled               │ dir  │ 2024-07-19 22:59 CEST │ 14 B      │
# │ target/semantic_manifest.json │ file │ 2024-07-19 23:25 CEST │ 234 B     │
# │ target/run_results.json       │ file │ 2024-07-19 23:25 CEST │ 10.1 KiB  │
# │ target/manifest.json          │ file │ 2024-07-19 23:25 CEST │ 419.7 KiB │
# │ target/partial_parse.msgpack  │ file │ 2024-07-19 23:25 CEST │ 412.7 KiB │
# │ target/graph_summary.json     │ file │ 2024-07-19 23:25 CEST │ 1.4 KiB   │
# │ target/graph.gpickle          │ file │ 2024-07-19 23:25 CEST │ 15.7 KiB  │
# └───────────────────────────────┴──────┴───────────────────────┴───────────┘
# ```
#
# And the db directory:
# ```sh
# modal volume ls dbt-inference-vol db/
#                   Directory listing of 'db/' in 'dbt-inference-vol'
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
# ┃ Filename                                 ┃ Type ┃ Created/Modified      ┃ Size    ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
# │ db/review_sentiments.parquet             │ file │ 2024-07-19 23:25 CEST │ 9.6 KiB │
# │ db/product_reviews_sentiment_agg.parquet │ file │ 2024-07-19 23:25 CEST │ 756 B   │
# └──────────────────────────────────────────┴──────┴───────────────────────┴─────────┘
# ```
#
# ## Python dbt model
# The python dbt model in `dbt_modal_inference_proj/models/product_reviews_sentiment.py` is quite simple.
# It configures dbt to store the results locally in parquet, fetches the inference url,
# uses a batch reader to iterate over the inputs and call the `batcher` method
# and adds a new field from the LLM
#
# ```python
# import json
#
# import pyarrow as pa
# import requests
#
# def model(dbt, session):
#     dbt.config(
#         materialized="external",
#         location="/root/vol/db/review_sentiments.parquet",
#     )
#     inference_url = dbt.config.get("inference_url")
#
#     big_model = dbt.ref("product_reviews")
#     batch_reader = big_model.record_batch(100)
#     batch_iter = batcher(batch_reader, inference_url)
#     new_schema = batch_reader.schema.append(
#         pa.field("review_sentiment", pa.string())
#     )
#     return pa.RecordBatchReader.from_batches(new_schema, batch_iter)
#
# ```
#
# The `batcher` method makes the actual inference web request
# And passes in a prompt with a product review and adds the result to the
# given batch and returns it
#
# ```python
# def batcher(batch_reader: pa.RecordBatchReader, inference_url: str):
#     for batch in batch_reader:
#         df = batch.to_pandas()
#
#         prompts = (
#             df["product_review"]
#             .apply(lambda review: get_prompt(review))
#             .tolist()
#         )
#
#         res = requests.post(
#             inference_url,
#             json={"prompts": prompts},
#         )
#
#         df["review_sentiment"] = json.loads(res.content)
#
#         yield pa.RecordBatch.from_pandas(df)
# ```
# Finally, we have a method for generating the llm prompt, here we create a prompt with a few examples,
# something you could include in your deployed model configuration instead,
# but since we're using an example we put it here:
#
# ```python
# def get_prompt(review):
#     return (
#         """
# You are an expert at analyzing product reviews sentiment.
# Your task is to classify the given product review into one of the following labels: ["positive", "negative", "neutral"]
# Here are some examples:
# 1. "example": "Packed with innovative features and reliable performance, this product exceeds expectations, making it a worthwhile investment."
#    "label": "positive"
# 2. "example": "Despite promising features, the product's build quality and performance were disappointing, failing to meet expectations."
#    "label": "negative"
# 3. "example": "While the product offers some useful functionalities, its overall usability and durability may vary depending on individual needs and preferences."
#    "label": "neutral"
# Label the following review:
# """
#         + '"'
#         + review
#         + '"'
#         + """
# Respond in a single word with the label.
# """
#     )
# ```
#
# And it's that simple to call a modal web endpoint from dbt!
#
