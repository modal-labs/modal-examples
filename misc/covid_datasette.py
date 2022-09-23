# ---
# integration-test: false
# ---
# # Publish interactive datasets with Datasette
#
# ![Datasette user interface](./covid_datasette_ui.png)
#
# This example shows how to serve a Datasette application on Modal. The published dataset
# is COVID-19 case data from Johns Hopkins University which is refreshed daily.
#
# Some Modal features it uses:
# * Shared volumes: a persisted volume lets us store and grow the published dataset over time
# * Scheduled functions: the underlying dataset is refreshed daily, so we schedule a function to run daily
# * Webhooks: exposes the Datasette application for web browser interaction and API requests.
#
# ## Basic setup
#
# Let's get started writing code.
# For the Modal container image, we need a few Python packages,
# including `GitPython`, which we'll use to download the dataset.

import pathlib
import sys

import modal

stub = modal.Stub("covid-datasette")
datasette_image = (
    modal.DebianSlim()
    .pip_install(
        [
            "datasette",
            "sqlite-utils",
            "GitPython",
        ]
    )
    .apt_install(["git"])
)

# ## Persistent dataset storage
#
# To separate database creation and maintenance from serving, we'll need the underlying
# database file to be stored persistently. To acheive this we use a [`SharedVolume`](/docs/guide/shared-volumes),
# a writable volume that can be attached to Modal functions and persisted across function runs.

volume = modal.SharedVolume().persist("covid-dataset-cache-vol")

CACHE_DIR = "/cache"
REPO_DIR = pathlib.Path(CACHE_DIR, "COVID-19")
DB_PATH = pathlib.Path(CACHE_DIR, "covid-19.db")

# ## Getting a dataset
#
# Johns Hopkins has been publishing up-to-date COVID-19 pandemic data on Github since early February 2020, and
# as of late September 2022 daily reporting is still rolling in. Their dataset is what this example will use to
# show off Modal and Datasette's capabilities.
#
# The full git repository size for the dataset is over 6GB, but we only need to shallow clone around 300MB.


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def download_dataset(cache=True):
    import git
    import shutil

    if REPO_DIR.exists():
        if cache:
            print("Dataset already present. Skipping download.")
            return
        shutil.rmtree(REPO_DIR)

    git_url = "https://github.com/CSSEGISandData/COVID-19"
    git.Repo.clone_from(git_url, REPO_DIR, depth=1)


# ## Data munging
#
# This dataset is no swamp, but a bit of data cleaning is still in order. The following two
# funtions are used to read a handful of .csv files from the git repository and cleaning the
# rows data before inserting into SQLite. You can see that the daily reports are somewhat inconsistent
# in their column names.


def load_daily_reports():
    jhu_csse_base = REPO_DIR
    reports_path = jhu_csse_base / "csse_covid_19_data" / "csse_covid_19_daily_reports"
    daily_reports = list(reports_path.glob("*.csv"))
    for filepath in daily_reports:
        yield from load_report(filepath)


def load_report(filepath):
    import csv

    mm, dd, yyyy = filepath.stem.split("-")
    with filepath.open() as fp:
        for row in csv.DictReader(fp):
            province_or_state = (
                row.get("\ufeffProvince/State")
                or row.get("Province/State")
                or row.get("Province_State")
                or None
            )
            country_or_region = row.get("Country_Region") or row.get("Country/Region")
            yield {
                "day": f"{yyyy}-{mm}-{dd}",
                "country_or_region": country_or_region.strip()
                if country_or_region
                else None,
                "province_or_state": province_or_state.strip()
                if province_or_state
                else None,
                "confirmed": int(float(row["Confirmed"] or 0)),
                "deaths": int(float(row["Deaths"] or 0)),
                "recovered": int(float(row["Recovered"] or 0)),
                "active": int(row["Active"]) if row.get("Active") else None,
                "last_update": row.get("Last Update") or row.get("Last_Update") or None,
            }


# ## Inserting into SQLite
#
# With the CSV processing out of the way, we're ready to create an SQLite DB and feed data into it.
# Importantly, the `prep_db` function mounts the same shared volume used by `download_dataset()`, and
# row inserts are batched to view progress in logs, as the full COVID-19 has millions of rows and does
# take some time to be fully inserted.
#
# A more sophisticated implementation would only load new data instead of performing a full refresh,
# but for this example things are kept simple.


def chunks(it, size, *, max_chunks=None):
    import itertools

    for i, chunk in enumerate(iter(lambda: tuple(itertools.islice(it, size)), ())):
        if max_chunks and i == max_chunks:
            return
        yield chunk


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def prep_db(max_records=None):
    import shutil
    import tempfile
    import sqlite_utils

    print("Loading daily reports...")
    records = load_daily_reports()

    with tempfile.NamedTemporaryFile() as tmp:
        db = sqlite_utils.Database(tmp.name)
        table = db["johns_hopkins_csse_daily_reports"]

        batch_size = 100_000
        for i, batch in enumerate(
            chunks(
                records, size=batch_size, max_chunks=min(max_records // batch_size, 1)
            )
        ):
            truncate = True if i == 0 else False
            table.insert_all(batch, batch_size=batch_size, truncate=truncate)
            print(f"Inserted {len(batch)} rows into DB.")

        table.create_index(["day"], if_not_exists=True)
        table.create_index(["province_or_state"], if_not_exists=True)
        table.create_index(["country_or_region"], if_not_exists=True)

        print("Syncing DB with shared volume.")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp.name, DB_PATH)


# ## Keeping fresh
#
# Johns Hopkins commits new data to the dataset repository every day, so we
# setup a scheduled Modal function running once every 24 hours.


@stub.function(schedule=modal.Period(hours=24))
def refresh_db():
    from datetime import datetime

    print(f"Running scheduled refresh at {datetime.now()}")
    download_dataset(cache=False)
    prep_db()


# ## Webhook
#
# Hooking up the SQLite database to a Modal webhook is as simple as it gets.
# The `@stub.asgi` decorator wraps two lines of code. One `import`` and a single
# line to instantiate the `Datasette` instance and return a reference to its ASGI app object.


@stub.asgi(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def app():
    from datasette.app import Datasette

    return Datasette(files=[DB_PATH]).app()


# ## Publishing to the web
#
# You can run this script with the 'serve' command and it will create a short-lived
# web url that exists until you terminate the script.
#
# When publishing the interactive Datasette app you'll want to create a persistent URL.
# This is acheived by deploying the script with `modal app deploy covid_datasette.py`.

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "serve":
        stub.serve()
    elif cmd == "prep":
        with stub.run():
            print("Downloading COVID-19 dataset...")
            download_dataset()
            print("Prepping SQLite DB...")
            max_records = 10_000  # Use small dataset when developing app.
            prep_db(max_records)
    else:
        exit("Unknown command. Support commands [serve, prep]")
