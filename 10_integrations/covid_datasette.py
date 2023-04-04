# ---
# deploy: true
# ---
# # Publish interactive datasets with Datasette
#
# ![Datasette user interface](./covid_datasette_ui.png)
#
# This example shows how to serve a Datasette application on Modal. The published dataset
# is COVID-19 case data from Johns Hopkins University which is refreshed daily.
# Try it out for yourself at [modal-labs-example-covid-datasette-app.modal.run/covid-19](https://modal-labs-example-covid-datasette-app.modal.run/covid-19/johns_hopkins_csse_daily_reports).
#
# Some Modal features it uses:
# * Shared volumes: a persisted volume lets us store and grow the published dataset over time
# * Scheduled functions: the underlying dataset is refreshed daily, so we schedule a function to run daily
# * Webhooks: exposes the Datasette application for web browser interaction and API requests.
#
# ## Basic setup
#
# Let's get started writing code.
# For the Modal container image we need a few Python packages,
# including `GitPython`, which we'll use to download the dataset.

import asyncio
import pathlib
import shutil
import tempfile
from datetime import datetime, timedelta

import modal

stub = modal.Stub("example-covid-datasette")
datasette_image = (
    modal.Image.debian_slim()
    .pip_install(
        "datasette~=0.63.2",
        "flufl.lock",
        "GitPython",
        "sqlite-utils",
    )
    .apt_install("git")
)

# ## Persistent dataset storage
#
# To separate database creation and maintenance from serving, we'll need the underlying
# database file to be stored persistently. To achieve this we use a [`SharedVolume`](/docs/guide/shared-volumes),
# a writable volume that can be attached to Modal functions and persisted across function runs.

volume = modal.SharedVolume().persist("covid-dataset-cache-vol")

CACHE_DIR = "/cache"
LOCK_FILE = str(pathlib.Path(CACHE_DIR, "lock-reports"))
REPO_DIR = pathlib.Path(CACHE_DIR, "COVID-19")
DB_PATH = pathlib.Path(CACHE_DIR, "covid-19.db")

# ## Getting a dataset
#
# Johns Hopkins has been publishing up-to-date COVID-19 pandemic data on GitHub since early February 2020, and
# as of late September 2022 daily reporting is still rolling in. Their dataset is what this example will use to
# show off Modal and Datasette's capabilities.
#
# The full git repository size for the dataset is over 6GB, but we only need to shallow clone around 300MB.


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
    retries=2,
)
def download_dataset(cache=True):
    import git
    from flufl.lock import Lock

    if REPO_DIR.exists() and cache:
        print(f"Dataset already present and {cache=}. Skipping download.")
        return
    elif REPO_DIR.exists():
        print(
            "Acquiring lock before deleting dataset, which may be in use by other runs."
        )
        with Lock(LOCK_FILE, default_timeout=timedelta(hours=1)):
            shutil.rmtree(REPO_DIR)
        print("Cleaned dataset before re-downloading.")

    git_url = "https://github.com/CSSEGISandData/COVID-19"
    git.Repo.clone_from(git_url, REPO_DIR, depth=1)


# ## Data munging
#
# This dataset is no swamp, but a bit of data cleaning is still in order. The following two
# functions are used to read a handful of `.csv` files from the git repository and cleaning the
# rows data before inserting into SQLite. You can see that the daily reports are somewhat inconsistent
# in their column names.


def load_daily_reports():
    jhu_csse_base = REPO_DIR
    reports_path = (
        jhu_csse_base / "csse_covid_19_data" / "csse_covid_19_daily_reports"
    )
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
            country_or_region = row.get("Country_Region") or row.get(
                "Country/Region"
            )
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
                "last_update": row.get("Last Update")
                or row.get("Last_Update")
                or None,
            }


# ## Inserting into SQLite
#
# With the CSV processing out of the way, we're ready to create an SQLite DB and feed data into it.
# Importantly, the `prep_db` function mounts the same shared volume used by `download_dataset()`, and
# rows are batch inserted with progress logged after each batch, as the full COVID-19 has millions
# of rows and does take some time to be fully inserted.
#
# A more sophisticated implementation would only load new data instead of performing a full refresh,
# but for this example things are kept simple.


def chunks(it, size):
    import itertools

    return iter(lambda: tuple(itertools.islice(it, size)), ())


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
    timeout=900,
)
def prep_db():
    import sqlite_utils
    from flufl.lock import Lock

    print("Loading daily reports...")
    records = load_daily_reports()

    with Lock(
        LOCK_FILE,
        lifetime=timedelta(minutes=2),
        default_timeout=timedelta(hours=1),
    ) as lck, tempfile.NamedTemporaryFile() as tmp:
        db = sqlite_utils.Database(tmp.name)
        table = db["johns_hopkins_csse_daily_reports"]

        batch_size = 100_000
        for i, batch in enumerate(chunks(records, size=batch_size)):
            truncate = True if i == 0 else False
            table.insert_all(batch, batch_size=batch_size, truncate=truncate)
            lck.refresh()
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
# setup a [scheduled](/docs/guide/cron) Modal function to run automatically once every 24 hours.


@stub.function(schedule=modal.Period(hours=24), timeout=1000)
def refresh_db():
    print(f"Running scheduled refresh at {datetime.now()}")
    download_dataset.call(cache=False)
    prep_db.call()


# ## Webhook
#
# Hooking up the SQLite database to a Modal webhook is as simple as it gets.
# The Modal `@stub.asgi_app` decorator wraps a few lines of code: one `import` and a few
# lines to instantiate the `Datasette` instance and return a reference to its ASGI app object.


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
@stub.asgi_app()
def app():
    from datasette.app import Datasette

    ds = Datasette(files=[DB_PATH])
    asyncio.run(ds.invoke_startup())
    return ds.app()


# ## Publishing to the web
#
# Run this script using `modal run covid_datasette.py` and it will create the database.
#
# You can run this script using `modal serve covid_datasette.py` and it will create a
# short-lived web URL that exists until you terminate the script.
#
# When publishing the interactive Datasette app you'll want to create a persistent URL.
# This is achieved by deploying the script with `modal deploy covid_datasette.py`.


@stub.local_entrypoint
def run():
    print("Downloading COVID-19 dataset...")
    download_dataset.call()
    print("Prepping SQLite DB...")
    prep_db.call()


# You can go explore the data over at [modal-labs-covid-datasette-app.modal.run/covid-19/](https://modal-labs-example-covid-datasette-app.modal.run/covid-19/johns_hopkins_csse_daily_reports).
