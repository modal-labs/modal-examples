# ---
# deploy: true
# ---

# # Publish interactive datasets with Datasette

# ![Datasette user interface](https://modal-cdn.com/cdnbot/imdb_datasetteqzaj3q9d_a83d82fd.webp)

# Build and deploy an interactive movie database that automatically updates daily with the latest IMDb data.
# This example shows how to serve a Datasette application on Modal with millions of movie and TV show records.

# Try it out for yourself [here](https://modal-labs-examples--example-cron-datasette-ui.modal.run).

# Along the way, we will learn how to use the following Modal features:

# * [Volumes](https://modal.com/docs/guide/volumes): a persisted volume lets us store and grow the published dataset over time.

# * [Scheduled functions](https://modal.com/docs/guide/cron): the underlying dataset is refreshed daily, so we schedule a function to run daily.

# * [Web endpoints](https://modal.com/docs/guide/webhooks): exposes the Datasette application for web browser interaction and API requests.

# ## Basic setup

# Let's get started writing code.
# For the Modal container image we need a few Python packages.

import asyncio
import gzip
import pathlib
import shutil
import tempfile
from datetime import datetime
from urllib.request import urlretrieve

import modal

app = modal.App("example-cron-datasette")
cron_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasette==0.65.1", "sqlite-utils==3.38", "tqdm~=4.67.1", "setuptools<80"
)

# ## Persistent dataset storage

# To separate database creation and maintenance from serving, we'll need the underlying
# database file to be stored persistently. To achieve this we use a
# [Volume](https://modal.com/docs/guide/volumes).

volume = modal.Volume.from_name(
    "example-cron-datasette-cache-vol", create_if_missing=True
)
DB_FILENAME = "imdb.db"
VOLUME_DIR = "/cache-vol"
DATA_DIR = pathlib.Path(VOLUME_DIR, "imdb-data")
DB_PATH = pathlib.Path(VOLUME_DIR, DB_FILENAME)

# ## Getting a dataset

# [IMDb Datasets](https://datasets.imdbws.com/) are available publicly and are updated daily.
# We will download the title.basics.tsv.gz file which contains basic information about all titles (movies, TV shows, etc.).
# Since we are serving an interactive database which updates daily, we will download the files into a temporary directory and then move them to the volume to prevent downtime.

BASE_URL = "https://datasets.imdbws.com/"
IMDB_FILES = [
    "title.basics.tsv.gz",
]


@app.function(
    image=cron_image,
    volumes={VOLUME_DIR: volume},
    retries=2,
    timeout=1800,
)
def download_dataset(force_refresh=False):
    """Download IMDb dataset files."""
    if DATA_DIR.exists() and not force_refresh:
        print(
            f"Dataset already present and force_refresh={force_refresh}. Skipping download."
        )
        return

    TEMP_DATA_DIR = pathlib.Path(VOLUME_DIR, "imdb-data-temp")
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)

    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading IMDb dataset...")

    try:
        for filename in IMDB_FILES:
            print(f"Downloading {filename}...")
            url = BASE_URL + filename
            output_path = TEMP_DATA_DIR / filename

            urlretrieve(url, output_path)
            print(f"Successfully downloaded {filename}")

        if DATA_DIR.exists():
            # move the current data to a backup location
            OLD_DATA_DIR = pathlib.Path(VOLUME_DIR, "imdb-data-old")
            if OLD_DATA_DIR.exists():
                shutil.rmtree(OLD_DATA_DIR)
            shutil.move(DATA_DIR, OLD_DATA_DIR)

            # move the new data into place
            shutil.move(TEMP_DATA_DIR, DATA_DIR)

            # clean up the old data
            shutil.rmtree(OLD_DATA_DIR)
        else:
            shutil.move(TEMP_DATA_DIR, DATA_DIR)

        volume.commit()
        print("Finished downloading dataset.")

    except Exception as e:
        print(f"Error during download: {e}")
        if TEMP_DATA_DIR.exists():
            shutil.rmtree(TEMP_DATA_DIR)
        raise


# ## Data processing

# This dataset is no swamp, but a bit of data cleaning is still in order.
# The following function reads a .tsv file, cleans the data and yields batches of records.


def parse_tsv_file(filepath, batch_size=50000, filter_year=None):
    """Parse a gzipped TSV file and yield batches of records."""
    import csv

    with gzip.open(filepath, "rt", encoding="utf-8") as gz_file:
        reader = csv.DictReader(gz_file, delimiter="\t")
        batch = []
        total_processed = 0

        for row in reader:
            # map missing values to None
            row = {k: (None if v == "\\N" else v) for k, v in row.items()}

            # remove nsfw data
            if row.get("isAdult") == "1":
                continue

            if filter_year:
                start_year = int(row.get("startYear", 0) or 0)
                if start_year < filter_year:
                    continue

            batch.append(row)
            total_processed += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield any remaining records
        if batch:
            yield batch

        print(f"Finished processing {total_processed:,} titles.")


# ## Inserting into SQLite

# With the TSV processing out of the way, we’re ready to create a SQLite database and feed data into it.

# Importantly, the `prep_db` function mounts the same volume used by `download_dataset`, and rows are batch inserted with progress logged after each batch,
# as the full IMDb dataset has millions of rows and does take some time to be fully inserted.

# A more sophisticated implementation would only load new data instead of performing a full refresh,
# but we’re keeping things simple for this example!
# We will also create indexes for the titles table to speed up queries.


@app.function(
    image=cron_image,
    volumes={VOLUME_DIR: volume},
    timeout=900,
)
def prep_db(filter_year=None):
    """Process IMDb data files and create SQLite database."""
    import sqlite_utils
    import tqdm

    volume.reload()

    # Create database in a temporary directory first
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        tmp_db_path = tmpdir_path / DB_FILENAME

        db = sqlite_utils.Database(tmp_db_path)

        # Process title.basics.tsv.gz
        titles_file = DATA_DIR / "title.basics.tsv.gz"

        if titles_file.exists():
            titles_table = db["titles"]
            batch_count = 0
            total_processed = 0

            with tqdm.tqdm(desc="Processing titles", unit="batch", leave=True) as pbar:
                for i, batch in enumerate(
                    parse_tsv_file(
                        titles_file, batch_size=50000, filter_year=filter_year
                    )
                ):
                    titles_table.insert_all(batch, batch_size=50000, truncate=(i == 0))
                    batch_count += len(batch)
                    total_processed += len(batch)
                    pbar.update(1)
                    pbar.set_postfix({"titles": f"{total_processed:,}"})

            print(f"Total titles in database: {batch_count:,}")

            # Create indexes for titles so we can query the database faster
            print("Creating indexes...")
            titles_table.create_index(["tconst"], if_not_exists=True, unique=True)
            titles_table.create_index(["primaryTitle"], if_not_exists=True)
            titles_table.create_index(["titleType"], if_not_exists=True)
            titles_table.create_index(["startYear"], if_not_exists=True)
            titles_table.create_index(["genres"], if_not_exists=True)
            print("Created indexes for titles table")

        db.close()

        # Copy the database to the volume
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_db_path, DB_PATH)

    print("Syncing DB with volume.")
    volume.commit()
    print("Volume changes committed.")


# ## Keep it fresh

# IMDb updates their data daily, so we set up
# a [scheduled](https://modal.com/docs/guide/cron) function to automatically refresh the database
# every 24 hours.


@app.function(schedule=modal.Period(hours=24), timeout=4000)
def refresh_db():
    """Scheduled function to refresh the database daily."""
    print(f"Running scheduled refresh at {datetime.now()}")
    download_dataset.remote(force_refresh=True)
    prep_db.remote()


# ## Web endpoint

# Hooking up the SQLite database to a Modal webhook is as simple as it gets.
# The Modal `@asgi_app` decorator wraps a few lines of code: one `import` and a few
# lines to instantiate the `Datasette` instance and return its app server.

# First, let's define a metadata object for the database.
# This will be used to configure Datasette to display a custom UI with some pre-defined queries.

columns = {
    "tconst": "Unique identifier",
    "titleType": "Type (movie, tvSeries, short, etc.)",
    "primaryTitle": "Main title",
    "originalTitle": "Original language title",
    "startYear": "Release year",
    "endYear": "End year (for TV series)",
    "runtimeMinutes": "Runtime in minutes",
    "genres": "Comma-separated genres",
}

queries = {
    "movies_2024": {
        "sql": """
                        SELECT
                            primaryTitle as title,
                            genres,
                            runtimeMinutes as runtime
                        FROM titles
                        WHERE titleType = 'movie'
                        AND startYear = 2024
                        ORDER BY primaryTitle
                        LIMIT 100
                    """,
        "title": "Movies Released in 2024",
    },
    "longest_movies": {
        "sql": """
                        SELECT
                            primaryTitle as title,
                            startYear as year,
                            runtimeMinutes as runtime,
                            genres
                        FROM titles
                        WHERE titleType = 'movie'
                        AND runtimeMinutes IS NOT NULL
                        AND runtimeMinutes > 180
                        ORDER BY runtimeMinutes DESC
                        LIMIT 50
                    """,
        "title": "Longest Movies (3+ hours)",
    },
    "genre_breakdown": {
        "sql": """
                        SELECT
                            genres,
                            COUNT(*) as count
                        FROM titles
                        WHERE titleType = 'movie'
                        AND genres IS NOT NULL
                        GROUP BY genres
                        ORDER BY count DESC
                        LIMIT 25
                    """,
        "title": "Popular Genres",
    },
}


metadata = {
    "title": "IMDb Database Explorer",
    "description": "Explore IMDb movie and TV show data",
    "databases": {
        "imdb": {
            "tables": {
                "titles": {
                    "description": "Basic information about all titles (movies, TV shows, etc.)",
                    "columns": columns,
                }
            },
            "queries": {
                "movies_2024": queries["movies_2024"],
                "longest_movies": queries["longest_movies"],
                "genre_breakdown": queries["genre_breakdown"],
            },
        }
    },
}

# Now we can define the web endpoint that will serve the Datasette application


@app.function(
    image=cron_image,
    volumes={VOLUME_DIR: volume},
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def ui():
    """Web endpoint for Datasette UI."""
    from datasette.app import Datasette

    ds = Datasette(
        files=[DB_PATH],
        settings={
            "sql_time_limit_ms": 60000,
            "max_returned_rows": 10000,
            "allow_download": True,
            "facet_time_limit_ms": 5000,
            "allow_facet": True,
        },
        metadata=metadata,
    )
    asyncio.run(ds.invoke_startup())
    return ds.app()


# ## Publishing to the web

# Run this script using `modal run cron_datasette.py` and it will create the database under 5 minutes!

# If you would like to force a refresh of the dataset, you can use:

# `modal run cron_datasette.py --force-refresh`

# If you would like to filter the data to be after a specific year, you can use:

# `modal run cron_datasette.py --filter-year year`

# You can then use `modal serve cron_datasette.py` to create a short-lived web URL
# that exists until you terminate the script.

# When publishing the interactive Datasette app you'll want to create a persistent URL.
# Just run `modal deploy cron_datasette.py` and your app will be deployed in seconds!


@app.local_entrypoint()
def run(force_refresh: bool = False, filter_year: int = None):
    if force_refresh:
        print("Force refreshing the dataset...")

    if filter_year:
        print(f"Filtering data to be after {filter_year}")

    print("Downloading IMDb dataset...")
    download_dataset.remote(force_refresh=force_refresh)
    print("Processing data and creating SQLite DB...")
    prep_db.remote(filter_year=filter_year)
    print("\nDatabase ready! You can now run:")
    print("  modal serve cron_datasette.py  # For development")
    print("  modal deploy cron_datasette.py  # For production deployment")


# You can explore the data at the [deployed web endpoint](https://modal-labs-examples--example-cron-datasette-ui.modal.run).
