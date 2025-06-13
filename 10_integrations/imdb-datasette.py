# ---
# deploy: true
# ---

# # Publish interactive datasets with Datasette

# ![Datasette user interface](./imdb_datasette_ui.png)

# This example shows how to serve a Datasette application on Modal. The published dataset
# is IMDB movie and TV show data which is refreshed daily.
# Try it out for yourself [here](https://modal-labs--example-imdb-datasette-ui.modal.run).

# Some Modal features it uses:

# * Volumes: a persisted volume lets us store and grow the published dataset over time.

# * Scheduled functions: the underlying dataset is refreshed daily, so we schedule a function to run daily.

# * Web endpoints: exposes the Datasette application for web browser interaction and API requests.

# ## Basic setup

# Let's get started writing code.
# For the Modal container image we need a few Python packages.

import asyncio
import argparse
import gzip
import pathlib
import shutil
import tqdm
import tempfile
from datetime import datetime
from urllib.request import urlretrieve

import modal

app = modal.App("example-imdb-datasette-1")
imdb_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("setuptools")
    .pip_install("tqdm")
    .pip_install("datasette~=0.63.2", "sqlite-utils")
)

# ## Persistent dataset storage

# To separate database creation and maintenance from serving, we'll need the underlying
# database file to be stored persistently. To achieve this we use a
# [`Volume`](https://modal.com/docs/guide/volumes).

volume = modal.Volume.from_name(
    "example-imdb-datasette-cache-vol", create_if_missing=True
)

DB_FILENAME = "imdb.db"
VOLUME_DIR = "/cache-vol"
DATA_DIR = pathlib.Path(VOLUME_DIR, "imdb-data")
DB_PATH = pathlib.Path(VOLUME_DIR, DB_FILENAME)

# ## Getting a dataset

# IMDB datasets are available at https://datasets.imdbws.com/
# IMDB publishes data that is updated daily. We'll filter it to only include movies and TV series.

# IMDB datasets we'll download
IMDB_FILES = [
    "title.basics.tsv.gz",      # Core movie/TV info
]

@app.function(
    image=imdb_image,
    volumes={VOLUME_DIR: volume},
    retries=2,
    timeout=1800,  # 30 minutes for large downloads
)
def download_dataset(force_refresh=False):
    """Download IMDB dataset files."""
    if DATA_DIR.exists() and not force_refresh:
        print(f"Dataset already present and force_refresh={force_refresh}. Skipping download.")
        return
    elif DATA_DIR.exists():
        print("Cleaning dataset before re-downloading...")
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading IMDB datasets...")
    base_url = "https://datasets.imdbws.com/"
    
    for filename in IMDB_FILES:
        print(f"Downloading {filename}...")
        url = base_url + filename
        output_path = DATA_DIR / filename
        
        try:
            urlretrieve(url, output_path)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise

    print("Committing the volume...")
    volume.commit()
    print("Finished downloading dataset.")


# ## Data processing

# IMDB data comes as gzipped TSV files. We need to decompress and parse them properly.

def parse_tsv_file(filepath, batch_size=50000, filter_year=None):
    """Parse a gzipped TSV file and yield batches of records."""
    import csv
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as gz_file:
        reader = csv.DictReader(gz_file, delimiter='\t')
        batch = []
        total_processed = 0
        
        for row in reader:
            # Filter: Only keep movies and TV series
            if filter_year:
                if row.get('startYear') < filter_year:
                    continue
            

            cleaned_row = {k: (None if v == '\\N' else v) for k, v in row.items()}
            
            # Type conversions for titles
            if cleaned_row.get('runtimeMinutes'):
                try:
                    cleaned_row['runtimeMinutes'] = int(cleaned_row['runtimeMinutes'])
                except (ValueError, TypeError):
                    cleaned_row['runtimeMinutes'] = None
            
            if cleaned_row.get('startYear'):
                try:
                    cleaned_row['startYear'] = int(cleaned_row['startYear'])
                except (ValueError, TypeError):
                    cleaned_row['startYear'] = None
                    
            if cleaned_row.get('endYear'):
                try:
                    cleaned_row['endYear'] = int(cleaned_row['endYear'])
                except (ValueError, TypeError):
                    cleaned_row['endYear'] = None
                    
            # Convert isAdult to boolean
            cleaned_row['isAdult'] = cleaned_row.get('isAdult') == '1'
            
            batch.append(cleaned_row)
            total_processed += 1
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield any remaining records
        if batch:
            yield batch
        
        print(f"Finished processing {total_processed:,} titles.")


# ## Inserting into SQLite
# Process IMDB data files and create SQLite database with proper indexes and views.

@app.function(
    image=imdb_image,
    volumes={VOLUME_DIR: volume},
    timeout=900, 
)
def prep_db(filter_year=None):
    """Process IMDB data files and create SQLite database."""
    import sqlite_utils
    
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
            
            with tqdm.tqdm(desc="Processing titles", unit=" batches") as pbar:
                for i, batch in enumerate(parse_tsv_file(titles_file, batch_size=50000, filter_year=filter_year)):
                    titles_table.insert_all(batch, batch_size=50000, truncate=(i == 0))
                    batch_count += len(batch)
                    total_processed += len(batch)
                    pbar.update(1)
                    pbar.set_postfix({"titles": f"{total_processed:,}"})

            print(f"Total titles in database: {batch_count:,}")
            
            # Create indexes for titles
            print("Creating indexes...")
            titles_table.create_index(["tconst"], if_not_exists=True, unique=True)
            titles_table.create_index(["primaryTitle"], if_not_exists=True)
            titles_table.create_index(["titleType"], if_not_exists=True)
            titles_table.create_index(["startYear"], if_not_exists=True)
            titles_table.create_index(["genres"], if_not_exists=True)
            print("Created indexes for titles table")
        
        # Create views for interesting queries
        db.execute("""
            CREATE VIEW IF NOT EXISTS recent_movies AS
            SELECT 
                tconst,
                primaryTitle,
                startYear,
                genres,
                runtimeMinutes
            FROM titles
            WHERE titleType = 'movie'
            AND startYear >= 2020
            ORDER BY startYear DESC, primaryTitle
        """)
        
        db.execute("""
            CREATE VIEW IF NOT EXISTS genre_stats AS
            SELECT 
                CASE 
                    WHEN genres LIKE '%Action%' THEN 'Action'
                    WHEN genres LIKE '%Comedy%' THEN 'Comedy'
                    WHEN genres LIKE '%Drama%' THEN 'Drama'
                    WHEN genres LIKE '%Horror%' THEN 'Horror'
                    WHEN genres LIKE '%Romance%' THEN 'Romance'
                    WHEN genres LIKE '%Thriller%' THEN 'Thriller'
                    WHEN genres LIKE '%Documentary%' THEN 'Documentary'
                    WHEN genres LIKE '%Animation%' THEN 'Animation'
                    ELSE 'Other'
                END as genre,
                COUNT(*) as title_count,
                AVG(runtimeMinutes) as avg_runtime
            FROM titles
            WHERE titleType = 'movie'
            AND runtimeMinutes IS NOT NULL
            GROUP BY genre
            ORDER BY title_count DESC
        """)
        
        db.close()
        
        # Copy the database to the volume
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_db_path, DB_PATH)
    
    print("Syncing DB with volume.")
    volume.commit()
    print("Volume changes committed.")


# ## Keep it fresh

# IMDB updates their data daily, so we set up
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

@app.function(
    image=imdb_image,
    volumes={VOLUME_DIR: volume},
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def ui():
    """Web endpoint for Datasette UI."""
    from datasette.app import Datasette
    
    # Configure Datasette with custom metadata
    metadata = {
        "title": "IMDB Database Explorer",
        "description": "Explore IMDB movie and TV show data",
        "databases": {
            "imdb": {
                "tables": {
                    "titles": {
                        "description": "Basic information about all titles (movies, TV shows, etc.)",
                        "columns": {
                            "tconst": "Unique identifier",
                            "titleType": "Type (movie, tvSeries, short, etc.)",
                            "primaryTitle": "Main title",
                            "originalTitle": "Original language title",
                            "isAdult": "Adult content flag",
                            "startYear": "Release year",
                            "endYear": "End year (for TV series)",
                            "runtimeMinutes": "Runtime in minutes",
                            "genres": "Comma-separated genres"
                        }
                    }
                },
                "queries": {
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
                        "title": "Movies Released in 2024"
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
                        "title": "Longest Movies (3+ hours)"
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
                        "title": "Popular Genres"
                    }
                }
            }
        }
    }
    
    ds = Datasette(
        files=[DB_PATH],
        settings={
            "sql_time_limit_ms": 60000,
            "max_returned_rows": 10000,
            "allow_download": True,
            "facet_time_limit_ms": 5000,
            "allow_facet": True,
        },
        metadata=metadata
    )
    asyncio.run(ds.invoke_startup())
    return ds.app()


# ## Publishing to the web

# Run this script using `modal run imdb_datasette.py` and it will create the database.

# You can then use `modal serve imdb_datasette.py` to create a short-lived web URL
# that exists until you terminate the script.

# When publishing the interactive Datasette app you'll want to create a persistent URL.
# Just run `modal deploy imdb_datasette.py`.

@app.local_entrypoint()
def run(*arglist):
    parser = argparse.ArgumentParser(description="IMDB Datasette App")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh the dataset")
    parser.add_argument("--filter-year", type=int, help="Filter data to be after a specific year")
    args = parser.parse_args(args=arglist)

    force_refresh = args.force_refresh
    filter_year = args.filter_year

    if force_refresh:
        print("Force refreshing the dataset...")

    if filter_year:
        print(f"Filtering data to be after {filter_year}")

    print("Downloading IMDB dataset...")
    download_dataset.remote(force_refresh=force_refresh)
    print("Processing data and creating SQLite DB...")
    prep_db.remote(filter_year=filter_year)
    print("\nDatabase ready! You can now run:")
    print("  modal serve imdb_datasette.py  # For development")
    print("  modal deploy imdb_datasette.py  # For production deployment")


# You can explore the data at the deployed web endpoint.