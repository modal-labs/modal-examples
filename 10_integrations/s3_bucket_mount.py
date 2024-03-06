# # Mount S3 buckets
#
# This example shows how to mount an S3 bucket in a Modal app with [`CloudBucketMount`](/docs/reference/modal.CloudBucketMount).
# We will both write and read from that bucket in parallel using [`map`](/docs/reference/modal.Function#map) and [`starmap`](/docs/reference/modal.Function#starmap).
#
# ## Basic setup
#
# You will need to have a S3 bucket and AWS credentials to run this example. Refer to the documentation
# for detailed [IAM permissions](/docs/guide/cloud-bucket-mounts#iam-permissions) your credentials will need.
#
# After you are done creating a bucket and configuring IAM settings, 
# you now need to create a [Modal Secret](/docs/guide/secrets). Navigate to the "Secrets" tab and
# click on the AWS card, then fill in the fields with the AWS key and secret created
# previously. Name the Secret `s3-bucket-secret`.

from datetime import datetime
from pathlib import Path

from modal import CloudBucketMount, Secret, Image, Stub

image = (
    Image.debian_slim()
        .pip_install(
            "requests==2.31.0",
            "duckdb==0.10.0"
        )
)
stub = Stub(image=image)

MOUNT_PATH: Path = Path("/bucket")
YELLOW_TAXI_DATA_PATH: Path = MOUNT_PATH / "yellow_taxi"


# Dependencies are not available locally. The following block instructs Modal
# to only make imports inside the container.
with image.imports():
    import duckdb
    import requests


# ## Download New York City's taxi data
#
# NYC makes data about taxi rides publicly available. The city's [Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
# publishes files in the Parquet format. Files are organized by year and month.
#
# We are going to download all available files and store them in an S3 bucket. We do this by
# mounting a `modal.CloudBucketMount` with the S3 bucket name and its respective credentials.
# The bucket will be mounted in `MOUNT_PATH`.
#
# This function takes two parameters: year and month. This allows us to use Modal's [`starmap`](/docs/reference/modal.Function#starmap)
# to paralellize this function, downloading files in parallel into our S3 bucket.
@stub.function(
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=Secret.lookup("s3-bucket-secret"),
        )
    },
)
def download_data(year:int, month:int) -> str:
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        if not YELLOW_TAXI_DATA_PATH.exists():
            YELLOW_TAXI_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Skip downloading if file exists.
        s3_path = MOUNT_PATH / filename
        if not s3_path.exists():
            print(f"downloading => {s3_path}")
            with open(s3_path, "wb") as file:
                for chunk in r.iter_content(chunk_size=8192):
                    file.write(chunk)

    return s3_path.as_posix()

# ## Analyze data with DuckDB
#
# [DuckDB](https://duckdb.org/) is an analytical database with rich support for Parquet files.
# It is also very fast. We will write a Modal Function that aggregates the number of yellow taxi trips
# per Parquet file in our S3 bucket (files are organized by year and month combination).
# This will allow for parallelism using Modal's [`map`](/docs/reference/modal.Function#map).
@stub.function(
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=Secret.lookup("s3-bucket-secret"),
        )
    },
)
def aggregate_data(path:str) -> tuple[datetime, int]:
    print(f"processing => {path}")

    # Parse file.
    year_month_part = path.split("yellow_tripdata_")[1]
    year, month = year_month_part.split("-")
    month = month.replace(".parquet", "")

    # Make DuckDB query using in-memory storage.
    con = duckdb.connect(database=":memory:")
    q = """
    with sub as (
        select tpep_pickup_datetime::date d, count(1) c
        from read_parquet(?)
        group by 1
    )
    select d, c from sub
    where date_part('year', d) = ?  -- filter out garbage
    and date_part('month', d) = ?   -- same
    """
    con.execute(q, (path, year, month))
    return list(con.fetchall())

# ## Plot daily taxi rides
#
# Here we write a function to create a plot withour results.
# The plot shows the number of yellow taxi rides per day in NYC.
# This function will run locally and store the resulting plot image in `./nyc_yellow_taxi_trips.png`.
def plot(dataset) -> None:
    import matplotlib.pyplot as plt

    # Sorting data by date
    dataset.sort(key=lambda x: x[0])

    # Unpacking dates and values
    dates, values = zip(*dataset)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dates, values)
    plt.title("Number of NYC yellow taxi trips by weekday, 2018-2023")
    plt.ylabel("Number of daily trips")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./nyc_yellow_taxi_trips_s3_mount.png")

# ## Run everything
#
# Create a `@stub.local_entrypoint()` for your Modal program. This allows you to
# run your program with the CLI by calling `modal run s3_bucket_mount.py`. We call both our Modal Functions
# in parallel. We first call `download_data()` with `starmap`. `starmap` will map
# a tuple of inputs (year, month) to the function's parameters. This will download
# all yellow taxi data files into our locally mounted S3 bucket. Then, we call
# `aggregate_data()` with `map` on a list of Parquet file paths. These files are
# also read from our S3 bucket. One function writes files to S3 and the other
# reads files from S3 in; both run in parallel.
#
# Finally, we call `plot()` locally generating the following image:
#
# ![Number of NYC yellow taxi trips by weekday, 2018-2023](./10_integrations/nyc_yellow_taxi_trips_s3_mount.png)
#
# This program shoulld run in less than 30 seconds.
@stub.local_entrypoint()
def main():

    # List of tuples[year, month].
    inputs = [
        (year, month)
        for year in range(2018, 2023)
        for month in range(1, 13)
    ]

    # List of file paths in S3.
    parquet_files:list[str] = []
    for path in download_data.starmap(inputs):
        print(f"done => {path}")
        parquet_files.append(path)

    # List of datetimes and number of yellow taxi trips.
    dataset = []
    for r in aggregate_data.map(parquet_files):
        dataset += r

    plot(dataset)
