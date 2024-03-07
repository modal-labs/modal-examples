# # Use DuckDB to analyze lots of datasets in parallel
#
# The Taxi and Limousine Commission of NYC posts
# [datasets](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
# with all trips in New York City.
# They are all Parquet files, which are very well suited for
# [DuckDB](https://duckdb.org/) which has excellent
# [Parquet support](https://duckdb.org/docs/data/parquet).
# In fact, DuckDB lets us query remote Parquet data
# [over HTTP](https://duckdb.org/docs/guides/import/http_import)
# which is excellent for what we want to do here.
#
# Running this script should generate a plot like this in just 10-20 seconds,
# processing a few gigabytes of data:
#
# ![nyc taxi chart](./nyc_yellow_taxi_trips_s3_mount.png)
#
# ## Basic setup
#
# We need various imports and to define an image with DuckDB installed:

import io
import os
from datetime import datetime

import modal

stub = modal.Stub(
    "example-duckdb-nyc-taxi",
    image=modal.Image.debian_slim().pip_install("matplotlib", "duckdb"),
)


# ## DuckDB Modal function
#
# Defining the function that queries the data.
# This lets us run a SQL query against a remote Parquet file over HTTP
# Our query is pretty simple: it just aggregates total count numbers by date,
# but we also have some filters that remove garbage data (days that are outside
# the range).


@stub.function()
def get_data(year, month):
    import duckdb

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    print("processing", url, "...")

    con = duckdb.connect(database=":memory:")
    con.execute("install httpfs")  # TODO: bake into the image
    con.execute("load httpfs")
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
    con.execute(q, (url, year, month))
    return list(con.fetchall())


# ## Plot results
#
# Let's define a separate function which:
# 1. Parallelizes over all files and dispatches calls to the previous function
# 2. Aggregate the data and plot the result


@stub.function()
def create_plot():
    from matplotlib import pyplot

    # Map over all inputs and combine the data
    inputs = [
        (year, month)
        for year in range(2018, 2023)
        for month in range(1, 13)
        if (year, month) <= (2022, 6)
    ]
    data: list[list[tuple[datetime, int]]] = [
        [] for i in range(7)
    ]  # Initialize a list for every weekday
    for r in get_data.starmap(inputs):
        for d, c in r:
            data[d.weekday()].append((d, c))

    # Initialize plotting
    pyplot.style.use("ggplot")
    pyplot.figure(figsize=(16, 9))

    # For each weekday, plot
    for i, weekday in enumerate(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ):
        data[i].sort()
        dates = [d for d, _ in data[i]]
        counts = [c for _, c in data[i]]
        pyplot.plot(dates, counts, linewidth=3, alpha=0.8, label=weekday)

    # Plot annotations
    pyplot.title("Number of NYC yellow taxi trips by weekday, 2018-2022")
    pyplot.ylabel("Number of daily trips")
    pyplot.legend()
    pyplot.tight_layout()

    # Dump PNG and return
    with io.BytesIO() as buf:
        pyplot.savefig(buf, format="png", dpi=300)
        return buf.getvalue()


# ## Entrypoint
#
# Finally, we have some simple entrypoint code that kicks everything off.
# Note that the plotting function returns raw PNG data that we store locally.
#
# Run this local entrypoint with `modal run`.


@stub.local_entrypoint()
def main():
    output_dir = "/tmp/nyc"
    os.makedirs(output_dir, exist_ok=True)

    fn = os.path.join(output_dir, "nyc_taxi_chart.png")
    png_data = create_plot.remote()
    with open(fn, "wb") as f:
        f.write(png_data)
    print(f"wrote output to {fn}")
