# ---
# integration-test: false
# output-directory: "/tmp/nyc"
# ---
import io
import os
from collections import Counter

import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(["numpy", "matplotlib", "duckdb"])
)


@stub.function
def get_matrix(url):
    import duckdb

    print("processing", url, "...")

    con = duckdb.connect(database=":memory:")
    con.execute("install httpfs")  # TODO: bake into the image
    con.execute("load httpfs")
    con.execute(
        "select PULocationID, DOLocationID, count(1) from read_parquet(?) group by 1, 2",
        (url,),
    )
    return {(i, j): t for i, j, t in con.fetchall()}


@stub.function
def main():
    import numpy
    from matplotlib import pyplot

    urls = []

    for year in range(2018, 2021):
        for month in range(1, 13):
            urls.append(
                f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
            )

    M = Counter()
    for m in get_matrix.map(urls):
        M += m

    max_id = max(max(k) for k in M.keys())
    matrix = numpy.matrix(
        [[M.get((i, j), 0) for j in range(max_id + 1)] for i in range(max_id + 1)]
    )

    pyplot.matshow(matrix)
    pyplot.title("Matrix of %d taxi trips" % (matrix.sum()))
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    return buf.getvalue()


OUTPUT_DIR = "/tmp/nyc"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fn = os.path.join(OUTPUT_DIR, "nyc_taxi_matrix.png")

    with stub.run():
        png_data = main()
        with open(fn, "wb") as f:
            f.write(png_data)
        print(f"wrote output to {fn}")
