# ---
# integration-test: false
# output-directory: "/tmp/nyc"
# ---
import io
import os
import urllib.request
from collections import Counter
from tempfile import TemporaryDirectory

import modal

stub = modal.Stub(
    image=modal.DebianSlim().pip_install(["numpy", "matplotlib", "pyarrow"])
)


if stub.is_inside():
    import numpy
    import pyarrow.parquet as pq
    from matplotlib import pyplot


@stub.function
def get_matrix(url):
    print("downloading", url, "...")

    with TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "temp.parquet")

        urllib.request.urlretrieve(url, temp_path)

        t = pq.read_table(temp_path, columns=["PULocationID", "DOLocationID"])
        c = Counter(
            (pul.as_py(), dol.as_py())
            for pul, dol in zip(t["PULocationID"], t["DOLocationID"])
        )

    return url, c


@stub.function
def main():
    urls = []

    for year in range(2018, 2021):
        for month in range(1, 13):
            urls.append(
                f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
            )

    M = Counter()
    for url, m in get_matrix.map(urls):
        M += m
        print(url, "done!")

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
