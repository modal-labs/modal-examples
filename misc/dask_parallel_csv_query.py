# Parallel Dask dataframe processing on large CSV files.
#
# This seemingly contrived but actually real-world example (https://twitter.com/pwang/status/1574622180067123200)
# shows how to:
#
#     1. Use multi-part uploads to create a single massive .CSV file in parallel
#     2. Use Dask dataframe partitioning to query that uploaded .CSV file in parallel.
#
# All this example requires as setup is an AWS user/role with permissions to create buckets and
# R/W them.

import csv
import io
import random
import string
import sys
import time

import modal

stub = modal.Stub("dask-parallel-csv")
image = modal.Image.debian_slim().pip_install(
    [
        "boto3",
        "dask",
        "numpy",
        "pandas",
        "pyarrow",
        "s3fs",
    ]
)

BUCKET_NAME = "temp-big-data-csv"

# ## Uploading a lot of fake data quickly
#
# On Reddit u/kawaii_kebab posted that their simple data analysis problem
# was taking an annoying amount of time with Apache Spark – 10 minutes on a 90GB CSV file.
#
# CSV is an inefficient, but splittable data format. The following code uses AWS S3's multipart-upload
# to create an N gigabyte CSV file conforming to the schema given in u/kawaii_kebab's Reddit post.


def random_alphanum_str(min_len: int, max_len: int) -> str:
    s_len = random.randrange(min_len, max_len)
    return "".join(random.choices(string.ascii_letters + string.digits, k=s_len))


def fake_csv_data(size_mb: int):
    """u/kawaii_kebab's analysis problem had the schema 'email STRING, password STRING.'"""
    domains = {}
    csvfile = io.StringIO()
    spamwriter = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)

    approx_entry_bytes = 42
    entries_per_mb = (1024 * 1024) / approx_entry_bytes
    required_entries = int(size_mb * entries_per_mb)
    domains = {
        "gmail.com": 0.25,
        "hotmail.com": 0.2,
        "yahoo.com": 0.2,
        "outlook.com": 0.15,
        "proton.me": 0.15,
        "foo.com": 0.05,
    }
    print(f"Producing {required_entries} lines of (email, password) CSV data.")
    for dom in random.choices(
        population=list(domains.keys()),
        weights=list(domains.values()),
        k=required_entries,
    ):
        local_part = random_alphanum_str(min_len=5, max_len=25)
        email = f"{local_part}@{dom}"
        password = random_alphanum_str(min_len=5, max_len=25)
        spamwriter.writerow([email, password])
    data = csvfile.getvalue()
    csvfile.close()
    return data


@stub.function(
    concurrency_limit=30,
    image=image,
    secret=modal.Secret.from_name("personal-aws-user"),
)
def upload_part(bucket, key, upload_id, part_num, size_mb):
    import boto3

    s3_resource = boto3.resource("s3")
    print(f"Uploading part {part_num} for upload ID {upload_id}")
    upload_part = s3_resource.MultipartUploadPart(
        bucket,
        key,
        upload_id,
        part_num,
    )

    part_data = fake_csv_data(size_mb=size_mb)
    print(f"Part {part_num} is {sys.getsizeof(part_data)} bytes")
    part_response = upload_part.upload(
        Body=part_data,
    )
    return (part_num, part_response)


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def upload_fake_csv(desired_mb: int):
    import boto3

    bucket_name = BUCKET_NAME
    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=bucket_name)

    print(boto3.client("sts").get_caller_identity())

    key = f"{desired_mb}_mb.csv"
    multipart_upload = s3_client.create_multipart_upload(
        ACL="private",
        Bucket=bucket_name,
        Key=key,
    )

    upload_id = multipart_upload["UploadId"]
    print(f"Upload ID: {upload_id}")
    upload_size_mb = 200
    num_uploads = desired_mb // upload_size_mb

    uploads = [(bucket_name, key, upload_id, i, upload_size_mb) for i in range(1, num_uploads + 1)]

    parts = []
    for (part_number, part_response) in upload_part.starmap(uploads):
        parts.append({"PartNumber": part_number, "ETag": part_response["ETag"]})

    print("Completing upload...")
    result = s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=key,
        MultipartUpload={"Parts": parts},
        UploadId=multipart_upload["UploadId"],
    )
    print("✅ Done. S3 upload result: ")
    print(result)


# ## Using Dask to partition the CSV and query in parallel.
#
# This example's query over the CSV file is a simple `grep | wc -l`, checking how many
# rows in the fake dataset contain Gmail email addresses.
#
# Becase we created the dataset, we know the expected count is about 25% of the total rows.


@stub.function(
    image=image,
    concurrency_limit=100,
    cpu=5.0,
    memory=1024,
    secret=modal.Secret.from_name("personal-aws-user"),
)
def process_block(i, df):
    import logging
    from dask.diagnostics import Profiler

    logging.getLogger("fsspec").setLevel(logging.DEBUG)
    logging.getLogger("s3fs").setLevel(logging.DEBUG)
    start = time.time()
    series = df.partitions[i]["email"]
    with Profiler() as prof:
        count = series.str.endswith("@gmail.com").count().compute(scheduler="single-threaded")
    end = time.time()
    elapsed = end - start
    print(f"Counted {count} in csv partition {i}. Took {(elapsed):.2f} seconds.")
    if elapsed > 60:
        print(prof.results)
    return int(count)


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def count_by_filter_fast(bucket: str, key: str) -> int:
    import boto3
    from dask.dataframe.io import read_csv

    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=bucket, Key=key)
    size = response["ContentLength"]
    print(f"CSV file is {size} bytes")

    blocksize = "128 MiB"
    print(f"Reading and partitioning csv with block size of {blocksize}")

    csv_file_path = f"s3://{bucket}/{key}"
    df = read_csv(
        urlpath=csv_file_path,
        blocksize=blocksize,
        sep=" ",
        names=["email", "password"],
    )

    print(f"Created dataframe has {df.npartitions} partitions.")
    return sum(process_block.starmap(((i, df) for i in range(df.npartitions)), order_outputs=False))


USAGE = """
python main.py create-csv GIGABYTES
python main.py run GIGABYTES

COMMANDS:

create-csv: parallel uploads to an S3 bucket `GIGABYTES` Gigabytes of fake data as a single .csv file.
run:        run parallel query using Dask on a previously uploaded .csv `GIGABYTES` Gigabytes size.
"""

if __name__ == "__main__":
    cmd = sys.argv[1]
    gb_size = int(sys.argv[2])
    with stub.run() as app:
        print(f"{app.app_id=}")

        if cmd == "create-csv":
            start = time.time()
            upload_fake_csv(desired_mb={gb_size * 1000})
            end = time.time()
            print(f"Created fake data in {end - start} seconds.")
        elif cmd == "run":
            key = f"{gb_size * 1000}_mb.csv"
            start = time.time()
            count = count_by_filter_fast(bucket=BUCKET_NAME, key=key)
            end = time.time()
            print(f"Returned {count=} in {end - start} seconds.")
        else:
            exit(f"Unknown '{cmd}'.\n\n{USAGE}")
