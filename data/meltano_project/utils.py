BUCKET = "modal-examples"
KEY = "meltano.db"
DB_PATH = "/meltano_project/.meltano/meltano.db"


def download_meltano_db():
    import boto3

    s3 = boto3.client("s3")
    try:
        contents = s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()
        with open(DB_PATH, "wb") as f:
            f.write(contents)
        print(f"Downloaded sqlite DB of size {len(contents)} bytes.")
    except s3.exceptions.NoSuchKey:
        print("No DB found.")


def upload_meltano_db():
    import boto3

    s3 = boto3.client("s3")
    contents = open(DB_PATH, "rb").read()
    s3.put_object(Bucket=BUCKET, Key=KEY, Body=contents)
    print(f"Uploaded sqlite DB of size {len(contents)} bytes.")
