import os
import pathlib
import subprocess
import modal


bucket_creds = modal.Secret.from_name("aws-s3-modal-examples-datasets", environment_name="main")
bucket_name = "modal-examples-datasets"
volume = modal.Volume.from_name("example-imagenet", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("kaggle")
app = modal.App(
    "example-imagenet-dataset-import",
    image=image,
    secrets=[modal.Secret.from_name("kaggle-api-token")],
)

@app.function(
    volumes={"/vol/": volume},
    timeout=60 * 60 * 5,  # 5 hours
    _allow_background_volume_commits=True,
)
def import_transform_load() -> None:
    kaggle_api_token_data = os.environ["KAGGLE_API_TOKEN"]
    kaggle_token_filepath = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    kaggle_token_filepath.parent.mkdir(exist_ok=True)
    kaggle_token_filepath.write_text(kaggle_api_token_data)

    p = pathlib.Path("/vol/imagenet/")
    subprocess.run(f"kaggle competitions download -c imagenet-object-localization-challenge --path {p}", shell=True, check=True)
