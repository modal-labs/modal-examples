import os
import pathlib
import shutil
import subprocess
import zipfile

import modal


bucket_creds = modal.Secret.from_name("aws-s3-modal-examples-datasets", environment_name="main")
bucket_name = "modal-examples-datasets"
volume = modal.CloudBucketMount(
    bucket_name,
    secret=bucket_creds,
)
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

    tmp_path = pathlib.Path("/tmp/imagenet/")
    vol_path = pathlib.Path("/vol/imagenet/")
    filename = "imagenet-object-localization-challenge.zip"
    dataset_path = vol_path / filename
    if dataset_path.exists():
        dataset_size = dataset_path.stat().st_size
        if dataset_size < (150 * 1024 * 1024 * 1024):
            dataset_size_gib = dataset_size / (1024 * 1024 * 1024)
            raise RuntimeError(f"Partial download of dataset .zip. It is {dataset_size_gib} but should be > 150GiB")
    else:
        subprocess.run(
            f"kaggle competitions download -c imagenet-object-localization-challenge --path {tmp_path}",
            shell=True,
            check=True
        )
        shutil.copy(tmp_path / filename, dataset_path)

    # Extract dataset
    extracted_dataset_path = vol_path / "extracted"
    extracted_dataset_path.mkdir(exist_ok=True)
    print("Extracting .zip into volume...")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dataset_path)
    print(f"Unzipped {dataset_path} to {extracted_dataset_path}")
