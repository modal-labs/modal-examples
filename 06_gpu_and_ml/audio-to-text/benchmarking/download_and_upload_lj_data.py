# ---
# lambda-test: false  # close to timeout
# ---

import modal
from common import app, dataset_volume

DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

image = (
    modal.Image.debian_slim().pip_install("requests").add_local_python_source("common")
)


@app.function(
    volumes={"/data": dataset_volume},
    image=image,
    timeout=1200,  # 20 minutes
)
def download_and_upload_lj_data():
    import tarfile
    import tempfile
    from pathlib import Path

    import requests

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tar_path = tmpdir_path / "LJSpeech-1.1.tar.bz2"

        print("📥 Downloading dataset...")
        with requests.get(DATA_URL, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("📦 Extracting dataset...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=tmpdir_path)

        dataset_dir = tmpdir_path / "LJSpeech-1.1"

        print("☁️ Uploading to Modal volume under 'raw/'...")
        with dataset_volume.batch_upload() as batch:
            for path in dataset_dir.rglob("*"):
                if path.is_file():
                    relative_path = path.relative_to(dataset_dir)
                    remote_path = f"/raw/{relative_path}"
                    batch.put_file(str(path), remote_path)

        print("✅ Upload complete!")


@app.function(
    volumes={"/data": dataset_volume},
    image=image,
    timeout=1200,  # 20 minutes
)
def upload_lj_data_subset():
    """Upload a subset of LJSpeech files from a local ZIP file to the Modal volume."""
    import zipfile
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = Path(__file__).parent / "LJSpeech-1.1-subset.zip"

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=tmpdir_path)

        dataset_dir = tmpdir_path / "LJSpeech-1.1"

        print("☁️ Uploading to Modal volume under 'raw/'...")
        with dataset_volume.batch_upload() as batch:
            for path in dataset_dir.rglob("*"):
                if path.is_file():
                    relative_path = path.relative_to(dataset_dir)
                    remote_path = f"/raw/{relative_path}"
                    batch.put_file(str(path), remote_path)

        print("✅ Upload complete!")
