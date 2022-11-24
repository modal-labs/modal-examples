import json
import pathlib
import shutil
import tarfile
import tempfile
import urllib.request
from typing import Iterable

from ...config import _get_logger
from .structure import Example

logger = _get_logger()

enron_raw_dataset_url_root = "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/raw/"
enron_dataset_files = {
    "ham": ["beck-s", "farmer-d", "kaminski-v", "kitchen-l", "lokay-m", "williams-w3"],
    "spam": ["BG", "GP", "SH"],
}


def _download_and_extract_dataset(destination_root_path: pathlib.Path):
    logger.info("Downloading raw enron dataset.")
    for key, files in enron_dataset_files.items():
        for value in files:
            logger.info(f"Downloading raw enron dataset file {key}/{value}.")
            dataset_file_url = f"{enron_raw_dataset_url_root}{key}/{value}.tar.gz"
            destination_path = destination_root_path / key / f"{value}.tar.gz"
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(
                dataset_file_url,
                # Set a user agent to avoid 403 response from some podcast audio servers.
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req) as response, open(destination_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

            logger.info(f"Extracting raw enron dataset file {key}/{value}")
            try:
                tar = tarfile.open(destination_path)
                tar.extractall(destination_path.parent)
                tar.close()
            except EOFError:
                # @ Thu Nov 24 2022 some tarballs are broken, and only provide partial data.
                pass
            destination_path.unlink()


def download(destination: pathlib.Path) -> None:
    processed_dataset_path = destination / "processed_raw_dataset.json"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_path_name:
        tmp_path = pathlib.Path(tmp_path_name)
        _download_and_extract_dataset(destination_root_path=tmp_path)
        # TODO(Jonathon): This only produces ~50,000 examples.
        # Other links to the dataset claim ~500,000 examples, eg.
        # https://www.kaggle.com/wcukierski/enron-email-dataset
        ds: list[Example] = []
        for pth in tmp_path.glob("**/*"):
            if pth.is_dir() or str(pth).endswith("tar.gz"):
                continue
            # A single file looks like 'datasets/enron/raw/ham/lokay-m/enron_t_s/25'
            # and it contains a single plain text email.
            is_spam = "raw/spam" in str(pth)
            with open(pth, "r", encoding="latin-1") as f:
                ex = Example(
                    email=f.read(),
                    spam=is_spam,
                )
                ds.append(ex)

    logger.info("writing processed raw dataset to file.")
    with open(processed_dataset_path, "w") as f:
        json.dump(ds, f, indent=4)
