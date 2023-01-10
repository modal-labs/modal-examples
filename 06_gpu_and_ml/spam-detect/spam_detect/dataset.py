"""
Module for the fetching, pre-processing, and loading of spam classification datasets.
Currently only provides access to the ENRON email dataset.
"""
import csv
import json
import pathlib
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import timedelta
from typing import Iterable, NamedTuple, MutableSequence

# TODO:
# This dataset only produces ~50,000 examples.
# Other links to the dataset claim ~500,000 examples, eg. https://www.kaggle.com/wcukierski/enron-email-dataset
# which links to https://www.cs.cmu.edu/~./enron/.
enron_dataset_url = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"


class Example(NamedTuple):
    email: str
    spam: bool


RawEnronDataset = MutableSequence[Example]
CleanEnronDataset = dict[str, Example]


def dataset_path(base: pathlib.Path) -> pathlib.Path:
    return base / "raw" / "enron" / "all.json"


def deserialize_dataset(dataset_path: pathlib.Path) -> RawEnronDataset:
    with open(dataset_path, "r") as f:
        items = json.load(f)
    return [Example(email=item[0], spam=bool(item[1])) for item in items]


def _download_and_extract_dataset(destination_root_path: pathlib.Path, logger):
    logger.info("Downloading raw enron dataset.")
    destination_path = destination_root_path / "enron.zip"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        enron_dataset_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response, open(destination_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    with zipfile.ZipFile(destination_path, "r") as zip_ref:
        logger.info(f"Extracting zip with contents: ")
        zip_ref.printdir()
        zip_ref.extractall(destination_root_path)

    return destination_root_path / "enron_spam_data.csv"


def fix_nulls(f):
    for line in f:
        yield line.replace("\0", "")


def download(logger, base: pathlib.Path) -> None:
    dest = dataset_path(base)
    dest.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = pathlib.Path(tempfile.TemporaryDirectory().name)
    dataset_csv_path = _download_and_extract_dataset(destination_root_path=tmp_path, logger=logger)
    ds: list[Example] = []
    spam_count = 0
    with open(dataset_csv_path, "r") as csvfile:
        csv.field_size_limit(100_000_000)
        reader = csv.DictReader(fix_nulls(csvfile), delimiter=",")
        for row in reader:
            is_spam = row["Spam/Ham"] == "spam"
            if is_spam:
                spam_count += 1
            ex = Example(
                email=row["Subject"] + " " + row["Message"],
                spam=is_spam,
            )
            ds.append(ex)

    spam_percentage = round((spam_count / len(ds)) * 100, ndigits=4)
    logger.info(
        f"writing processed raw dataset to file. dataset contains {len(ds)} examples and is {spam_percentage}% spam"
    )
    with open(dest, "w") as f:
        json.dump(ds, f, indent=4)
