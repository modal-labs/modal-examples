import json
import pathlib

from typing import NamedTuple, MutableSequence


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


def deserialize_clean_dataset(dataset_path: pathlib.Path) -> CleanEnronDataset:
    with open(dataset_path, "r") as f:
        clean_data = json.load(f)
    return {key: Example(email=item[0], spam=bool(item[1])) for key, item in clean_data.items()}
