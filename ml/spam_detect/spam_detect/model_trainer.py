"""
The model trainer module contains functions for the training
management, serialization, and storage of the email spam models defined
within models.py.
"""
import datetime
import hashlib
import io
import json
import pathlib
import subprocess

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
)

from . import config
from .datasets.enron import structure as dataset

logger = config._get_logger()

Email = str
Prediction = float
SpamClassifier = Callable[[Email], Prediction]
Dataset = Iterable[dataset.Example]
TrainingFunc = Callable[[Dataset], Any]
ModelBuilder = Callable[[Dataset, Optional[TrainingFunc]], SpamClassifier]

MODEL_REGISTRY_FILENAME = "registry.json"


class ClassifierMetadata(NamedTuple):
    impl_name: str
    save_date: str
    git_commit_hash: str


Sha256Hash = str
ModelRegistryMetadata = Dict[Sha256Hash, ClassifierMetadata]


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii").strip()


def serialize_classifier(
    classifier_func: SpamClassifier,
) -> bytes:
    try:
        from datasets.utils.py_utils import Pickler
    except ModuleNotFoundError:
        from pickle import Pickler

    def dumps(obj, **kwds):
        file = io.BytesIO()
        Pickler(file, **kwds).dump(obj)
        return file.getvalue()

    return dumps(classifier_func)


def store_classifier(
    *,
    classifier_func: SpamClassifier,
    classifier_destination_root: pathlib.Path,
    current_git_commit_hash: str,
) -> pathlib.Path:
    logger.info("storing spam classifier to model registry.")

    serialized_classifier = serialize_classifier(classifier_func)
    hash_base = hashlib.sha256(serialized_classifier).hexdigest().upper()
    ser_clssfr_hash = f"sha256.{hash_base}"

    logger.info(f"serialized classifier's hash is {ser_clssfr_hash}")

    model_registry_metadata = load_classifier_registry_metadata(
        classifier_destination_root=classifier_destination_root,
    )

    classifier_dest_path = classifier_destination_root / ser_clssfr_hash
    if classifier_dest_path.is_file():
        logger.warning(
            (
                f"Classifier {ser_clssfr_hash} already exists. No need to save again. "
                "Consider caching model training to save compute cycles."
            )
        )
    else:
        logger.info(f"Saving classifier to file at '{classifier_dest_path}'")
        classifier_dest_path.write_bytes(serialized_classifier)

    logger.info(f"Updating models registry metadata to include information about {ser_clssfr_hash}")
    metadata = ClassifierMetadata(
        impl_name=classifier_name_from_function(classifier_func),
        save_date=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        git_commit_hash=current_git_commit_hash,
    )
    store_classifier_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        classifier_sha256_hash=ser_clssfr_hash,
        classifier_metadata=metadata,
        classifier_destination_root=classifier_destination_root,
    )
    logger.info("Done! Classifier model stored 📦.")
    return ser_clssfr_hash


def classifier_name_from_function(classifier_func: SpamClassifier) -> str:
    # NOTE: This may be buggy, and create name clashes or ambiguity.
    return classifier_func.__qualname__


def load_classifier_registry_metadata(
    *,
    classifier_destination_root: pathlib.Path,
):
    model_registry_metadata_filepath = classifier_destination_root / MODEL_REGISTRY_FILENAME
    if not model_registry_metadata_filepath.exists():
        # Create registry metadata file on first save of a model.
        model_registry_metadata_filepath.write_text("{}")

    with open(model_registry_metadata_filepath, "r") as model_registry_f:
        data = json.load(model_registry_f)
    model_registry_metadata: ModelRegistryMetadata = {
        key: ClassifierMetadata(
            impl_name=value["impl_name"],
            save_date=value["save_date"],
            git_commit_hash=value["git_commit_hash"],
        )
        for key, value in data.items()
    }
    return model_registry_metadata


def retrieve_classifier_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    classifier_sha256_hash: str,
) -> Optional[ClassifierMetadata]:
    return model_registry_metadata.get(classifier_sha256_hash)


def store_classifier_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    classifier_sha256_hash: str,
    classifier_metadata: ClassifierMetadata,
    classifier_destination_root: pathlib.Path,
) -> None:
    existing_metadata = retrieve_classifier_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        classifier_sha256_hash=classifier_sha256_hash,
    )
    if existing_metadata is not None:
        logger.debug("Classifier with matching hash found in registry.")
        # compare new metadata with old to detect registry corruption or
        # strange renaming.
        if classifier_metadata.impl_name != existing_metadata.impl_name:
            raise RuntimeError(
                "Existing classifier with identical sha256 hash to current classifier found "
                "with conflicting metadata. "
                "Something has gone wrong."
            )
    model_registry_metadata_dict = {key: value._asdict() for key, value in model_registry_metadata.items()}
    # NOTE: Potentially overwrites with new metadata.
    model_registry_metadata_dict[classifier_sha256_hash] = classifier_metadata._asdict()
    with open(classifier_destination_root / MODEL_REGISTRY_FILENAME, "w") as model_registry_f:
        json.dump(model_registry_metadata_dict, model_registry_f, indent=4)


def load_serialized_classifier(
    *,
    classifier_sha256_hash: str,
    classifier_destination_root: pathlib.Path,
) -> SpamClassifier:
    # TODO: Check registry first???

    def check_integrity(*, expected_hash: str, actual_hash: str) -> None:
        if not expected_hash == actual_hash:
            err_msg = f"Shasum integrity check failure. Expected '{expected_hash}' but got '{actual_hash}'"
            raise ValueError(err_msg)

    expected_prefix = "sha256."
    if not classifier_sha256_hash.startswith(expected_prefix):
        raise ValueError(f"Classifier sha256 hashes are expected to start with the prefix '{expected_prefix}")

    classifier_path = classifier_destination_root / classifier_sha256_hash
    with open(classifier_path, "rb") as f:
        classifier_bytes = f.read()

    hash_base = hashlib.sha256(classifier_bytes).hexdigest().upper()
    loaded_classifier_hash = f"sha256.{hash_base}"

    check_integrity(
        expected_hash=classifier_sha256_hash,
        actual_hash=loaded_classifier_hash,
    )
    import pickle

    return pickle.loads(classifier_bytes)


# def main(argv: Optional[Sequence[str]] = None) -> int:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-registry-root", required=True)
#     args = parser.parse_args(argv)
#     model_registry_root = pathlib.Path(args.model_registry_root)

#     store_classifier(
#         classifier_func=models.bad_words_spam_classifier,
#         classifier_destination_root=model_registry_root,
#         current_git_commit_hash=get_git_revision_hash(),
#     )
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
