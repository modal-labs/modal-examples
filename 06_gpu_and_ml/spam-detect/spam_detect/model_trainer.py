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
import pickle
import random
import string
import subprocess

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    NamedTuple,
    Optional,
)

from . import config
from . import dataset
from .model_registry import ModelMetadata, TrainMetrics

logger = config.get_logger()

Email = str
Prediction = float
SpamClassifier = Callable[[Email], Prediction]
Dataset = Iterable[dataset.Example]
TrainingFunc = Callable[[Dataset], Any]
ModelBuilder = Callable[[Dataset, Optional[TrainingFunc]], SpamClassifier]


Sha256Hash = str
ModelRegistryMetadata = Dict[Sha256Hash, ModelMetadata]


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii").strip()


def serialize_model(
    model_func: SpamClassifier,
) -> bytes:
    try:
        from datasets.utils.py_utils import Pickler
    except ModuleNotFoundError:
        from pickle import Pickler  # type: ignore

    def dumps(obj, **kwds):
        file = io.BytesIO()
        Pickler(file, **kwds).dump(obj)
        return file.getvalue()

    return dumps(model_func)


def create_hashtag_from_dir(dir: pathlib.Path) -> str:
    dgst = hashlib.sha256()
    for f in dir.glob("**/*"):
        dgst.update(f.name.encode())
        dgst.update(f.read_bytes())
    return f"sha256.{dgst.hexdigest().upper()}"


def create_hashtag_from_bytes(b: bytes) -> str:
    hash_base = hashlib.sha256(b).hexdigest().upper()
    return f"sha256.{hash_base}"


def store_huggingface_model(
    trainer: Any,
    train_metrics: TrainMetrics,
    model_name: str,
    model_destination_root: pathlib.Path,
    git_commit_hash: str,
) -> str:
    """
    Accepts a Hugginface model that implements `save_model()` and stores it in model
    registry and persistent filesystem.
    """
    tmp_dirname = "".join(random.choices(string.ascii_uppercase + string.digits, k=20))
    model_save_path = model_destination_root / tmp_dirname
    trainer.save_model(output_dir=model_save_path)
    model_hashtag = create_hashtag_from_dir(model_save_path)
    model_save_path.rename(model_destination_root / model_hashtag)

    logger.info(f"serialized model's hash is {model_hashtag}")

    model_registry_metadata = load_model_registry_metadata(
        model_registry_root=model_destination_root,
    )

    model_dest_path = model_destination_root / model_hashtag
    if model_dest_path.is_file():
        logger.warning(
            (
                f"model {model_hashtag} already exists. No need to save again. "
                "consider caching model training to save compute cycles."
            )
        )

    logger.info(f"updating models registry metadata to include information about {model_hashtag}")
    metadata = ModelMetadata(
        impl_name=model_name,
        save_date=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        git_commit_hash=git_commit_hash,
    )
    store_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=model_hashtag,
        metadata=metadata,
        destination_root=model_destination_root,
    )
    logger.info("📦 done! Model stored.")
    return model_hashtag


def store_pickleable_model(
    *,
    classifier_func: SpamClassifier,
    metrics: TrainMetrics,
    model_destination_root: pathlib.Path,
    current_git_commit_hash: str,
) -> str:
    """
    Stores a pickle-able model in registry and persistent filesystem.
    The `pickle` process only works on a single classifier Python object,
    and should only be used for simple, pure-Python classifiers.
    """
    logger.info("storing spam model to model registry using pickling.")

    serialized_model = serialize_model(classifier_func)
    ser_clssfr_hash = create_hashtag_from_bytes(serialized_model)

    logger.info(f"serialized model's hash is {ser_clssfr_hash}")

    model_registry_metadata = load_model_registry_metadata(
        model_registry_root=model_destination_root,
    )

    model_dest_path = model_destination_root / ser_clssfr_hash
    if model_dest_path.is_file():
        logger.warning(
            (
                f"model {ser_clssfr_hash} already exists. No need to save again. "
                "consider caching model training to save compute cycles."
            )
        )
    else:
        logger.info(f"saving model to file at '{model_dest_path}'")
        model_dest_path.write_bytes(serialized_model)

    logger.info(f"updating models registry metadata to include information about {ser_clssfr_hash}")
    metadata = ModelMetadata(
        impl_name=model_name_from_function(classifier_func),
        save_date=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        git_commit_hash=current_git_commit_hash,
    )
    store_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=ser_clssfr_hash,
        metadata=metadata,
        destination_root=model_destination_root,
    )
    logger.info("📦 done! Model stored.")
    return ser_clssfr_hash


def model_name_from_function(model_func: SpamClassifier) -> str:
    # NOTE: This may be buggy, and create name clashes or ambiguity.
    return model_func.__qualname__


def load_model_registry_metadata(
    *,
    model_registry_root: pathlib.Path,
):
    model_registry_metadata_filepath = model_registry_root / config.MODEL_REGISTRY_FILENAME
    if not model_registry_metadata_filepath.exists():
        # Create registry metadata file on first save of a model.
        model_registry_metadata_filepath.write_text("{}")

    with open(model_registry_metadata_filepath, "r") as model_registry_f:
        data = json.load(model_registry_f)
    model_registry_metadata: ModelRegistryMetadata = {
        key: ModelMetadata(
            impl_name=value["impl_name"],
            save_date=value["save_date"],
            git_commit_hash=value["git_commit_hash"],
        )
        for key, value in data.items()
    }
    return model_registry_metadata


def retrieve_model_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    sha256_hash: str,
) -> Optional[ModelMetadata]:
    return model_registry_metadata.get(sha256_hash)


def store_model_registry_metadata(
    *,
    model_registry_metadata: ModelRegistryMetadata,
    sha256_hash: str,
    metadata: ModelMetadata,
    destination_root: pathlib.Path,
) -> None:
    existing_metadata = retrieve_model_registry_metadata(
        model_registry_metadata=model_registry_metadata,
        sha256_hash=sha256_hash,
    )
    if existing_metadata is not None:
        logger.debug("classifier with matching hash found in registry.")
        # compare new metadata with old to detect registry corruption or
        # strange renaming.
        if metadata.impl_name != existing_metadata.impl_name:
            raise RuntimeError(
                "Existing classifier with identical sha256 hash to current classifier found "
                "with conflicting metadata. "
                "Something has gone wrong."
            )
    model_registry_metadata_dict = {key: value._asdict() for key, value in model_registry_metadata.items()}
    # NOTE: Potentially overwrites with new metadata.
    model_registry_metadata_dict[sha256_hash] = metadata._asdict()
    with open(destination_root / config.MODEL_REGISTRY_FILENAME, "w") as model_registry_f:
        json.dump(model_registry_metadata_dict, model_registry_f, indent=4)


def load_pickle_serialized_model(
    *,
    sha256_hash: str,
    destination_root: pathlib.Path,
) -> SpamClassifier:
    def check_integrity(*, expected_hash: str, actual_hash: str) -> None:
        if not expected_hash == actual_hash:
            err_msg = f"Shasum integrity check failure. Expected '{expected_hash}' but got '{actual_hash}'"
            raise ValueError(err_msg)

    expected_prefix = "sha256."
    if not sha256_hash.startswith(expected_prefix):
        raise ValueError(f"model sha256 hashes are expected to start with the prefix '{expected_prefix}")

    model_path = destination_root / sha256_hash
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    hash_base = hashlib.sha256(model_bytes).hexdigest().upper()
    filestored_model_hashtag = f"sha256.{hash_base}"

    check_integrity(
        expected_hash=sha256_hash,
        actual_hash=filestored_model_hashtag,
    )
    return pickle.loads(model_bytes)
