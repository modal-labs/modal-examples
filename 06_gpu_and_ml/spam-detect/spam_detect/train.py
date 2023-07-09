# # A Plan for Spam, 20 Years On: LLM vs. Naive Bayes
#
# This example trains multiple models (LLM, Naive Bayes) to perform
# spam classification on the ENRON email dataset. This is a return to Paul Graham's
# well-known 2002 post, A Plan For Spam (http://www.paulgraham.com/spam.html).
#
# Graham's original post focused on the Naive Bayes model. Here we pit that model against
# a current state-of-the-art large-language-model (LLM). Both models are trained on the dataset
# and served via a model API (serving.py).
#
# This module, train.py, is the model training entrypoint, providing functions to do CPU/GPU training
# before saving to disk. The other significant modules are as follows:
#
# * models.py — contains the core `SpamModel` interface and three implementing model classes, including `LLMSpamClassifier`.
# * serving.py — a minimal FastAPI model serving API, loading models by ID from a Modal persistent volume.
# * model_registry.py — defines minimal data structures and CLI commands for a model registry stored on Modal.
# * model_storage.py — functions concerned withn serializing and deserializing (ie. loading) the trained ML models.
#

import pathlib
import random
import subprocess
from datetime import timedelta

import modal

from . import config
from . import dataset
from . import models
from .app import stub, volume


def fetch_git_commit_hash(allow_dirty: bool) -> str:
    # Ensure git state is clean so that the git commit hash accurately reflects
    # the configuration of the training run.
    #
    # Ignoring dirty git state when kicking off a training run means accepting
    # unreproducible model training outcomes.
    if not allow_dirty:
        if (
            subprocess.run(
                ("git", "diff-index", "--quiet", "--cached", "HEAD", "--")
            ).returncode
            != 0
        ):
            breakpoint()
            raise RuntimeError(
                "Dirty git status. Repository has staged but not yet committed changes.\n"
                "Commit these changes or remove them to get a clean git state."
            )
        elif subprocess.run(("git", "diff-files", "--quiet")).returncode != 0:
            raise RuntimeError(
                "Dirty git status. Repository has changes that could be staged.\n"
                "Commit these changes or add them to .gitignore."
            )
        res = subprocess.run(
            ("git", "ls-files", "--exclude-standard", "--others"),
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Could not check `git` for untracked files. {res.stderr.decode()}"
            )
        if res.stdout:
            raise RuntimeError(
                "Dirty git status. Repository has untracked files.\n"
                "Remove these files, commit them, or add them to .gitignore."
            )
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
    )
    return result.stdout.decode().strip()


@stub.function(volumes={config.VOLUME_DIR: volume})
def init_volume():
    config.MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    stub.app.volume.commit()  # Persist changes


@stub.function(
    timeout=int(timedelta(minutes=8).total_seconds()),
    volumes={config.VOLUME_DIR: volume},
)
def prep_dataset():
    logger = config.get_logger()
    datasets_path = config.DATA_DIR
    datasets_path.mkdir(parents=True, exist_ok=True)
    dataset.download(base=datasets_path, logger=logger)
    stub.app.volume.commit()  # Persist changes


@stub.function(
    volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def train(
    model: models.SpamModel, dataset_path: pathlib.Path, git_commit_hash: str
):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    random.shuffle(enron_dataset)
    classifier, metrics = model.train(enron_dataset)
    model_id = model.save(
        fn=classifier,
        metrics=metrics,
        model_registry_root=config.MODEL_STORE_DIR,
        git_commit_hash=git_commit_hash,
    )
    stub.app.volume.commit()  # Persist changes
    logger.info(f"saved model to model store. {model_id=}")
    # Reload the model
    logger.info("🔁 testing reload of model")
    classifier = model.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )
    is_spam = classifier("fake email!")
    print(f"classification: {is_spam=}")


@stub.function(
    volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
    gpu=modal.gpu.T4(),
)
def train_gpu(
    model: models.SpamModel, dataset_path: pathlib.Path, git_commit_hash: str
):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    random.shuffle(enron_dataset)
    classifier, metrics = model.train(enron_dataset)
    model_id = model.save(
        fn=classifier,
        metrics=metrics,
        model_registry_root=config.MODEL_STORE_DIR,
        git_commit_hash=git_commit_hash,
    )
    stub.app.volume.commit()  # Persist changes
    logger.info(f"saved model to model store. {model_id=}")


@stub.function(
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def main(git_commit_hash: str, model_type=config.ModelType.BAD_WORDS):
    logger = config.get_logger()
    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )
    dataset_path = dataset.dataset_path(config.DATA_DIR)

    logger.info(
        f"💪 training a {model_type} model at git commit {git_commit_hash[:8]}"
    )
    if model_type == config.ModelType.NAIVE_BAYES:
        train.call(
            model=models.NaiveBayes(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    elif model_type == config.ModelType.LLM:
        train_gpu.call(
            model=models.LLM(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    elif model_type == config.ModelType.BAD_WORDS:
        train.call(
            model=models.BadWords(),
            dataset_path=dataset_path,
            git_commit_hash=git_commit_hash,
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'")


# Pass in the string representation of a supported `ModelType` to train
# a model of that type on the dataset.
#
# Example:
#
# ```
# modal run spam_detect.train::stub.train_model --model-type "BAD_WORDS"
# ```


@stub.local_entrypoint()
def train_model(model_type: str):
    model_type_val = config.ModelType(model_type)
    # All training runs are versioned against git repository state.
    git_commit_hash: str = fetch_git_commit_hash(allow_dirty=False)
    init_volume.call()
    main.call(
        git_commit_hash=git_commit_hash,
        model_type=model_type_val,
    )


if __name__ == "__main__":
    with stub.run():
        train_model(model_type="NAIVE_BAYES")
