import pathlib
from datetime import timedelta

import modal

from . import config
from . import models
from .app import stub, volume
from .datasets.enron import structure as enron


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
)
def train(model: models.SpamModel, dataset_path: pathlib.Path):
    logger = config.get_logger()
    enron_dataset = enron.deserialize_dataset(dataset_path)
    classifier = model.train(enron_dataset)
    model_id = model.save(fn=classifier, model_registry_root=config.MODEL_STORE_DIR)
    logger.info(f"saved model to model store. {model_id=}")
    # Reload the model
    logger.info(f"🔁 testing reload of model")
    classifier = model.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )
    print(classifier)
    print(classifier("fake email!"))


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
    # NOTE: Can't use A100 easily because:
    # "Modal SharedVolume data will not be shared between A100 and non-A100 functions"
    gpu=True,
)
def train_gpu(model: models.SpamModel, dataset_path: pathlib.Path):
    logger = config.get_logger()
    enron_dataset = enron.deserialize_dataset(dataset_path)
    classifier = model.train(enron_dataset)
    model_id = model.save(fn=classifier, model_registry_root=config.MODEL_STORE_DIR)
    logger.info(f"saved model to model store. {model_id=}")


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
    gpu=True,
)
def main():
    logger = config.get_logger()
    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )
    dataset_path = enron.dataset_path(config.DATA_DIR)

    model_type = "LLM"  # Change to train different models.

    logger.info("💪 training ...")
    if model_type == "NAIVE BAYES":
        model = models.NaiveBayes()
        train(model, dataset_path=dataset_path)
    elif model_type == "LLM":
        model = models.LLM()
        train_gpu(model, dataset_path=dataset_path)
    elif model_type == "BAD WORDS":
        model = models.BadWords()
        train(model, dataset_path=dataset_path)
    else:
        raise ValueError("Unknown model type")


@stub.function(shared_volumes={config.VOLUME_DIR: volume}, interactive=True, timeout=10000)
def inspect():
    model = models.LLM()

    model_id = "sha256.2F4C9AFED1C5A6404DCB5E28FCE2EB7F8F8A4A3B22A8E3BF677071EB42FB0AF3"
    classifier = model.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )

    pred = classifier("I am a fake email this should be classifer as spam or ham.")
    print(pred)


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def init_volume():
    config.MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    with stub.run():
        # init_volume()
        # main()
        inspect()
