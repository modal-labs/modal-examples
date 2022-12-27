import pathlib
from datetime import timedelta

import modal

from . import config
from . import dataset
from . import models
from .app import stub, volume


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def init_volume():
    config.MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
)
def train(model: models.SpamModel, dataset_path: pathlib.Path):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    classifier = model.train(enron_dataset)
    model_id = model.save(fn=classifier, model_registry_root=config.MODEL_STORE_DIR)
    logger.info(f"saved model to model store. {model_id=}")
    # Reload the model
    logger.info(f"🔁 testing reload of model")
    classifier = model.load(
        sha256_digest=model_id,
        model_registry_root=config.MODEL_STORE_DIR,
    )
    is_spam = classifier("fake email!")
    print(f"classification: {is_spam=}")


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
    # NOTE: Can't use A100 easily because:
    # "Modal SharedVolume data will not be shared between A100 and non-A100 functions"
    gpu=modal.gpu.T4(),
)
def train_gpu(model: models.SpamModel, dataset_path: pathlib.Path):
    logger = config.get_logger()
    enron_dataset = dataset.deserialize_dataset(dataset_path)
    classifier = model.train(enron_dataset)
    model_id = model.save(fn=classifier, model_registry_root=config.MODEL_STORE_DIR)
    logger.info(f"saved model to model store. {model_id=}")


@stub.function(
    shared_volumes={config.VOLUME_DIR: volume},
    secrets=[modal.Secret({"PYTHONHASHSEED": "10"})],
    timeout=int(timedelta(minutes=30).total_seconds()),
)
def main(model_type: str = str(config.ModelTypes.LLM)):
    logger = config.get_logger()
    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )
    dataset_path = dataset.dataset_path(config.DATA_DIR)

    logger.info("💪 training ...")
    model: models.SpamModel
    if model_type == config.ModelTypes.NAIVE_BAYES:
        model = models.NaiveBayes()
        train.call(model, dataset_path=dataset_path)
    elif model_type == config.ModelTypes.LLM:
        model = models.LLM()
        train_gpu.call(model, dataset_path=dataset_path)
    elif model_type == config.ModelTypes.BAD_WORDS:
        model = models.BadWords()
        train.call(model, dataset_path=dataset_path)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")


if __name__ == "__main__":
    with stub.run():
        init_volume.call()
        main.call()
