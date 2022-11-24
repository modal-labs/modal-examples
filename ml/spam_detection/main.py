import json
import pathlib
import sys
from datetime import timedelta

import modal

from . import config
from . import models
from .datasets.enron import structure as enron

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    [
        "datasets~=2.7.1",
        "evaluate~=0.3.0",
        "loguru~=0.6.0",
        "scikit-learn~=1.1.3",  # Required by evaluate pkg.
        "torch~=1.13.0",
        "transformers~=4.24.0",
    ]
)
stub = modal.Stub(name="example-spam-detect-llm", image=image)
volume = modal.SharedVolume().persist("example-spam-detect-vol")


# NOTE: Can't use A100 easily because "Modal SharedVolume data will not be shared between A100 and non-A100 functions"
@stub.function(shared_volumes={config.VOLUME_DIR: volume}, timeout=int(timedelta(minutes=30).total_seconds()), gpu=True)
def train():
    logger = config._get_logger()
    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )
    dataset_path = pathlib.Path(
        config.VOLUME_DIR, "enron", "processed_raw_dataset.json"
    )  # TODO: Shouldn't need to hardcode.
    # models.train_llm_classifier(dataset)
    classifier = models.train_naive_bayes_classifier(enron.deserialize_dataset(dataset_path))
    print(classifier)


@stub.function(interactive=True)
def inference(email: str):
    model_path = config.MODEL_STORE_DIR / "tmpmodel"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    breakpoint()


if __name__ == "__main__":
    with stub.run():
        train()
        # inference()
