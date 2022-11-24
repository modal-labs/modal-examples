import pathlib
from datetime import timedelta

from . import config
from .datasets.enron import download
from .main import stub, volume


@stub.function(
    timeout=int(timedelta(minutes=5).total_seconds()),
    shared_volumes={config.VOLUME_DIR: volume},
)
def prep():
    dataset_path = pathlib.Path(config.VOLUME_DIR, "enron")
    dataset_path.mkdir(parents=True, exist_ok=True)
    download.download(destination=dataset_path)


if __name__ == "__main__":
    with stub.run():
        prep()
