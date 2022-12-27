import pathlib
from datetime import timedelta

from . import config
from .datasets.enron import download
from .app import stub, volume


@stub.function(
    timeout=int(timedelta(minutes=5).total_seconds()),
    shared_volumes={config.VOLUME_DIR: volume},
)
def prep():
    datasets_path = config.DATA_DIR
    datasets_path.mkdir(parents=True, exist_ok=True)
    download.download(base=datasets_path)


if __name__ == "__main__":
    with stub.run():
        prep()
