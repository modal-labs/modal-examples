# ---
# integration-test: false
# ---
import subprocess

import os
import modal

LOCAL_CONFIG_PATH = "./algolia_helpers/config.json"

algolia_image = modal.DockerhubImage(
    tag="algolia/docsearch-scraper",
    # This image has both 3.6 and 3.7 installed, but 3.6 is aliased to `python`.
    # TODO: Modal doesn't let you set the Python executable to be used yet.
    setup_commands=["ln -sfn /usr/bin/python3.7 /usr/bin/python"],
)

stub = modal.Stub( "algolia-indexer", image=algolia_image)


@stub.function(secrets=[modal.ref("algolia-secret")])
def crawl(config):
    # Installed with a 3.6 venv; Python 3.6 is unsupported by Modal, so use a subprocess instead.
    subprocess.run(["pipenv", "run", "python", "-m", "src.index"], env={**os.environ, "CONFIG": config})

if __name__ == "__main__":
    with stub.run():
        with open(LOCAL_CONFIG_PATH) as f:
            config = f.read()

        crawl(config)
