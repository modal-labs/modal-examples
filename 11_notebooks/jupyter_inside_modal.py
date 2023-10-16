# ---
# args: ["--timeout", 10]
# ---
#
# Quick snippet to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal network file systems.
# This uses https://github.com/ekzhang/bore to expose the server to the public internet.

import os
import subprocess
import time

import modal

stub = modal.Stub(
    image=modal.Image.debian_slim()
    .pip_install("jupyter", "bing-image-downloader~=1.1.2")
    .apt_install("curl")
    .run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y")
    .run_commands(". $HOME/.cargo/env && cargo install bore-cli")
)
# This volume is not persisted, so the data will be deleted when this demo app is stopped.
volume = modal.NetworkFileSystem.new()

CACHE_DIR = "/root/cache"
JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


@stub.function(
    network_file_systems={CACHE_DIR: volume},
)
def seed_volume():
    # Bing it!
    from bing_image_downloader import downloader

    # This will save into the Modal volume and allow you view the images
    # from within Jupyter at a path like `/cache/modal labs/Image_1.png`.
    downloader.download(
        query="modal labs",
        limit=10,
        output_dir=CACHE_DIR,
        force_replace=False,
        timeout=60,
        verbose=True,
    )


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with network file system contents
# without having to download it to your host computer.


@stub.function(
    concurrency_limit=1, network_file_systems={CACHE_DIR: volume}, timeout=1_500
)
def run_jupyter(timeout: int):
    jupyter_process = subprocess.Popen(
        [
            "jupyter",
            "notebook",
            "--no-browser",
            "--allow-root",
            "--port=8888",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
        ],
        env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
    )

    bore_process = subprocess.Popen(
        ["/root/.cargo/bin/bore", "local", "8888", "--to", "bore.pub"],
    )

    try:
        end_time = time.time() + timeout
        while time.time() < end_time:
            time.sleep(5)
        print(f"Reached end of {timeout} second timeout period. Exiting...")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        bore_process.kill()
        jupyter_process.kill()


@stub.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    seed_volume.remote()
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like http://bore.pub:$PORT/. Visit this address
# in your browser, and enter the security token you set for `JUPYTER_TOKEN`.
