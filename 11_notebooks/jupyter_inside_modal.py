# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)

import os
import subprocess
import time

import modal

app = modal.App(
    "example-jupyter-inside-modal",
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "jupyter", "bing-image-downloader~=1.1.2"
    ),
)
volume = modal.Volume.from_name(
    "modal-examples-jupyter-inside-modal-data", create_if_missing=True
)

CACHE_DIR = "/root/cache"
JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


@app.function(volumes={CACHE_DIR: volume})
def seed_volume():
    # Bing it!
    from bing_image_downloader import downloader

    # This will save into the Modal volume and allow you view the images
    # from within Jupyter at a path like `/root/cache/modal labs/Image_1.png`.
    downloader.download(
        query="modal labs",
        limit=10,
        output_dir=CACHE_DIR,
        force_replace=False,
        timeout=60,
        verbose=True,
    )
    volume.commit()


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(max_containers=1, volumes={CACHE_DIR: volume}, timeout=1_500)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    seed_volume.remote()
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like https://u35iiiyqp5klbs.r3.modal.host.
# Visit this address in your browser, and enter the security token
# you set for `JUPYTER_TOKEN`.
