# # Fold proteins with Chai-1

# In biology, function follows form quite literally:
# the physical shapes of proteins dictate their behavior.
# Measuring those shapes directly is difficult
# and first-principles physical simulation prohibitively expensive.

# And so predicting protein shape from content --
# determining how the one-dimensional chain of amino acids encoded by DNA _folds_ into a 3D object --
# has emerged as a key application for machine learning and neural networks in biology.

# In this example, we demonstrate how to run the open source [Chai-1](https://github.com/chaidiscovery/chai-lab/)
# protein structure prediction model on Modal's flexible serverless infrastructure.
# For details on how the Chai-1 model works and what it can be used for,
# see the authors' [technical report on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.10.615955).

# This simple script is meant as a starting point showing how to handle fiddly bits
# like installing dependencies, loading weights, and formatting outputs so that you can get on with the fun stuff.
# To experience the full power of Modal, try scaling inference up and running on hundreds or thousands of structures!

# <center>
# <a href="https://molstar.org/viewer"> <video controls autoplay loop muted> <source src="https://modal-cdn.com/example-chai1-folding.mp4" type="video/mp4"> </video> </a>
# </center>

# ## Setup

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

app = modal.App(name="example-chai1-inference")

# ## Fold a protein from the command line

# The logic for running Chai-1 is encapsulated in the function below,
# which you can trigger from the command line by running

# ```shell
# modal run chai1
# ```

# This will set up the environment for running Chai-1 inference in Modal's cloud,
# run it, and then save the results remotely and locally. The results are returned in the
# [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File) format,
# which you can render with the online [Molstar Viewer](https://molstar.org/).

# To see more options, run the command with the `--help` flag.

# To learn how it works, read on!


@app.local_entrypoint()
def main(
    force_redownload: bool = False,
    fasta_file: str = None,
    inference_config_file: str = None,
    output_dir: str = None,
    run_id: str = None,
):
    print("🧬 checking inference dependencies")
    download_inference_dependencies.remote(force=force_redownload)

    if fasta_file is None:
        fasta_file = here / "data" / "chai1_default_input.fasta"
    print(f"🧬 running Chai inference on {fasta_file}")
    fasta_content = Path(fasta_file).read_text()

    if inference_config_file is None:
        inference_config_file = here / "data" / "chai1_default_inference.json"
    print(f"🧬 loading Chai inference config from {inference_config_file}")
    inference_config = json.loads(Path(inference_config_file).read_text())

    if run_id is None:
        run_id = hashlib.sha256(uuid4().bytes).hexdigest()[:8]  # short id
    print(f"🧬 running inference with {run_id=}")

    results = chai1_inference.remote(fasta_content, inference_config, run_id)

    if output_dir is None:
        output_dir = Path("/tmp/chai1")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧬 saving results to disk locally in {output_dir}")
    for ii, (scores, cif) in enumerate(results):
        (Path(output_dir) / f"{run_id}-scores.model_idx_{ii}.npz").write_bytes(
            scores
        )
        (Path(output_dir) / f"{run_id}-preds.model_idx_{ii}.cif").write_text(
            cif
        )


# ## Installing Chai-1 Python dependencies on Modal

# Code running on Modal runs inside containers built from [container images](https://modal.com/docs/guide/images)
# that include that code's dependencies.

# Because Modal images include [GPU drivers](https://modal.com/docs/guide/cuda) by default,
# installation of higher-level packages like `chai_lab` that require GPUs is painless.

# Here, we do it with one line, using the `uv` package manager for extra speed.

image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "uv pip install --system --compile-bytecode chai_lab==0.5.0 hf_transfer==0.1.8"
)

# ## Storing Chai-1 model weights on Modal with Volumes

# Not all "dependencies" belong in a container image. Chai-1, for example, depends on
# the weights of several models.

# Rather than loading them dynamically at run-time (which would add several minutes of GPU time to each inference),
# or installing them into the image (which would require they be re-downloaded any time the other dependencies changed),
# we load them onto a [Modal Volume](https://modal.com/docs/guide/volumes).
# A Modal Volume is a file system that all of your code running on Modal (or elsewhere!) can access.
# For more on storing model weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

chai_model_volume = (
    modal.Volume.from_name(  # create distributed filesystem for model weights
        "chai1-models",
        create_if_missing=True,
    )
)
models_dir = Path("/models/chai1")

# The details of how we handle the download here (e.g. running concurrently for extra speed)
# are in the [Addenda](#addenda).

image = image.env(  # update the environment variables in the image to...
    {
        "CHAI_DOWNLOADS_DIR": str(models_dir),  # point the chai code to it
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # speed up downloads
    }
)

# ## Storing Chai-1 outputs on Modal Volumes

# Chai-1 produces its outputs by writing to disk --
# the model's scores for the structure and the structure itself along with rich metadata.

# But Modal is a _serverless_ platform, and the filesystem your Modal Functions write to
# is not persistent. Any file can be converted into bytes and sent back from a Modal Function
# -- and we mean any! You can send files that are gigabytes in size that way.
# So we do that below.

# But for larger jobs, like folding every protein in the PDB, storing bytes on a local client
# like a laptop won't cut it.

# So we again lean on Modal Volumes, which can store thousands of files each.
# We attach a Volume to a Modal Function that runs Chai-1 and the inference code
# saves the results to distributed storage, without any fuss or source code changes.

chai_preds_volume = modal.Volume.from_name(
    "chai1-preds", create_if_missing=True
)
preds_dir = Path("/preds")

# ## Running Chai-1 on Modal

# Now we're ready to define a Modal Function that runs Chai-1.

# We put our function on Modal by wrapping it in a decorator, `@app.function`.
# We provide that decorator with some arguments that describe the infrastructure our code needs to run:
# the Volumes we created, the Image we defined, and of course a fast GPU!

# Note that Chai-1 takes a file path as input --
# specifically, a path to a file in the [FASTA format](https://en.wikipedia.org/wiki/FASTA_format).
# We pass the file contents to the function as a string and save them to disk so they can be picked up by the inference code.

# Because Modal is serverless, we don't need to worry about cleaning up these resources:
# the disk is ephemeral and the GPU only costs you money when you're using it.


@app.function(
    timeout=15 * MINUTES,
    gpu="H100",
    volumes={models_dir: chai_model_volume, preds_dir: chai_preds_volume},
    image=image,
)
def chai1_inference(
    fasta_content: str, inference_config: dict, run_id: str
) -> list[(bytes, str)]:
    from pathlib import Path

    import torch
    from chai_lab import chai1

    N_DIFFUSION_SAMPLES = 5  # hard-coded in chai-1

    fasta_file = Path("/tmp/inputs.fasta")
    fasta_file.write_text(fasta_content.strip())

    output_dir = Path("/preds") / run_id

    chai1.run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir,
        device=torch.device("cuda"),
        **inference_config,
    )

    print(
        f"🧬 done, results written to /{output_dir.relative_to('/preds')} on remote volume"
    )

    results = []
    for ii in range(N_DIFFUSION_SAMPLES):
        scores = (output_dir / f"scores.model_idx_{ii}.npz").read_bytes()
        cif = (output_dir / f"pred.model_idx_{ii}.cif").read_text()

        results.append((scores, cif))

    return results


# ## Addenda

# Above, we glossed over just how we got hold of the model weights --
# the `local_entrypoint` just called a function named `download_inference_dependencies`.

# Here's that function's implementation.

# A few highlights:

# - This Modal Function can access the model weights Volume, like the inference Function,
# but it can't access the model predictions Volume.

# - This Modal Function has a different Image and doesn't use a GPU. Modal helps you
# separate the concerns, and the costs, of your infrastructure's components.

# - We use the `async` keyword here so that we can run the download for each model file
# as a separate task, concurrently. We don't need to worry about this use of `async`
# spreading to the rest of our code -- Modal launches just this Function in an async runtime.


@app.function(
    volumes={models_dir: chai_model_volume},
    image=modal.Image.debian_slim().pip_install("requests"),
)
async def download_inference_dependencies(force=False):
    import asyncio

    import aiohttp

    base_url = "https://chaiassets.com/chai1-inference-depencencies/"  # sic
    inference_dependencies = [
        "conformers_v1.apkl",
        "models_v2/trunk.pt",
        "models_v2/token_embedder.pt",
        "models_v2/feature_embedding.pt",
        "models_v2/diffusion_module.pt",
        "models_v2/confidence_head.pt",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # launch downloads concurrently
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for dep in inference_dependencies:
            if not force:
                print(f"🧬 checking {dep}")
            local_path = models_dir / dep
            if force or not local_path.exists():
                url = base_url + dep
                print(f"🧬 downloading {dep}")
                tasks.append(download_file(session, url, local_path))
            else:
                print("🧬 found, skipping")

        # run all of the downloads and await their completion
        await asyncio.gather(*tasks)

    chai_model_volume.commit()  # ensures models are visible on remote filesystem before exiting, otherwise takes a few seconds, racing with inference


async def download_file(session, url: str, local_path: Path):
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while chunk := await response.content.read(8192):
                f.write(chunk)
