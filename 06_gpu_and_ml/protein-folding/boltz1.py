# # Fold proteins with Boltz-1

# Boltz-1 is a fully open source (that means training code too!) protein folding
# model that matches the SOTA performance of closed source models like AlphaFold3. It was
# created by the [MIT Jameel Clinic](https://jclinic.mit.edu/boltz-1/), which
# is a research center focused on applying AI to solve big problems in healthcare.

# ## Setup
from pathlib import Path

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

app = modal.App(name="example-boltz1-inference")

# ## Fold a protein from the command line

# The logic for running Boltz-1 is encapsulated in the function below,
# which you can trigger from the command line by running

# ```shell
# modal run boltz1
# ```

# This will set up the environment for running Boltz-1 inference in Modal's cloud,
# run it, and then save the results locally. The results are returned
# are a tarball of many files including a
# [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File),
# which you can render with the online [Molstar Viewer](https://molstar.org/).

# To see more options, run the command with the `--help` flag.

# To learn how it works, read on!


@app.local_entrypoint()
def main(
    force_redownload: bool = False,
):
    print("checking inference dependencies")
    download_inference_dependencies.remote(force=force_redownload)

    msa_path = here / "data" / "boltz1_seq1.a3m"
    msadata = open(msa_path).read()

    ligand_path = here / "data" / "boltz1_ligand.yaml"
    ligandyaml = open(ligand_path).read()

    print(f"running boltz with\n\tLigand: {ligand_path}\n\tMSA: {msa_path}")
    output = boltz1_inference.remote(msadata, ligandyaml)

    output_path = here / "data" / "boltz1_result.tar.gz"
    print(f"locally writing output tar to {output_path}")
    with open(output_path, "wb") as f:
        f.write(output)


# ## Installing Boltz-1 Python dependencies on Modal

# Code running on Modal runs inside containers built from [container images](https://modal.com/docs/guide/images)
# that include that code's dependencies.

# Because Modal images include [GPU drivers](https://modal.com/docs/guide/cuda) by default,
# installation of higher-level packages like `boltz` that require GPUs is painless.

# Here, we do it in a few lines, using the `uv` package manager for extra speed.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "uv pip install --system --compile-bytecode boltz==0.3.2 hf_transfer==0.1.8"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# We also add `os` to the image imports for running the `boltz` binary.

with image.imports():
    import os

# ## Storing Boltz-1 model weights on Modal with Volumes

# Not all "dependencies" belong in a container image. Boltz-1, for example, depends on
# the weights of the model and a [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD) file.

# Rather than loading them dynamically at run-time (which would add several minutes of GPU time to each inference),
# or installing them into the image (which would require they be re-downloaded any time the other dependencies changed),
# we load them onto a [Modal Volume](https://modal.com/docs/guide/volumes).
# A Modal Volume is a file system that all of your code running on Modal (or elsewhere!) can access.
# For more on storing model weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

boltz_model_volume = modal.Volume.from_name(
    "boltz1-models", create_if_missing=True
)
models_dir = Path("/models/boltz1")

# ## Running Boltz-1 on Modal

# To run inference on Modal we wrap our function in a decorator, `@app.function`.
# We provide that decorator with some arguments that describe the infrastructure our code needs to run:
# the Volume we created, the Image we defined, and of course a fast GPU!

# Note that Boltz-1 takes a ligand yaml file path as input that includes
# an amino acid sequence as well as a path to an MSA file. We create these
# files paths in the function below so they can be picked up by the inference code.


@app.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="A100",
)
def boltz1_inference(msadata, ligandyaml):
    ligand_filename = "ligand.yaml"
    temp_filename = "temp.tar.gz"

    Path("./seq1.a3m").write_text(msadata)
    Path(f"./{ligand_filename}").write_text(ligandyaml)

    print(f"predicting using boltz model {models_dir}")
    os.system(f"boltz predict {ligand_filename} --cache={models_dir}")

    print(f"creating tar file {temp_filename} of outputs")
    os.system(f"tar czvf {temp_filename} boltz_results_ligand")

    print(f"converting {temp_filename} to bytes for returning")
    output = open(f"./{temp_filename}", "rb").read()
    return output


# ## Addenda

# Above, we glossed over just how we got hold of the model weights --
# the `local_entrypoint` just called a function named `download_inference_dependencies`.

# Here's that function's implementation.

# A couple highlights:

# - This Modal Function has a different Image and doesn't use a GPU. Modal helps you
# separate the concerns, and the costs, of your infrastructure's components.

# - We use the `async` keyword here so that we can run the download for each model file
# as a separate task, concurrently. We don't need to worry about this use of `async`
# spreading to the rest of our code -- Modal launches just this Function in an async runtime.


@app.function(
    volumes={models_dir: boltz_model_volume},
    timeout=20 * MINUTES,
    image=modal.Image.debian_slim().pip_install("requests"),
)
async def download_inference_dependencies(force=False):
    import asyncio

    import aiohttp

    base_url = "https://huggingface.co/boltz-community/boltz-1/resolve/e01950840c2a2ec881695f26e994a73b417af0b2/"  # sic
    inference_dependencies = [
        "boltz1.ckpt",
        "ccd.pkl",
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

    boltz_model_volume.commit()  # ensures models are visible on remote filesystem before exiting, otherwise takes a few seconds, racing with inference


async def download_file(session, url: str, local_path: Path):
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while chunk := await response.content.read(8192):
                f.write(chunk)
