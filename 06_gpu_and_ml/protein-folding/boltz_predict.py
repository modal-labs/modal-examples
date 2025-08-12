# # Fold proteins with Boltz-2

# <figure style="width: 70%; margin: 0 auto; display: block;">
# <img src="https://modal-cdn.com/cdnbot/boltz_examplecd5u3m0j_9fa47e43.webp" alt="Boltz-2" />
# <figcaption style="text-align: center"><em>Example of Boltz-2 protein structure prediction
# of a <a style="text-decoration: underline;" href="https://github.com/jwohlwend/boltz/blob/main/examples/affinity.yaml" target="_blank">protein-ligand complex</a></em></figcaption>
# </figure>

# Boltz-2 is an open source molecular structure prediction model.
# In contrast to previous models like Boltz-1, [Chai-1](https://modal.com/docs/examples/chai1), and AlphaFold-3, it not only predicts protein structures but also the [binding affinities](https://en.wikipedia.org/wiki/Ligand_(biochemistry)#Receptor/ligand_binding_affinity) between proteins and [ligands](https://en.wikipedia.org/wiki/Ligand_(biochemistry)).
# It was created by the [MIT Jameel Clinic](https://jclinic.mit.edu/boltz-2/).
# For details, see [their technical report](https://jeremywohlwend.com/assets/boltz2.pdf).

# Here, we demonstrate how to run Boltz-2 on Modal.

# ## Setup

from pathlib import Path
from typing import Optional

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

app = modal.App(name="example-boltz-predict")

# ## Fold a protein from the command line

# The logic for running Boltz-2 is encapsulated in the function below,
# which you can trigger from the command line by running

# ```shell
# modal run boltz_predict.py
# ```

# This will set up the environment for running Boltz-2 inference in Modal's cloud,
# run it, and then save the results locally as a [tarball](https://computing.help.inf.ed.ac.uk/FAQ/whats-tarball-or-how-do-i-unpack-or-create-tgz-or-targz-file).
# That tarball archive contains, among other things, the predicted structure as a
# [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File),
# which you can render with the online [Molstar Viewer](https://molstar.org/viewer).

# You can pass any options for the [`boltz predict` command line tool](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)
# as a string, like

# ``` shell
# modal run boltz_predict.py --args "--sampling_steps 10"
# ```

# To see more options, run the command with the `--help` flag.

# To learn how it works, read on!


@app.local_entrypoint()
def main(
    force_download: bool = False, input_yaml_path: Optional[str] = None, args: str = ""
):
    print("🧬 loading model remotely")
    download_model.remote(force_download)

    if input_yaml_path is None:
        input_yaml_path = here / "data" / "boltz_affinity.yaml"
    input_yaml = input_yaml_path.read_text()

    print(f"🧬 running boltz with input from {input_yaml_path}")
    output = boltz_inference.remote(input_yaml)

    output_path = Path("/tmp") / "boltz" / "boltz_result.tar.gz"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🧬 writing output to {output_path}")
    output_path.write_bytes(output)


# ## Installing Boltz-2 Python dependencies on Modal

# Code running on Modal runs inside containers built from [container images](https://modal.com/docs/guide/images)
# that include that code's dependencies.

# Because Modal images include [GPU drivers](https://modal.com/docs/guide/cuda) by default,
# installation of higher-level packages like `boltz` that require GPUs is painless.

# Here, we do it in a few lines, using the `uv` package manager for extra speed.

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("boltz==2.1.1")

# ## Storing Boltz-2 model weights on Modal with Volumes

# Not all "dependencies" belong in a container image. Boltz-2, for example, depends on
# the weights of the model and a [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD) file.

# Rather than loading them dynamically at run-time (which would add several minutes of GPU time to each inference),
# or installing them into the image (which would require they be re-downloaded any time the other dependencies changed),
# we load them onto a [Modal Volume](https://modal.com/docs/guide/volumes).
# A Modal Volume is a file system that all of your code running on Modal (or elsewhere!) can access.
# For more on storing model weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).
# For details on how we download the weights in this case, see the [Addenda](#addenda).

boltz_model_volume = modal.Volume.from_name("boltz-models", create_if_missing=True)
models_dir = Path("/models/boltz")

# ## Running Boltz-2 on Modal

# To run inference on Modal we wrap our function in a decorator, `@app.function`.
# We provide that decorator with some arguments that describe the infrastructure our code needs to run:
# the Volume we created, the Image we defined, and of course a fast GPU!

# Note that the `boltz` command-line tool we use takes the path to a
# [specially-formatted YAML file](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md#yaml-format)
# that includes definitions of molecules to predict the structures of and optionally paths to
# [Multiple Sequence Alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSA) files
# for any protein molecules. We pass the [--use_msa_server](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) flag to auto-generate the MSA using the mmseqs2 server.


@app.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="H100",
)
def boltz_inference(boltz_input_yaml: str, args="") -> bytes:
    import shlex
    import subprocess

    input_path = Path("input.yaml")
    input_path.write_text(boltz_input_yaml)

    args = shlex.split(args)

    print(f"🧬 predicting structure using boltz model from {models_dir}")
    subprocess.run(
        ["boltz", "predict", input_path, "--use_msa_server", "--cache", str(models_dir)]
        + args,
        check=True,
    )

    print("🧬 packaging up outputs")
    output_bytes = package_outputs(f"boltz_results_{input_path.with_suffix('').name}")

    return output_bytes


# ## Addenda

# Above, we glossed over just how we got hold of the model weights --
# the `local_entrypoint` just called a function named `download_model`.

# Here's the implementation of that function. For details, see our
# [guide to storing model weights on Modal](https://modal.com/docs/guide/model-weights).

download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.26.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
)


@app.function(
    volumes={models_dir: boltz_model_volume},
    timeout=20 * MINUTES,
    image=download_image,
)
def download_model(
    force_download: bool = False,
    revision: str = "6fdef46d763fee7fbb83ca5501ccceff43b85607",
):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=revision,
        local_dir=models_dir,
        force_download=force_download,
    )
    boltz_model_volume.commit()

    print(f"🧬 model downloaded to {models_dir}")


# We package the outputs into a tarball which contains the predicted structure as a
# [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File)
# and the binding affinity as a JSON file.
# You can render the structure with the online [Molstar Viewer](https://molstar.org/viewer).


def package_outputs(output_dir: str) -> bytes:
    import io
    import tarfile

    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(output_dir, arcname=output_dir)

    return tar_buffer.getvalue()
