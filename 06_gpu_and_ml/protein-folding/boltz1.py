# # Fold proteins with Boltz-1

# Boltz-1 is an open source molecular structure prediction model that matches the performance of closed source models like AlphaFold 3.
# It was created by the [MIT Jameel Clinic](https://jclinic.mit.edu/boltz-1/).
# For details, see [their technical report](https://gcorso.github.io/assets/boltz1.pdf).

# Here, we demonstrate how to run Boltz-1 on Modal.

# ## Setup

from dataclasses import dataclass
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
# run it, and then save the results locally as a [tarball](https://computing.help.inf.ed.ac.uk/FAQ/whats-tarball-or-how-do-i-unpack-or-create-tgz-or-targz-file).
# That tarball archive contains, among other things, the predicted structure as a
# [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File),
# which you can render with the online [Molstar Viewer](https://molstar.org/viewer).

# You can pass any options for the [`boltz predict` command line tool](https://github.com/jwohlwend/boltz/blob/2355c62c957e95305527290112e9742d0565c458/docs/prediction.md)
# as a string, like

# ``` shell
# modal run boltz1 --args "--sampling_steps 10"
# ```

# To see more options, run the command with the `--help` flag.

# To learn how it works, read on!


@app.local_entrypoint()
def main(
    force_download: bool = False, input_yaml_path: str = None, args: str = ""
):
    print("🧬 loading model remotely")
    download_model.remote(force_download)

    if input_yaml_path is None:
        input_yaml_path = here / "data" / "boltz1_ligand.yaml"
    input_yaml = input_yaml_path.read_text()

    msas = find_msas(input_yaml_path)

    print(f"🧬 running boltz with input from {input_yaml_path}")
    output = boltz1_inference.remote(input_yaml, msas)

    output_path = Path("/tmp") / "boltz1" / "boltz1_result.tar.gz"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🧬 writing output to {output_path}")
    output_path.write_bytes(output)


# ## Installing Boltz-1 Python dependencies on Modal

# Code running on Modal runs inside containers built from [container images](https://modal.com/docs/guide/images)
# that include that code's dependencies.

# Because Modal images include [GPU drivers](https://modal.com/docs/guide/cuda) by default,
# installation of higher-level packages like `boltz` that require GPUs is painless.

# Here, we do it in a few lines, using the `uv` package manager for extra speed.

image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "uv pip install --system --compile-bytecode boltz==0.3.2"
)

# ## Storing Boltz-1 model weights on Modal with Volumes

# Not all "dependencies" belong in a container image. Boltz-1, for example, depends on
# the weights of the model and a [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD) file.

# Rather than loading them dynamically at run-time (which would add several minutes of GPU time to each inference),
# or installing them into the image (which would require they be re-downloaded any time the other dependencies changed),
# we load them onto a [Modal Volume](https://modal.com/docs/guide/volumes).
# A Modal Volume is a file system that all of your code running on Modal (or elsewhere!) can access.
# For more on storing model weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).
# For details on how we download the weights in this case, see the [Addenda](#addenda).

boltz_model_volume = modal.Volume.from_name(
    "boltz1-models", create_if_missing=True
)
models_dir = Path("/models/boltz1")

# ## Running Boltz-1 on Modal

# To run inference on Modal we wrap our function in a decorator, `@app.function`.
# We provide that decorator with some arguments that describe the infrastructure our code needs to run:
# the Volume we created, the Image we defined, and of course a fast GPU!

# Note that the `boltz` command-line tool we use takes the path to a
# [specially-formatted YAML file](https://github.com/jwohlwend/boltz/blob/2355c62c957e95305527290112e9742d0565c458/docs/prediction.md)
# that includes definitions of molecules to predict the structures of and optionally paths to
# [Multiple Sequence Alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSA) files
# for any protein molecules. See the [Addenda](#addenda) for details.


@app.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="H100",
)
def boltz1_inference(
    boltz_input_yaml: str, msas: list["MSA"], args=""
) -> bytes:
    import shlex
    import subprocess

    input_path = Path("input.yaml")
    input_path.write_text(boltz_input_yaml)

    for msa in msas:
        msa.path.write_text(msa.data)

    args = shlex.split(args)

    print(f"🧬 predicting structure using boltz model from {models_dir}")
    subprocess.run(
        ["boltz", "predict", input_path, "--cache", str(models_dir)] + args,
        check=True,
    )

    print("🧬 packaging up outputs")
    output_bytes = package_outputs(
        f"boltz_results_{input_path.with_suffix('').name}"
    )

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
    revision: str = "7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1",
):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-1",
        revision=revision,
        local_dir=models_dir,
        force_download=force_download,
    )
    boltz_model_volume.commit()

    print(f"🧬 model downloaded to {models_dir}")


# Additionally, the YAML format accepted by the `boltz predict` command
# includes the option to specify the sequence alignments for any input
# `protein` via a path to an MSA file (in the "aligned-FASTA" format,
# [`.a3m`](https://yanglab.qd.sdu.edu.cn/trRosetta/msa_format.html)).

# To ensure these files are available to the Modal Function running remotely,
# we parse the YAML file and extract the paths to and data from the MSA files.


@dataclass
class MSA:
    data: str
    path: Path


def find_msas(boltz_yaml_path: Path) -> list[MSA]:
    """Finds the MSA data in a YAML file in the Boltz input format.

    See https://github.com/jwohlwend/boltz/blob/2355c62c957e95305527290112e9742d0565c458/docs/prediction.md for details."""
    import yaml

    data = yaml.safe_load(boltz_yaml_path.read_text())
    data_dir = boltz_yaml_path.parent

    sequences = data["sequences"]
    msas = []
    for sequence in sequences:
        if protein := sequence.get("protein"):
            if msa_path := protein.get("msa"):
                if msa_path == "empty":  # special value
                    continue
                if not msa_path.startswith("."):
                    raise ValueError(
                        f"Must specify MSA paths relative to the input yaml path, but got {msa_path}"
                    )
                msa_data = (data_dir / Path(msa_path).name).read_text()
                msas.append(MSA(msa_data, Path(msa_path)))
    return msas


def package_outputs(output_dir: str) -> bytes:
    import io
    import tarfile

    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(output_dir, arcname=output_dir)

    return tar_buffer.getvalue()
