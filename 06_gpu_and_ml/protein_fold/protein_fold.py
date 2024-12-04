# ---
# output-directory: "/tmp/protein-fold"
# ---

# # Protein Fold sequences with ESM3

# If you haven't heard of [AlphaFold](https://deepmind.google/technologies/alphafold/)
# or [ESM3](https://github.com/facebookresearch/esm), you've likely been living under a
# rock, it's time to come out! Protein folding is all the rage in the world of
# bioinformatics and ML. In this example, we'll show you how to
# use ESM3 to predict the three-dimensional structure of a protein from its raw
# amino acid sequence. As a bonus, we'll also show you how to visualize the results using
# [py3DMol](https://3dmol.org/).

# If you're new to protein folding check out the next section for a brief
# overview, otherwise you can skip to Basic Setup.

# ## What is Protein Folding?

# A protein's three-dimensional shape determines it's function in the body,
# i.e. everything from catalyzing biochemical reactions to building cellular
# structures. To develop new drugs and treatments for diseases, researchers
# need to understand how potential drugs bind to target proteins in the human
# body and thus need to predict their three-dimensional structure from their
# raw amino acid sequence.

# Historically determining the protein structure of a single sequence could
# take years of wet-lab experiments and millions of dollars, but with the advent
# of deep protein folding models like Meta's ESM3, a quality approximation can be
# done in seconds for a few cents on Modal.

# The bioinformatics community has created a number of resources for managing
# protein data, including the [Research Collaboratory for Structural Bioinformatics](https://www.rcsb.org/) (RCSB) which stores the known structure of over 200,000 proteins.
# These sets of proteins structures are known as Protein Data Banks (PDB).
# We'll also be using the open source [Biotite](https://www.biotite-python.org/)
# library to help us extract sequences and residues from PDBs. Biotite was
# created by Patrick Kunzmann and members of the Protein Bioinformatics lab
# at Rub University Bochum in Germany.

# ## Basic Setup

import logging as L
from pathlib import Path

import modal

MINUTES = 60  # seconds

app_name = "protein_fold"
app = modal.App(app_name)

# We'll use A10G GPUs for inference, which are able to generate 3D structures
# in seconds for a fews cents.

gpu = "A10G"

# ### Create a Volume to store cached pdb data

# To minimize htting the RCSB server for pdb data we'll cache pdb data on a modal
# [Volume](https://modal.com/docs/guide/volumes) so we never have to read the
# same data twice.


volume = modal.Volume.from_name("example-protein-fold", create_if_missing=True)
VOLUME_PATH = Path("/vol/data")
PDBS_PATH = VOLUME_PATH / "pdbs"


# ### Define dependencies in container images

# The container image for inference is based on Modal's default slim Debian
# Linux image with `esm` for loading and running the model, and `hf_transfer`
# for a higher bandwidth download of the model weights from hugging face.


esm3_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "esm==3.0.5",
        "torch==2.4.1",
        "huggingface_hub[hf_transfer]==0.26.2",
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# We'll also define a web app image as a frontend for the entire system that
# includes `gradio` for building a UI, `biotite` for extracting sequences and
# residues from PDBs, and `py3Dmol` for visualizing the 3D structures.


web_app_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "esm==3.0.5",
        "gradio~=4.44.0",
        "biotite==0.41.2",
        "pydssp==0.9.0",
        "py3Dmol==2.4.0",
        "torch==2.4.1",
        "fastapi[standard]==0.115.4",
    )
)


# Here we "pre-import" libraries that will be used by the functions we run
# on Modal in a given image using the `with image.imports` context manager.


with esm3_image.imports():
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


with web_app_image.imports():
    import os

    import biotite.database.rcsb as rcsb
    import biotite.structure as b_structure
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import CIFFile

    from esm.sdk.api import ESMProtein

    from py3DmolWrapper import py3DMolViewWrapper

# ## Defining a `Model` inference class for ESM3

# Next, we map the model's setup and inference code onto Modal.

# 1. To ensure the cached image includes the model weights, we download them in a
# method decorated with `@build`, this reduces cold boot times.
# 2. For any additional setup code that involves CPU or GPU RAM we put in a
# method deocrated with `@enter` which runs on container start.
# 3. To run the actual inference, we put it in a method decorated with `@method`


@app.cls(
    image=esm3_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=gpu,
    timeout=20 * MINUTES,
)
class Model:
    def setup_model(self):
        self.model: ESM3InferenceClient = ESM3.from_pretrained(
            "esm3_sm_open_v1"
        )

    @modal.build()
    def build(self):
        self.setup_model()

    @modal.enter()
    def enter(self):
        self.setup_model()
        self.model.to("cuda")

        # Enable half precision for faster inference
        self.model = self.model.half()
        torch.backends.cuda.matmul.allow_tf32 = True

        self.max_steps = 250
        L.info(f"Setting max ESM steps to: {self.max_steps}")

    @modal.method()
    def inference(self, sequence: str) -> bool:
        num_steps = min(len(sequence), self.max_steps)
        structure_generation_config = GenerationConfig(
            track="structure",
            num_steps=num_steps,
        )
        L.info("Running ESM3 inference with num_steps={num_steps}")
        esm_protein = self.model.generate(
            ESMProtein(sequence=sequence), structure_generation_config
        )

        L.info("Checking for errors...")
        if hasattr(esm_protein, "error_msg"):
            raise ValueError(esm_protein.error_msg)

        L.info("Moving all data off GPU before returning to caller...")
        esm_protein.ptm = esm_protein.ptm.to("cpu")
        return esm_protein


# ### Serving a Gradio UI with an `asgi_app`

# The `ModelInference` class above is available for use
# from any other Python environment with the right Modal credentials
# and the `modal` package installed -- just use [`lookup`](https://modal.com/docs/reference/modal.Cls#lookup).

# But we can also expose it via a web app using the `@asgi_app` decorator. Here
# we will specifically create a Gradio web app that allows us to visualize
# the 3D structure output of the ESM3 model as well as any known structures from
# the RCSB Protein Data Bank. In addition, to understand the confidence level
# of the ESM3 output, we'll include a visualization of the [pLDDT](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/) (predicted Local Distance Difference Test) scores
# for each residue (amino acid) in the folded protein structure.
#

# You should see the URL for this UI in the output of `modal deploy`
# or on your [Modal app dashboard](https://modal.com/apps) for this app.

assets_path = Path(__file__).parent / "assets"

@app.function(
    image=web_app_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    volumes={VOLUME_PATH: volume},
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def protein_fold_fastapi_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    width, height = 400, 400

    def extract_data(esm_protein):
        residue_pLDDTs = (100 * esm_protein.plddt).tolist()
        atoms = esm_protein.to_protein_chain().atom_array
        residue_id_to_sse = extract_residues(atoms)
        return residue_pLDDTs, residue_id_to_sse

    def run_esm(sequence, maybe_stale_pdb_id):
        L.info("Removing whitespace from text input")
        sequence = sequence.strip()

        L.info("Running ESM")
        esm_protein: ESMProtein = Model().inference.remote(sequence)
        residue_pLDDTs, residue_id_to_sse = extract_data(esm_protein)

        L.info("Constructing HTML for ESM3 prediction with residues.")
        esm_sse_html = py3DMolViewWrapper().build_html_with_secondary_structure(
            width, height, esm_protein.to_pdb_string(), residue_id_to_sse
        )

        L.info("Constructing HTML for ESM3 prediction with confidence.")
        esm_pLDDT_html = py3DMolViewWrapper().build_html_with_pLDDTs(
            width, height, esm_protein.to_pdb_string(), residue_pLDDTs
        )

        maybe_stale_pdb_id = maybe_stale_pdb_id.strip()
        maybe_stale_sequence = get_sequence(maybe_stale_pdb_id)
        if maybe_stale_sequence == sequence:
            pdb_id = maybe_stale_pdb_id
            L.info(f"Constructing HTML for RCSB entry for {pdb_id}")
            pdb_string, residue_id_to_sse = extract_pdb_and_residues(pdb_id)
            rcsb_sse_html = (
                py3DMolViewWrapper().build_html_with_secondary_structure(
                    width, height, pdb_string, residue_id_to_sse
                )
            )
        else:
            L.info("Sequence structure is unknown, generating HTML as such.")
            rcsb_sse_html = "<h3>Folding Structure of Sequence not found.</h3>"

        return [
            postprocess_html(h)
            for h in (esm_sse_html, esm_pLDDT_html, rcsb_sse_html)
        ]

    web_app = FastAPI()

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    example_pdbs = [
        "Myoglobin [1MBO]",
        "Insulin [1ZNI]",
        "Hemoglobin [1GZX]",
        "GFP [1EMA]",
        "Collagen [1CGD]",
        "Antibody / Immunoglobulin G [1IGT]",
        "Actin [1ATN]",
        "Ribonuclease A [5RSA]",
    ]

    # Number the examples.
    example_pdbs = [f"{i+1}) {x}" for i, x in enumerate(example_pdbs)]

    with gr.Blocks(
        theme=theme, css=css, title="ESM3 Protein Folding"
    ) as interface:
        gr.Markdown("# Fold Proteins using ESM3 Fold")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Custom PDB ID")
                pdb_id_box = gr.Textbox(
                    label="Enter PDB ID or select one on the right",
                    placeholder="e.g. '5JQ4', '1MBO', '1TPO', etc.",
                )
                get_sequence_button = gr.Button(
                    "Retrieve Sequence from PDB ID", variant="primary"
                )

                pdb_link_button = gr.Button(value="Open PDB page for ID")
                rcsb_link = "https://www.rcsb.org/structure/"
                pdb_link_button.click(
                    fn=None,
                    inputs=pdb_id_box,
                    js=f"""(pdb_id) => {{ window.open("{rcsb_link}" + pdb_id) }}""",
                )

            with gr.Column():
                def extract_pdb_id(example_idx):
                    pdb = example_pdbs[example_idx]
                    return pdb[pdb.index("[") + 1 : pdb.index("]")]

                half_len = int(len(example_pdbs) / 2)

                gr.Markdown("## Example PDB IDs")
                with gr.Row():
                    with gr.Column():
                        for i, pdb in enumerate(example_pdbs[:half_len]):
                            btn = gr.Button(pdb, variant="secondary")
                            btn.click(
                                fn=lambda j=i: extract_pdb_id(j),
                                outputs=pdb_id_box,
                            )

                    with gr.Column():
                        for i, pdb in enumerate(example_pdbs[half_len:]):
                            btn = gr.Button(pdb, variant="secondary")
                            btn.click(
                                fn=lambda j=i + half_len: extract_pdb_id(j),
                                outputs=pdb_id_box,
                            )

        gr.Markdown("## Sequence")
        sequence_box = gr.Textbox(
            label="Enter a sequence or retrieve it from a PDB ID",
            placeholder="e.g. 'MVTRLE...', 'GKQEG...', etc.",
        )
        run_esm_button = gr.Button(
            "Run ESM3 Fold on Sequence", variant="primary"
        )

        htmls = []
        legend_height = 100
        with gr.Row():
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - Secondary Structs")
                gr.Image(  # output image component
                    height=legend_height,
                    width=width,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False,
                    show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 2: ESM Prediction with Confidence coloring.
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - PLTT Confidence")
                gr.Image(  # output image component
                    height=legend_height,
                    width=width,
                    value="/assets/plddtLegend2.png",
                    show_download_button=False,
                    show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 3: Crystalized form of Protein if available.
            with gr.Column():
                gr.Markdown("## Crystalized Structure from RCSB's PDB")
                gr.Image(  # output image component
                    height=legend_height,
                    width=width,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False,
                    show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

        get_sequence_button.click(
            fn=get_sequence, inputs=[pdb_id_box], outputs=[sequence_box]
        )
        run_esm_button.click(
            fn=run_esm, inputs=[sequence_box, pdb_id_box], outputs=htmls
        )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

# ## Addenda

# The remainder of this code is boilerplate.


# ### Extracting Sequences and Residues from PDBs

# A fair amount of code is required to extract sequences and residue
# information from pdb strings, we build several helper functions for it below.

# There is more complex code for running py3Dmol which you can find in
# the `py3DmolWrapper.py` file.


def fetch_pdb_if_necessary(pdb_id, pdb_type):
    """Fetch PDB from RCSB if not already cached."""

    assert pdb_type in ("pdb", "pdbx")

    file_path = PDBS_PATH / f"{pdb_id}.{pdb_type}"
    if not os.path.exists(file_path):
        L.info(f"Loading PDB {file_path} from server...")
        rcsb.fetch(pdb_id, pdb_type, str(PDBS_PATH))
    return file_path


def get_sequence(pdb_id):
    try:
        pdb_id = pdb_id.strip()
        pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")

        structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)

        amino_sequences, _ = b_structure.to_sequence(
            structure[b_structure.filter_amino_acids(structure)]
        )

        sequence = "".join([str(s) for s in amino_sequences])
        return sequence

    except Exception as e:
        return f"Error: {e}"


def extract_residues(atoms):
    residue_secondary_structures = b_structure.annotate_sse(atoms)
    residue_ids = b_structure.get_residues(atoms)[0]
    residue_id_to_sse = {}
    for residue_id, sse in zip(residue_ids, residue_secondary_structures):
        residue_id_to_sse[int(residue_id)] = sse
    return residue_id_to_sse


def extract_pdb_and_residues(pdb_id):
    pdb_file_path = fetch_pdb_if_necessary(pdb_id, "pdb")
    pdb_string = Path(pdb_file_path).read_text()

    pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")
    structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)
    atoms = structure[b_structure.filter_amino_acids(structure)]
    residue_id_to_sse = extract_residues(atoms)

    return pdb_string, residue_id_to_sse

# ### Miscellaneous
# The remaining code includes small helper functions for cleaning up HTML
# generated by py3DMol for viewing on a webpage.

def remove_nested_quotes(html):
    """Remove triple nested quotes in HEADER"""
    if html.find("HEADER") == -1:
        return html

    i = html.index("HEADER")
    j = html[i:].index('"pdb");') + i - len('",')
    header_html = html[i:j]
    for delete_me in ("\\'", '\\"', "'", '"'):
        header_html = header_html.replace(delete_me, "")
    html = html[:i] + header_html + html[j:]
    return html


def postprocess_html(html):
    html = remove_nested_quotes(html)
    html = html.replace("'", '"')

    L.info("Wrapping py3DMol HTML in iframe so we can view multiple HTMLs.")
    html_wrapped = f"""<!DOCTYPE html><html>{html}</html>"""
    iframe_html = (
        f"""<iframe style="width: 100%; height: 400px;" """
        f"""allow="midi; display-capture;" frameborder="0" """
        f"""srcdoc='{html_wrapped}'></iframe>"""
    )
    return iframe_html
