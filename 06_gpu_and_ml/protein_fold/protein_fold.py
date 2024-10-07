# ---
# output-directory: "/tmp/protein-fold"
# ---

# Guide
# - A protein is a sequence of atoms, some but not all of those atoms may belong
# to an Amino Acid (AA).
# - Residue is the same as Amino Acid.
# - A residue ID corresponds to a specific AA, it is not an index in the
# protein. e.g. residue ID 2 has the name "ILE" which is Isoleucine.
#
# Acronyms
## AA  - Amino Acid
## SSE - Secondary Structure
# Background
## Atom14 refers to a reduced representation of 14 atoms per Amino Acid. Alpha
## Fold and ESMFold seem to run in this representation space for efficiency.
## Atom37 refers to an expanded representation that is computed from Atom14 and
## more closely resembles PDB structure.

# TODO
# - Do we need PDB & PDBX files? Kind of silly but py3DMol doesn't seem to like
# PDB files.
# - Error handling for bad sequence. E.g. non amino acid letter.

from fastapi import FastAPI
from fastapi.responses import FileResponse
import logging as L
import modal
from pathlib import Path
import time

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "esm==3.0.5",
        "torch==2.4.1",
    )
)

web_app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "esm==3.0.5",
        "gradio~=4.44.0",
        "biotite==0.41.2",
        "pydssp==0.9.0",
        "py3Dmol==2.4.0",
        "torch==2.4.1",
    )
)

app = modal.App("protein_fold")

volume = modal.Volume.from_name(
    "example-protein-fold", create_if_missing=True
)
volume_path = Path("/vol/data")
PDBS_PATH = volume_path / "pdbs"
HTMLS_PATH = volume_path / "htmls"

with esm3_image.imports():
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
    import torch

with web_app_image.imports():
    from py3DmolWrapper import py3DMolViewWrapper

SECONDS_PER_MINUTE = 60
@app.cls(
    gpu='A10G',
    image=esm3_image,
    timeout=20 * SECONDS_PER_MINUTE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ModelInference:
    @modal.build()
    @modal.enter()
    def build(self):
        from esm.models.esm3 import ESM3
        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
        # Optimizations from:
        # https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=cc0f0186
        self.model = self.model.half()
        torch.backends.cuda.matmul.allow_tf32 = True

        self.max_steps = 250

    @modal.method()
    def generate(self, sequence: str) -> bool:
        start_time = time.monotonic()
        # num_steps cannot be longer than the seq length in ESM.
        structure_generation_config = GenerationConfig(
            track="structure",
            num_steps=min(len(sequence), self.max_steps)
            )
        esm_protein = self.model.generate(
            ESMProtein(sequence=sequence),
            structure_generation_config
            )
        latency_s = time.monotonic() - start_time
        print (f"Latency {latency_s:.2f} seconds.")

        # Check that esm_protein did not error.
        if hasattr(esm_protein, "error_msg"):
            raise ValueError(esm_protein.error_msg)

        esm_protein.ptm = esm_protein.ptm.to('cpu') # Move off GPU
        return esm_protein

assets_path = Path(__file__).parent / "assets"

web_app = FastAPI()
@app.function(
    image=web_app_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    volumes={volume_path: volume},
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def protein_fold_fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    import biotite.database.rcsb as rcsb
    import biotite.structure as b_structure
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import CIFFile
    import biotite.application.blast as blast
    from esm.sdk.api import ESMProtein

    import os

    width, height = 400, 400

    def fetch_pdb_if_necessary(pdb_id, pdb_type):
        # Only fetch from server if it's not on the volume already.
        assert pdb_type in ("pdb", "pdbx")
        file_path = PDBS_PATH / f"{pdb_id}.{pdb_type}"
        if not os.path.exists(file_path):
            print (f"Loading PDB {file_path} from server...")
            rcsb.fetch(pdb_id, pdb_type, str(PDBS_PATH))
        return file_path

    def get_sequence(pdb_id):
        try:
            pdb_id = pdb_id.strip() # Remove whitespace
            pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")

            structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)

            # TODO Excluding atoms that are not in a Amino Acid?
            (amino_sequences, _) = (
                b_structure.to_sequence(
                    structure[b_structure.filter_amino_acids(structure)]))
            # Note: len(amino_sequences) == # of chains

            # TODO Should I put something between chains?
            sequence = "".join([str(s) for s in amino_sequences])
            return sequence

        except Exception as e:
            return f"Error: {e}"

    def find_pdb_id_from_sequence(query_sequence, maybe_stale_pdb_id):
        pdb_id_sequence = get_sequence(maybe_stale_pdb_id)

        if pdb_id_sequence == query_sequence:
            print ("Skipped BLAST run, PDB ID already known.")
            not_stale_pdb_id = maybe_stale_pdb_id
            return not_stale_pdb_id
        else:
            # TODO Use Blast or some kind of Sequence Database
            # Skipping Blast which is rate limited for now.
            return None

        print ("Running blast to find PDB ID of sequence...")
        blast_app = blast.BlastWebApp(
            "blastp", query_sequence, database="pdb")
        blast_app.start()
        blast_app.join()

        alignments = blast_app.get_alignments()
        if len(alignments) == 0:
            return None
        return alignments[0].hit_id

    def build_database_html(sequence, maybe_stale_pdb_id):
        pdb_id = find_pdb_id_from_sequence(sequence, maybe_stale_pdb_id)
        if pdb_id is None:
            return "<h3>Folding Structure of Sequence not found.</h3>"

        # Remove chain information if present.
        if pdb_id.find("_") != -1: # "1CRN_A" -> "1CRN"
            pdb_id = pdb_id[:pdb_id.index("_")]

        # Extract secondary structure from PDBX file.
        pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")
        structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)
        # TODO Should I only use AA atoms?
        atoms = structure[
            b_structure.filter_amino_acids(structure)]
        residue_secondary_structures = b_structure.annotate_sse(atoms)
        residue_ids = b_structure.get_residues(atoms)[0]
        residue_id_to_sse = {}
        for (residue_id, sse) in zip(residue_ids, residue_secondary_structures):
            residue_id_to_sse[int(residue_id)] = sse

        # Extract PDB string from PDB file.
        pdb_file_path = fetch_pdb_if_necessary(pdb_id, "pdb")
        pdb_string = Path(pdb_file_path).read_text()

        return py3DMolViewWrapper().build_html_with_secondary_structure(
            width, height, pdb_string, residue_id_to_sse)

    def remove_nested_quotes(html):
        """Remove nested quotes in HEADER"""

        if html.find("HEADER") == -1:
            return html

        # Need this because srcdoc adds another quote and triple is tricky.
        i = html.index("HEADER")
        # Getting the HEADER double quote end by matching:  \n","pdb");
        j = html[i:].index('"pdb");') + i - len('",')
        header_html = html[i:j]
        header_html = header_html.replace("\\'", "")
        header_html = header_html.replace('\\"', "")
        header_html = header_html.replace("'", "")
        header_html = header_html.replace('"', "")
        print ("Len before:", len(html))
        html = html[:i] + header_html + html[j:]
        print ("Len after:", len(html))
        return html

    def postprocess_html(html):
        html = remove_nested_quotes(html)
        html = html.replace("'", '"')

        html_wrapped =  f"""<!DOCTYPE html><html>{html}</html>"""
        iframe_html =  (f"""<iframe style="width: 100%; height: 400px;" """
            f"""allow="midi; display-capture;" frameborder="0" """
            f"""srcdoc='{html_wrapped}'></iframe>""")
        return iframe_html
        # return html_wrapped

    def run_esm_and_graph_debug(sequence, maybe_stale_pdb_id):
        # Data cleaning
        sequence = sequence.strip() # Remove whitespace
        maybe_stale_pdb_id = maybe_stale_pdb_id.strip() # Remove whitespace

        database_sse_html = build_database_html(sequence, maybe_stale_pdb_id)

        # Save HTMLs to volume for debugging issues.
        HTMLS_PATH.mkdir(parents=True, exist_ok=True)
        with open(str(HTMLS_PATH / "db.html"), "w") as f:
            f.write(postprocess_html(database_sse_html))

        return [postprocess_html(h)
                    # for h in (esm_sse_html, esm_pLDDT_html, database_sse_html)]
                    for h in [database_sse_html] * 3]

    def run_esm_and_graph(sequence, maybe_stale_pdb_id):
        # Data cleaning
        sequence = sequence.strip() # Remove whitespace
        maybe_stale_pdb_id = maybe_stale_pdb_id.strip() # Remove whitespace

        # Run ESM
        esm_protein: ESMProtein = ModelInference().generate.remote(sequence)
        residue_pLDDTs = (100 * esm_protein.plddt).tolist()
        atoms = esm_protein.to_protein_chain().atom_array
        residue_secondary_structures = b_structure.annotate_sse(atoms)
        residue_ids = b_structure.get_residues(atoms)[0]
        residue_id_to_sse = {}
        for (residue_id, sse) in zip(residue_ids, residue_secondary_structures):
            residue_id_to_sse[int(residue_id)] = sse

        esm_sse_html = (
            py3DMolViewWrapper().build_html_with_secondary_structure(
            width, height, esm_protein.to_pdb_string(),
            residue_id_to_sse))

        esm_pLDDT_html = (
            py3DMolViewWrapper().build_html_with_pLDDTs(
            width, height, esm_protein.to_pdb_string(), residue_pLDDTs))

        database_sse_html = build_database_html(sequence, maybe_stale_pdb_id)

        # Save HTMLs to volume for debugging issues.
        HTMLS_PATH.mkdir(parents=True, exist_ok=True)
        with open(str(HTMLS_PATH / "db.html"), "w") as f:
            f.write(postprocess_html(database_sse_html))

        return [postprocess_html(h)
                    for h in (esm_sse_html, esm_pLDDT_html, database_sse_html)]

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
        "Insulin [1ZNI]",
        "Myoglobin [1MBO]",
        "Hemoglobin [1GZX]",
        "GFP [1EMA]",
        "Collagen [1CGD]",
        "Antibody / Immunoglobulin G [1IGT]",
        "Actin [1ATN]",
        "Ribonuclease A [5RSA]",
    ]
    example_pdbs = [f"{i+1}) {x}" for i, x in enumerate(example_pdbs)]

    with gr.Blocks(
        theme=theme, css=css, title="ESM3 Protein Folding"
    ) as interface:
        # Title
        gr.Markdown("# Fold Proteins using ESM3 Fold")

        with gr.Row():
            # PDB ID text box + Get sequence from it button
            with gr.Column():
                gr.Markdown("## Custom PDB ID")
                pdb_id_box = gr.Textbox(
                    label="Enter PDB ID or select one on the right",
                    placeholder="e.g. '5JQ4', '1MBO', '1TPO', etc.")
                get_sequence_button = (
                    gr.Button("Retrieve Sequence from PDB ID", variant="primary"))

                pdb_link_button = gr.Button(value="Open PDB page for ID")
                rcsb_link = "https://www.rcsb.org/structure/"
                pdb_link_button.click(
                    fn=None,
                    inputs=pdb_id_box,
                    js=f"""(pdb_id) => {{ window.open("{rcsb_link}" + pdb_id) }}""",
                                )

            with gr.Column():
                def extract_pdb_id(idx):
                    pdb = example_pdbs[idx]
                    s = pdb.index("[") + 1
                    e = pdb.index("]")
                    return pdb[s:e]
                h = int(len(example_pdbs) / 2)

                gr.Markdown("## Example PDB IDs")
                with gr.Row():
                    with gr.Column():
                        # add in a few examples to inspire users
                        for i, pdb in enumerate(example_pdbs[:h]):
                            btn = gr.Button(pdb, variant="secondary")
                            btn.click(
                                fn=lambda j=i: extract_pdb_id(j), outputs=pdb_id_box)

                    with gr.Column():
                        # more examples
                        for i, pdb in enumerate(example_pdbs[h:]):
                            btn = gr.Button(pdb, variant="secondary")
                            btn.click(
                                fn=lambda j=i+4: extract_pdb_id(j), outputs=pdb_id_box)

        # Sequence text box + Run ESM from it button
        gr.Markdown("## Sequence")
        sequence_box = gr.Textbox(
            label="Enter a sequence or retrieve it from a PDB ID",
            placeholder="e.g. 'MVTRLE...', 'GKQEG...', etc.")
        run_esm_and_graph_button = (
            gr.Button("Run ESM3 Fold on Sequence", variant="primary"))

        # 3 Columns of Protein Folding
        htmls = []
        with gr.Row():
            # Column 1: ESM Prediction with SSE coloring.
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - Secondary Structs")
                gr.Image(  # output image component
                    height=100, width=400,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 2: ESM Prediction with Confidence coloring.
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - PLTT Confidence")
                gr.Image(  # output image component
                    height=100, width=400, value="/assets/plddtLegend2.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 3: Database showing Crystalized form of Protein if avail.
            with gr.Column():
                gr.Markdown("## Crystalized Structure from PDB")
                gr.Image(  # output image component
                    height=100, width=400,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

        get_sequence_button.click(
            fn=get_sequence, inputs=[pdb_id_box], outputs=[sequence_box])
        run_esm_and_graph_button.click(fn=run_esm_and_graph,
            inputs=[sequence_box, pdb_id_box], outputs=htmls)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

@app.local_entrypoint()
def main():
    pdb_id = "1ZNI"
    # pdb_id = "1MBO"
    test = retrieve_pdb.remote(pdb_id)

    # # YFP?
    # sequence = (
        # "AMFSKVNNQKMLEDCFYIRKKVFVEEQGIPEESEIDEYESESIHLIGYDNGQPVATARIRPINETTVKIERVAVMKSHRGQGMGRMLMQAVESLAKDEGFYVATMNAQCHAIPFYESLNFKMRGNIFLEEGIEHIEMTKKLT")
    # for i in range(10):
        # ModelInference().generate.remote(sequence)
        # x = 1

if __name__ == "__main__":
    main()
