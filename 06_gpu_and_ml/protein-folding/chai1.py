# # Fold proteins with Chai1

# ## Setup

import json
from pathlib import Path

import modal

chai_model_volume = modal.Volume.from_name(
    "chai1-models", create_if_missing=True
)
chai_preds_volume = modal.Volume.from_name(
    "chai1-preds", create_if_missing=True
)
models_dir = Path("/models/chai1")
preds_dir = Path("/preds")
here = Path(__file__).parent

MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12").run_commands(
        "uv pip install --system --compile-bytecode chai_lab==0.4.2 hf_transfer==0.1.8"
    )
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CHAI_DOWNLOADS_DIR": str(models_dir)})

app = modal.App(name="example-chai1-inference")

# ## Running Chai1 from the command line


@app.local_entrypoint()
def main(
    force_redownload: bool = False,
    fasta_file: str = None,
    inference_config_file: str = None,
    output_dir: str = None,
    run_id: str = None,
):
    import hashlib
    from uuid import uuid4

    print("🧬 checking inference dependencies")
    download_inference_dependencies.remote(force=force_redownload)
    if fasta_file is not None:
        print(f"🧬 running Chai inference on {fasta_file}")
        fasta_content = Path(fasta_file).read_text()
    else:
        fasta_content = ">protein|name=example-of-short-protein\n"
        fasta_content += "AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"

    if inference_config_file is None:
        inference_config_file = here / "chai1_default_inference.json"
    print(f"🧬 loading chai inference config from {inference_config_file}")
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


# ## Running Chai1 on Modal


@app.function(
    timeout=15 * MINUTES,
    gpu="A100",
    volumes={models_dir: chai_model_volume, preds_dir: chai_preds_volume},
    image=image,
)
def chai1_inference(
    fasta_content: str, inference_config: dict, run_id: str
) -> list[(bytes, str)]:
    from pathlib import Path

    import torch
    from chai_lab import chai1

    N_DIFFUSION_SAMPLES = 5  # hard-coded in chai1

    fasta_file = Path("/tmp/inputs.fasta")
    fasta_file.write_text(fasta_content)

    output_dir = Path("/preds") / run_id

    chai1.run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir,
        device=torch.device("cuda"),
        **inference_config,
    )

    print(
        f"🧬 done, results written to {output_dir.relative_to('/preds')} on {chai_preds_volume}"
    )

    results = []
    for ii in range(N_DIFFUSION_SAMPLES):
        scores = (output_dir / f"scores.model_idx_{ii}.npz").read_bytes()
        cif = (output_dir / f"pred.model_idx_{ii}.cif").read_text()

        results.append((scores, cif))

    return results


# ## Addenda


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

        await asyncio.gather(*tasks)


async def download_file(session, url: str, local_path: Path):
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while chunk := await response.content.read(8192):
                f.write(chunk)
