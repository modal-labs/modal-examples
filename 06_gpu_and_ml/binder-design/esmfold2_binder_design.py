# ---
# cmd: ["modal", "run", "-m", "06_gpu_and_ml.binder-design.esmfold2_binder_design::main"]
# ---

# # Design protein binders at scale with ESMFold2 and ESMC

# Protein folding was a landmark breakthrough in computational biology.
# But for many applications, we don't just want to predict the structures of existing proteins —
# we want to design new proteins that can modulate biology.

# One of the most important ways to do that is through binding.
# Protein-protein interactions drive much of biological function,
# and the ability to design molecules that bind specific targets
# opens the door to new research tools and therapeutics.
# Recent AI approaches have tackled binder design by inverting
# structure prediction models via an iterative optimization process:
# 1. Fold a candidate binder together with the target protein.
# 2. Score the resulting structure based on how well the binder folds and binds.
# 3. Take a step in sequence space that improves the score.
# 4. Repeat.

# In this example, we'll demonstrate how implement this process on Modal
# using [ESMFold2 and ESMC](https://biohub.ai/esm/protein/about), state-of-the-art models
# developed at [Biohub](https://biohub.ai/) that can predict the stucture of biomolecular complexes.
# Check out their [technical report](https://bhp-papers-prod.s3.us-west-2.amazonaws.com/esm_protein.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAU6GD3FYNPNY5VQML%2F20260601%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260601T211329Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE0aCXVzLXdlc3QtMiJHMEUCIQCo7iFVdbR8PdDuUKXkftQzwb17YIokN8eqsU4GVNLfXwIgHiY8F9BRKFhS52xYV8vva0yAJMDVBBqr%2BSVWTbnKmu4qqQUIFhAAGgwzMzk3MTMxNDIyOTgiDMbnfvOsb0AlGC9GIiqGBfl271MAE4q1YvlP0gVJcR0GQGmNLzYLOFZlKZuAl%2B2f10e9ff6O%2Fvq5OJvVoVFZPeFRAMFZnVpu7qmjZdTsDEkr9Kb2RUrItMI1ycrMkO%2FpdRXfLEZVCQ9l1frm93b3arhbJrA2QS0l7h7Kvqgn3vdh3sfkxP1oeOEf5H2noCepBdJdNozmQ9Mb%2FBU5BnVN02SzQ4W0HgY00XTNbIqdOcW2cQ%2B21zcFwMeoxxzDnLcNuIG6pAD2fJ7IJBmijW9aSLOhqR42e%2F0ZsJiRHIINIYC1x1z1OiPLIQM6ILV9Wxevn761h1zCONepQvxsSQ%2Byha7ztQ7%2BGL4dQskSiXA84IZa%2F0oUWSPPPJjwoIynM0dqONngLtjV3yX9uZ7o6NhesV0zEJgfgR1uuibPX3hmtd0uSTT1b%2FzqWwWqnVVpw8pXOPUaYO1uoD8kgud45U%2F2fKZYyHDkG9oL%2F3CQO7C%2B5okYLMBFDKQGJb4yb75uTCot0IYQC%2FlQbwrBkLiwCFvElX0%2FZ%2BpYOAiaeUFMiiJp6jCItrXPpvxKrHZnpOpRaXdaAkm%2F64Kwj2SW0mjok02hHW3PH39iY5dJ0DghclJBOEE1qFniEC5gFtzfEq4%2Bpm7J6gklIjIPdIbZT2vCKZBDeD1QjY9zdM8m%2B9sLAQokhqVyI4OJijDwzMoEA6vIrjjST8yy6dHq4oLrWAXBwt2dDJPRxNLOPKsbRRPfHbkwqxEu%2BPEfwkUchp5VchYrPlODlOWr6%2FdttlF%2FIwrFbMbCRJqLNQjCMpSaZJdZy1wWjMIWrsNx2KKoPIcs%2FaV26fN0lY%2BQ5DXESYLTCR1zBkCVXJirIK7o7xIEt715r0FlsAJDQqDAGVAw%2Bev30AY6mQFHzylgearF%2BJ1Qw4tnwpmoig%2FIAhz5LidTy1nyBTemeT2O0Pr8U9dU%2BkDaAGdvNL0Rkxs2SWvh0BQvQA6OKy9gILvE2ZXDqY2JnjaCyEKyrzigIL7sW2UFZUQwV86rVCEosqsY1fIvFRtKI3NRDkDhk%2Bqg7BfmA94bkXcp0PG863YrF76%2BtzXHNJa1vVmRv0CSrKWSX0bfAoY%3D&X-Amz-Signature=e478c493d9cbaa2d48de8a100b4e93182b00b064c0c07b6809d0dc79c136ed74&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)
# to see how the models were developed and used to design and experimentally validate binders against therapeutically relevant targets.

# We'll start by building a Modal Function that designs a single binder; then with only
# a few more lines of code, we'll write an orchestrator function
# that executes a large-scale search powered by Modal's autoscaling infrastructure and global GPU capacity.

# ## Setup

from pathlib import Path
from typing import Optional

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app = modal.App(
    name="example-esmfold2-binder-design",
)

# ## Defining our Modal Image

# We'll use `Image.micromamba` as our base image because a few of the packages we need
# are only available via Conda. We'll also install the [`esm`](https://github.com/Biohub/esm)
# library from CZ Biohub (which pulls in a custom fork of `transformers`) and a few other helpful libraries
# for working with protein sequences.

# We set `CUBLAS_WORKSPACE_CONFIG` which allows us to ensure reproducibility by calling
# `torch.use_deterministic_algorithms(True)` at the top of our remote code.

ESM_REVISION = (
    "f652b471d29da828b31e9b7a9cf7d0a7803240f5"  # see https://github.com/Biohub/esm
)

image = (
    modal.Image.micromamba(python_version="3.12")
    .run_commands("apt update && apt install -y git build-essential")
    .micromamba_install(
        "anarci=2024.05.21-0",
        channels=["conda-forge", "bioconda"],
    )
    .pip_install(
        f"esm @ git+https://github.com/Biohub/esm.git@{ESM_REVISION}",
        "abnumber==0.4.4",
        "pyarrow==18.1.0",
    )
    .env(
        {
            "HF_HOME": "/models",
            "HF_XET_HIGH_PERFORMANCE": "1",  # speed up Hugging Face downloads
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            # required for torch.use_deterministic_algorithms(True)
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)

# ## Caching weights and persisting results on Modal Volumes

# ESMFold2 builds on the 6B-parameter ESMC encoder; together with the four
# critic models used for final scoring, the model weights come in around ~50 GB.
# We cache them on a [Modal Volume](https://modal.com/docs/guide/volumes)
# which delivers much better performance at cold-start time than re-downloading
# from Hugging Face each time.

models_volume = modal.Volume.from_name("esmfold2-models", create_if_missing=True)
models_dir = Path("/models")

# A second Volume will store our results.

results_volume = modal.Volume.from_name(
    "esmfold2-binder-design-results", create_if_missing=True
)
results_dir = Path("/results")


# ## Designing a binder on Modal

# To run binder design on Modal, we define a `BinderDesignService` class and
# wrap it with the `@app.cls` decorator. The decorator takes arguments that
# describe the infrastructure our code needs: the Image and both Volumes we
# defined, plus an H100 GPU which has enough memory for the 6B-parameter ESMC encoder and the
# four ESMFold2 "hero" critic models.

# Inside the class, the [`@modal.enter()` lifecycle hook](https://modal.com/docs/guide/lifecycle-functions#modalenter)
# downloads and initializes those models once per container start, so subsequent
# `design` calls on the same container reuse the loaded weights.

# We decorate our `design` method with `@modal.method()` to enable remote
# execution. We'll see it called both via `.remote()` (single design) and via
# `.spawn()` + [`modal.FunctionCall.gather`](https://modal.com/docs/reference/modal.FunctionCall)
# (parallel sweep) further below. The class itself is a thin wrapper around
# [`ESMFold2Designer`](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/binder-design/binder_design/models.py)
# from the helper package, which handles the actual model loading and the
# gradient-guided optimization loop (`design_binder` in
# [`binder_design.design`](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/binder-design/binder_design/design.py)).


@app.cls(
    image=image,
    volumes={models_dir: models_volume},
    gpu="H100",
    timeout=1 * HOURS,
)
class BinderDesignService:
    """Modal entry point for ESMFold2-driven binder design.

    Set ``use_scaling_critics=True`` to also load the 15-checkpoint
    scaling-experiment ensemble (distogram binding confidence only).
    """

    use_scaling_critics: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        from .binder_design import ESMFold2Designer

        self._designer = ESMFold2Designer()
        self._designer.load(self.use_scaling_critics)

    @modal.method()
    def design(
        self,
        target_name: Optional[str] = None,
        target_sequence: Optional[str] = None,
        binder_name: Optional[str] = None,
        binder_sequence: Optional[str] = None,
        is_antibody: Optional[bool] = None,
        seed: int = 0,
        batch_size: int = 1,
    ):
        return self._designer.design(
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
        )


# ## Fanning out a sweep with selection

# A single design run gives you one candidate per batch slot. To recover the
# kind of hit rates reported in the paper, you want many seeds, several binder
# templates, and several targets, then a selection pass that ranks designs by
# a combined ipTM / distogram-ipTM-proxy score.

# We orchestrate from inside a Modal Function so you don't have to worry about
# keeping a long-running process alive locally or installing any local dependencies.


@app.function(
    image=image,
    volumes={results_dir: results_volume},
    gpu="H100",
    timeout=2 * HOURS,
)
def run_sweep(
    line_sweeps: dict[str, list],
    use_scaling_critics: bool = False,
    save_filename: str = "selection.parquet",
) -> bytes:
    """Fan a grid sweep across GPUs, gather results, select top designs, ave results + return parquet."""
    import io

    from .binder_design.sweep import expand_sweep, select_designs

    designer = BinderDesignService(use_scaling_critics=use_scaling_critics)
    configs = expand_sweep(line_sweeps)

    print(f"🧬 spawning {len(configs)} design jobs")
    calls = [designer.design.spawn(**cfg) for cfg in configs]
    raw_results = modal.FunctionCall.gather(*calls)

    df_select = select_designs(configs, raw_results)

    buf = io.BytesIO()
    df_select.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    save_path = results_dir / save_filename
    save_path.write_bytes(parquet_bytes)
    results_volume.commit()
    print(f"🧬 saved {len(df_select)} selected designs to volume:{save_path}")

    return parquet_bytes


# ## From the command line

# `main` runs a single design. Override the
# `target_name` / `binder_name` to try one of the
# [bundled targets](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/binder-design/binder_design/prompts.py)
# (`cd45`, `ctla4`, `egfr`, `pd-l1`, `pdgfr`) and binder templates
# (`minibinder`, `trastuzumab_framework_vhvl`, `atezolizumab_framework_vhvl`,
# `ocankitug_framework_vhvl`), or pass an arbitrary `target_sequence` /
# `binder_sequence` directly.

# ```shell
# modal run -m 06_gpu_and_ml.binder-design.esmfold2_binder_design::main \
#     --target-name pd-l1 --binder-name minibinder
# ```


@app.local_entrypoint()
def main(
    target_name: Optional[str] = "pd-l1",
    target_sequence: Optional[str] = None,
    binder_name: Optional[str] = "minibinder",
    binder_sequence: Optional[str] = None,
    is_antibody: Optional[bool] = None,
    use_scaling_critics: bool = False,
    seed: int = 0,
    batch_size: int = 1,
):
    designer = BinderDesignService(use_scaling_critics=use_scaling_critics)
    seq, trajectory, results = designer.design.remote(
        target_name=target_name,
        target_sequence=target_sequence,
        binder_name=binder_name,
        binder_sequence=binder_sequence,
        is_antibody=is_antibody,
        seed=seed,
        batch_size=batch_size,
    )

    avg_final_loss = sum(r["final_loss"] for r in results) / len(results)
    print(f"🧬 designed sequence: {seq}")
    print(f"🧬 trajectory length: {len(trajectory)} steps")
    print(f"🧬 average final loss: {avg_final_loss:.4f}")


# `sweep` runs a grid sweep across every `(target, binder, seed)` combination
# of the targets and binders you pass in, scaling design horizontally with Modal's
# [asynchronous job processing](https://modal.com/docs/guide/job-queue).
# The selection pass runs server-side and the resulting parquet is
# written to both the `esmfold2-binder-design-results` Volume and to a local
# file for inspection.

# `target_names` and `binder_names` are passed as comma-separated strings
# because Modal's local-entrypoint CLI doesn't accept lists directly. The
# defaults sweep one target across two binder modalities -- a `minibinder`
# and the `trastuzumab_framework_vhvl` antibody template -- so a single
# command fans out across both at once:

# ```shell
# modal run -m 06_gpu_and_ml.binder-design.esmfold2_binder_design::sweep \
#     --target-names pd-l1,ctla4 \
#     --binder-names minibinder,trastuzumab_framework_vhvl \
#     --n-seeds 8
# ```


@app.local_entrypoint()
def sweep(
    target_names: str = "pd-l1",
    binder_names: str = "minibinder,trastuzumab_framework_vhvl",
    use_scaling_critics: bool = False,
    n_seeds: int = 8,
    output_path: Optional[str] = None,
):
    target_name_list = [
        name.strip() for name in target_names.split(",") if name.strip()
    ]
    binder_name_list = [
        name.strip() for name in binder_names.split(",") if name.strip()
    ]

    line_sweeps = {
        "target_name": target_name_list,
        "target_sequence": [None],
        "binder_name": binder_name_list,
        "binder_sequence": [None],
        "seed": list(range(n_seeds)),
        "batch_size": [1],
    }

    print(
        f"🧬 launching sweep: targets={target_name_list}, binders={binder_name_list}, "
        f"n_seeds={n_seeds}, use_scaling_critics={use_scaling_critics}"
    )
    parquet_bytes = run_sweep.remote(
        line_sweeps, use_scaling_critics=use_scaling_critics
    )

    if output_path is None:
        output_path = Path("/tmp") / "esmfold2_binder_design" / "selection.parquet"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(parquet_bytes)
    print(f"🧬 wrote selection parquet to {output_path}")
