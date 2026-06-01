"""Helper package for the ``esmfold2_binder_design`` example.

Contains the heavy lifting (model loading, prompt factories, target sequences,
loss/score functions, the gradient-guided binder optimization loop, and the
``ESMFold2Designer`` orchestration class) that would otherwise clutter the
example page rendered from ``esmfold2_binder_design.py``.

All top-level imports across this package assume the
``example-esmfold2-binder-design`` Modal Image is active; modules here are
only ever imported from inside its container.

Submodule layout:

- ``constants``: token tables and design-loop hyperparameters.
- ``prompts``: ``PromptFactory``, the bundled binder templates, and target sequences.
- ``losses``: structure losses, distogram iPTM proxy, and ESMC pseudoperplexity.
- ``folding``: ESMFold2 input prep, forward pass, and ``ProteinComplex`` reconstruction.
- ``design``: the gradient-guided ``design_binder`` loop and its tensor primitives.
- ``models``: HF model loading and the ``ESMFold2Designer`` orchestrator.

The canonical reference for the algorithms is
[Language Modeling Materializes a World Model of Protein Biology]
(https://biohub.ai/papers/esm_protein.pdf).
"""

# ---
# lambda-test: false  # auxiliary helper package
# ---

import logging

import torch

# Required for `torch.use_deterministic_algorithms(True)`; matches the
# `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var set in the example's Modal Image.
torch.use_deterministic_algorithms(True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("binder_design").setLevel(logging.INFO)

from .models import ESMFold2Designer  # noqa: E402

__all__ = ["ESMFold2Designer"]
