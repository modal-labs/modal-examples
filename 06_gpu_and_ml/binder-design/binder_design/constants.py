"""Token tables, element tables, and design-loop hyperparameters."""

import logging

from esm.models.esmfold2 import ELEMENT_NUMBER_TO_SYMBOL
from esm.models.esmfold2.constants import PROTEIN_1TO3, RES_TYPE_TO_CCD

logger = logging.getLogger("binder_design")


# ---- Token / element tables ----

TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]

ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()

TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}

AA_DIMS = 20
# Cysteine index in the 20-dim AA space (TOKEN_IDS are offset by 2 for <pad> and -).
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2

# Marker character used inside binder prompt strings for "this position is mutable".
MUTABLE_TOKEN = "#"


# ---- Design-loop hyperparameters ----

LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
STEPS = 10
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15
CHECKPOINT_LM = False
COMPILE = False
# NOTE - This significantly reduces VRAM usage.
# On config (target_name="cd45", binder_name="trastuzumab_framework_vhvl", batch_size=1)
# this reduces VRAM from 51GB -> 27GB and enables increasing batch size up to 6.
# We are testing this setting in silico, and may change the default to True in the future.
REUSE_ESMC = False
