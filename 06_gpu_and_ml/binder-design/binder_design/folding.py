"""Structure-prediction glue: feature prep, ESMFold2 forward pass, and complex reconstruction.

`fold_and_get_distogram` is the central call. Given a target sequence and a
soft binder design, it stitches together a target|binder `StructurePredictionInput`,
runs the ESMFold2 model, and returns the distogram logits used by the design
losses (plus optional iPTM / pTM / pLDDT confidence scores when requested).

`build_complex` and its helpers convert the model's atom-coordinate output back
into an `esm.ProteinComplex` for downstream visualisation / serialisation.
"""

import string
from functools import cache
from typing import Any

import biotite.structure
import numpy as np
import torch
import torch.nn.functional as F
from esm.models.esmfold2 import (
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import MOL_TYPE_NONPOLYMER, PROTEIN_3TO1
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)

from .constants import ELEMENTS, TOKENS

# ---- Feature preparation ----


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


# ---- Folding ----


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    """Prepare inputs, run model forward, return distogram logits + raw output."""
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    # Argmax to get the designed sequence string.
    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
        for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for seq in seq_list:
        sequences = {
            sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))
        }
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for sequence, chain_id in sequences.items()
            ]
        )
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms))

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


# ---- Atom array / ProteinComplex reconstruction ----


_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


def build_complex(
    inputs: dict[str, torch.Tensor], output: dict[str, Any]
) -> ProteinComplex:
    """Build a `ProteinComplex` from model output."""
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
    )
    return ProteinComplex.from_chains(
        [ProteinChain.from_atomarray(a) for a in biotite.structure.chain_iter(atom_arr)]
    )
