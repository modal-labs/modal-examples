"""Gradient-guided binder sequence optimization.

`design_binder` is the entry point: given pre-loaded ESMFold2 inversion models,
HF critic models, and an ESMC LM, plus a target/binder spec, it runs the
gradient-descent loop from Algorithm 11 of the ESM protein paper and scores
the best per-batch sequences with each critic.

The smaller helpers in this module (`build_initial_soft_sequence_logits`,
`build_gradient_mask`, `normalized_gradient_tensor`, `sequence_to_one_hot`)
are sequence/gradient setup primitives that are tightly bound to the loop and
not used elsewhere.
"""

import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from esm.models.esmfold2.constants import PROTEIN_1TO3
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)

from .constants import (
    AA_DIMS,
    CYS_IDX,
    LEARNING_RATE,
    LOG_INTERVAL,
    MUTABLE_TOKEN,
    STEPS,
    TEMPERATURE_MIN,
    TOKEN_IDS,
    TOKENS,
    logger,
)
from .folding import build_complex, fold_and_get_distogram
from .losses import (
    compute_distogram_iptm_proxy,
    compute_esmc_pseudoperplexity_nll,
    compute_structure_losses,
)
from .prompts import BINDER_PROMPT_FACTORIES, TARGET_SEQUENCES

# ---- Sequence / gradient setup ----


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert an amino-acid sequence to a one-hot tensor [1, L, num_tokens]."""
    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """Initial logits with high confidence at fixed positions, ~0 at mutable positions, -1e6 at cysteines."""
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6  # remove cysteines
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:  # mutable position - random
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:  # fixed position
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """Per-position [B, L, V] gradient mask: 1 only at non-cysteine, mutable positions."""
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0  # (B, L)
    eff_L = index_has_nonzero_grad.sum(-1)  # (B,)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))  # (B,)
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask


# ---- Optimization loop ----


def design_binder(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM,
    target_name: str | None,
    target_sequence: str | None,
    binder_name: str | None,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
    """Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop. Returns (best_sequences, trajectory, critic_results).

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    `distogram_binding_confidence` / `cdr_distogram_binding_confidence` come from
    the distogram in all cases.
    """
    assert (target_name is None) ^ (target_sequence is None), (
        "Provide either target name or sequence."
    )
    assert (binder_name is None) ^ (binder_sequence is None), (
        "Provide either binder name or sequence."
    )

    device = "cuda"
    if target_name is not None:
        target_sequence = TARGET_SEQUENCES[target_name]
    else:
        assert target_sequence is not None
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name is None:
        assert binder_sequence is not None
        # If no binder_name and is_antibody is not specified, assume False.
        if is_antibody is None:
            is_antibody = False
    else:
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert binder_prompt_factor.is_antibody == is_antibody, (
                "Conflict in is_antibody settings."
            )
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)

    binder_length = len(binder_sequence)

    # By default, we only support single binder and target chains.
    # To support multi-chain cases, remove the asserts below and check that losses
    # and selection metrics are appropriate for your setup.
    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    # step -> {loss_name: [B] tensor on CPU}
    trajectory: dict[int, dict[str, torch.Tensor]] = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_model = list(inversion_models.values())[replicate_choice]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(
            fold_result["distogram_logits"], binder_length
        )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        # Recompute the logits -> design transform for a fresh graph.
        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=4,
                n_passes=4,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature

        optimizer.step()

        step = global_step
        step_losses = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        trajectory[step] = step_losses
        loss_str = "  ".join(
            f"{k}={v.mean().item():.4f}" for k, v in step_losses.items()
        )
        if step % LOG_INTERVAL == 0:
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}")
        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(STEPS):
        # Cosine schedule
        t = (step + 1) / STEPS
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    # Score every batch index against every critic.
    critic_results: list[dict] = []
    target_length = len(target_sequence.replace("|", ""))
    for batch_idx in range(batch_size):
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        for critic_name, critic_model in hf_critic_models.items():
            is_scaling_critic = "ESMFold2-Experimental-Fast-base" in critic_name
            if is_scaling_critic:
                critic_model.cuda()
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            if is_scaling_critic:
                critic_model.cpu()
            pred_complex = build_complex(final_fold["inputs"], final_fold["output"])
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"], target_length, binder_seq, is_antibody
            )
            iptm = final_fold["iptm"].item() if final_fold["iptm"] is not None else None
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "critic_name": critic_name,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_seq,
                    "complex": pred_complex,
                    "final_loss": trajectory[global_step - 1]["total_loss"][
                        batch_idx
                    ].item(),
                    "iptm": iptm,
                    "logits": logits[batch_idx].detach().cpu(),
                    **iptm_proxy_scores,
                }
            )

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": trajectory[global_step - 1]["total_loss"][
                        batch_idx
                    ].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return best_sequences, trajectory, critic_results
