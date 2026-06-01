"""Model loading and the `ESMFold2Designer` orchestrator.

`ESMFold2Designer.load` pulls down the inversion + critic ESMFold2 checkpoints and
the ESMC LM from Hugging Face, optionally adds the scaling critics, and
wires them into the gradient-descent loop in `binder_design.design.design_binder`.

`ESMFold2Designer.design` is the public entry point that the example's Modal
class calls into; it just forwards to `design_binder` with the loaded models.
"""

from functools import partial
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.models.esmc.modeling_esmc import (
    ESMCForMaskedLM,
    UnifiedTransformerBlock as TransformerBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_common import (
    CUE_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
    MSAEncoder as ESMFold2MSAEncoder,
)

from .constants import CHECKPOINT_LM, COMPILE, REUSE_ESMC
from .design import design_binder

_ESMC = None


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str
) -> Any:
    """Load an ESMFold2 critic from Hugging Face.

    Caches the ESMC-6B encoder across non-scaling checkpoints to save VRAM
    and load time."""
    global _ESMC
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        if _ESMC is None:
            model.load_esmc(model.config.esmc_id)
            _ESMC = model._esmc
        else:
            model._esmc = _ESMC
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """`torch.compile` the heavy submodules of an ESMFold2 model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock, TransformerBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)  # pyright: ignore

    model.apply(_maybe_compile_module)


class ESMFold2Designer:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool):
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(
                model_name, lm_dropout=0.5, cache_esmc=True, device="cuda"
            )
            for model_name in self.inversion_model_names
        }
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )
        for name in self.scaling_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=False, device="cpu"
            )

        self.esmc_model = ESMCForMaskedLM.from_pretrained(
            self.lm_name, torch_dtype=torch.float32
        )
        if REUSE_ESMC:
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = self.inversion_models[
                "ESMFold2-Experimental-Fast"
            ]._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

        if CHECKPOINT_LM:
            apply_activation_checkpointing(
                self.esmc_model,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda module: isinstance(module, TransformerBlock),
            )

    def design(
        self,
        target_name: str | None = None,
        target_sequence: str | None = None,
        binder_name: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
    ) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
        )
