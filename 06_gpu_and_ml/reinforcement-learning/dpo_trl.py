# ---
# cmd: ["modal", "run", "06_gpu_and_ml/reinforcement-learning/dpo_trl.py::train"]
# ---

# # Align a model with human preferences using DPO and TRL

# This example demonstrates how to align a language model with
# [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO) on
# Modal using the TRL [`DPOTrainer`](https://huggingface.co/docs/trl/main/en/dpo_trainer).

# DPO is an alternative to reinforcement learning algorithms like
# [GRPO](https://arxiv.org/pdf/2402.03300) and PPO for aligning models with human
# preferences. Rather than sampling rollouts, scoring them with a reward function,
# and updating the policy through an RL loop (as in the
# [GRPO example](https://modal.com/docs/examples/grpo_trl)), DPO optimizes directly
# on a static dataset of `(prompt, chosen, rejected)` triples using a closed-form
# loss derived from the same RLHF objective. This makes DPO simpler to implement,
# cheaper to train, and a good fit for cases where you already have — or can
# collect — pairwise preference data instead of a programmatic reward function.

# ## Setup

# Import the necessary modules for Modal deployment.
from pathlib import Path

import modal

MINUTES = 60  # seconds

# ## Defining the image and app

app = modal.App("example-dpo-trl")

# We define an image with TRL, a LoRA-compatible `peft`, and their dependencies.
# Following the [pinning conventions](https://modal.com/docs/guide/images) for this
# repo, every container dependency is pinned to at least a SemVer minor version.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.0",
        "transformers==4.57.1",  # 4.57.0 is yanked from PyPI
        "trl==0.28.0",
        "peft==0.15.2",
        "datasets==3.5.1",
        "accelerate==1.6.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster downloads
)

with image.imports():
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

# ## Caching weights and checkpoints

# We use [Modal Volumes](https://modal.com/docs/guide/volumes) to cache downloaded
# base model weights and to persist the trained LoRA checkpoints, so that repeat
# runs don't need to redownload or lose progress.

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # pin to avoid surprises!

MODELS_DIR = Path("/models")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "example-dpo-trl-checkpoints", create_if_missing=True
)

# ## Preference dataset

# We use [`trl-lib/ultrafeedback_binarized`](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized),
# a preprocessed, ready-to-train version of the UltraFeedback preference dataset.
# It has `chosen` and `rejected` columns, each a full conversation (a list of
# `{role, content}` turns) that share the same leading user turn but end with a
# different assistant response. `DPOTrainer` accepts this "implicit prompt" format
# directly, automatically splitting off the shared prefix as the prompt, so no
# further preprocessing is required.


def load_preference_dataset(max_examples: int):
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


# ## Kicking off a training run

# We attach a LoRA adapter rather than fine-tuning the full model, following the
# same efficiency motivation as the
# [Unsloth finetuning example](https://modal.com/docs/examples/unsloth_finetune):
# only a small fraction of parameters need gradients, which keeps this example
# runnable on a single GPU.


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, MODELS_DIR: checkpoints_volume},
)
def train(max_steps: int = 5, max_examples: int = 256) -> None:  # increase both!
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    train_dataset = load_preference_dataset(max_examples)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=str(MODELS_DIR / "dpo-qwen2.5-0.5b"),
        max_steps=max_steps,  # to simplify testing; remove for production use cases
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,  # controls how far the trained model may drift from the reference
        save_steps=max_steps,
        logging_steps=1,
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    checkpoints_volume.commit()


# To run: `modal run --detach dpo_trl.py::train --max-steps 500 --max-examples 0`
# (`--max-examples 0` uses the full dataset). The saved LoRA adapter can then be
# loaded for inference the same way as in the
# [Unsloth finetuning example](https://modal.com/docs/examples/unsloth_finetune), or
# served with a [vLLM inference endpoint](https://modal.com/docs/examples/vllm_inference).
