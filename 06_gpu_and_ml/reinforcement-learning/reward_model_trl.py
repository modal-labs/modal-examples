# ---
# cmd: ["modal", "run", "06_gpu_and_ml/reinforcement-learning/reward_model_trl.py::train"]
# ---

# # Train a reward model for RLHF using TRL

# This example demonstrates how to train a reward model on Modal using TRL's
# [`RewardTrainer`](https://huggingface.co/docs/trl/main/en/reward_trainer).

# A reward model scores a `(prompt, response)` pair with a single scalar,
# learned from pairwise human (or AI) preference data. It's the missing piece
# for classic RLHF pipelines built on PPO: PPO needs a reward *signal* to
# optimize against, which this example produces, whereas the
# [GRPO example](https://modal.com/docs/examples/grpo_trl) sidesteps a learned
# reward model entirely by scoring rollouts with a programmatic reward function,
# and the [DPO example](https://modal.com/docs/examples/dpo_trl) skips the RL
# loop altogether and optimizes directly on the preference pairs. All three are
# valid ways to turn preference data into a better model — which one fits
# depends on whether you have a programmatic reward, a static preference
# dataset, or need a reusable scorer for downstream RL.

# ## Setup

# Import the necessary modules for Modal deployment.
from pathlib import Path

import modal

MINUTES = 60  # seconds

# ## Defining the image and app

app = modal.App("example-reward-model-trl")

# We define an image with TRL, a LoRA-compatible `peft`, and their dependencies.
# Following the [pinning conventions](https://modal.com/docs/guide/images) for this
# repo, every container dependency is pinned to at least a SemVer minor version.

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.7.0",
    "transformers==4.57.1",  # 4.57.0 is yanked from PyPI
    "trl==0.28.0",
    "peft==0.15.2",
    "datasets==3.5.1",
    "accelerate==1.6.0",
    "huggingface-hub==0.36.0",
)

with image.imports():
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

# ## Caching weights and checkpoints

# We use [Modal Volumes](https://modal.com/docs/guide/volumes) to cache downloaded
# base model weights and to persist the trained reward model, so that repeat runs
# don't need to redownload or lose progress.

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # pin to avoid surprises!

MODELS_DIR = Path("/models")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "example-reward-model-trl-checkpoints", create_if_missing=True
)

# ## Preference dataset

# We use the same [`trl-lib/ultrafeedback_binarized`](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)
# dataset as the [DPO example](https://modal.com/docs/examples/dpo_trl): `chosen`
# and `rejected` columns, each a full conversation sharing a leading user turn but
# ending in a different assistant response. `RewardTrainer` accepts this raw,
# untokenized format directly — it applies the tokenizer's chat template and
# tokenizes `chosen`/`rejected` internally, so no preprocessing is required here.


def load_preference_dataset(max_examples: int):
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


# ## Kicking off a training run

# We load the base model with a sequence-classification head (a single scalar
# output, `num_labels=1`) rather than the causal-LM head used for text generation,
# and attach a LoRA adapter on top of it — the same efficiency motivation as the
# [Unsloth finetuning example](https://modal.com/docs/examples/unsloth_finetune).
# Note the `task_type="SEQ_CLS"` on the LoRA config, which differs from the
# `"CAUSAL_LM"` task type used for DPO and GRPO, since we're training a classifier
# head rather than a generator.


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, MODELS_DIR: checkpoints_volume},
)
def train(max_steps: int = 5, max_examples: int = 256) -> None:  # increase both!
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, num_labels=1
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    train_dataset = load_preference_dataset(max_examples)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_CLS",
    )

    training_args = RewardConfig(
        output_dir=str(MODELS_DIR / "reward-model-qwen2.5-0.5b"),
        max_steps=max_steps,  # to simplify testing; remove for production use cases
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_length=1024,
        # Penalizes reward magnitude drifting away from zero, which keeps
        # scores comparable across training runs. See the reward modeling
        # section of the InstructGPT paper: https://arxiv.org/abs/2203.02155
        center_rewards_coefficient=0.01,
        save_steps=max_steps,
        logging_steps=1,
        bf16=True,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    checkpoints_volume.commit()


# To run: `modal run --detach reward_model_trl.py::train --max-steps 500 --max-examples 0`
# (`--max-examples 0` uses the full dataset). The resulting reward model scores
# `(prompt, response)` pairs with a single scalar and can be used as the reward
# signal in a PPO-based RLHF pipeline, or as an automatic judge for rejection
# sampling / best-of-N selection over candidate completions.
