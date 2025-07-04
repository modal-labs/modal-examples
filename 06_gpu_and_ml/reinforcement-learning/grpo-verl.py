# ---
# lambda-test: false
# ---

# # Run GRPO on Modal using VERL

# This example demonstrates how to run [GRPO](https://arxiv.org/pdf/2402.03300) on modal using the [verl](https://github.com/volcengine/verl) framework.
# GRPO is a reinforcement learning algorithm introduced by DeepSeek, and was used to train DeepSeek R1.
# Verl is a reinforcement learning training library that is an implementation of [HybridFlow](https://arxiv.org/abs/2409.19256v2), an RLHF framework.

# The full code for this example can be found [here](https://github.com/modal-labs/modal-verl)

# ## Setup

# Import the necessary modules for Modal deployment
from __future__ import annotations

import subprocess
from typing import Literal, Optional

import modal

# ## Defining the image and app

app = modal.App("grpo-verl-example")

# We define an image where we clone the VERL repo and install its dependencies. We use a base verl image as a starting point

VERL_REPO_PATH: str = "/root/verl"
image = (
    modal.Image.from_registry(
        "whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3"
    )
    .apt_install("git")
    .run_commands([f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}"])
    .pip_install("verl[vllm]==0.4.1")
)

# ## Defining the dataset

# In this example, we'll use reinforcement learning to train a model to solve math problems.
# We use the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of math problems.
# We use a [modal volume](https://modal.com/docs/reference/cli/volume#modal-volume) to store the data

DATA_PATH: str = "/data"
data_volume = modal.Volume.from_name("grpo-verl-example-data", create_if_missing=True)


@app.function(image=image, volumes={DATA_PATH: data_volume})
def prep_dataset() -> None:
    subprocess.run(
        [
            "python",
            f"{VERL_REPO_PATH}/examples/data_preprocess/gsm8k.py",
            "--local_dir",
            DATA_PATH,
        ],
        check=True,
    )


# You can kickoff the dataset download with
# `modal run <filename.py>::prep_dataset`

# ## Defining a reward function

# In reinformcement learning, we define a reward function for the model
# We can define this in a separate file, that we then pass as an argument to verl.
# Here, we call the filename reward.py, but you may choose to call it what you please.
# We use a `default` reward function for GSM8K from the [verl repo](https://github.com/volcengine/verl/blob/v0.1/verl/utils/reward_score/gsm8k.py), modified to return 1.0 if it's a correct answer and 0 otherwise

# In `reward.py`
import re


def extract_solution(
    solution_str: str, method: Literal["strict", "flexible"] = "strict"
) -> Optional[str]:
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer: Optional[str] = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer: Optional[str] = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str: list[str] = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


# Reward functions need to follow a [predfined signature](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)


def compute_reward(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict
) -> float:
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


# We then define constants to pass into verl during the training run. We also make sure our image has the custom reward function

PATH_TO_REWARD_FUNCTION: str = "/root/reward.py"
REWARD_FUNCTION_NAME: str = "compute_reward"

image = image.add_local_file("./reward.py", PATH_TO_REWARD_FUNCTION)


# ## Kicking off a training run

## We define some constants for the training run
CHECKPOINTS_PATH: str = "/checkpoints"
TRAINING_FILES_PATH: str = f"{DATA_PATH}/train.parquet"
VALIDATION_FILES_PATH: str = f"{DATA_PATH}/test.parquet"
MAX_PROMPT_LENGTH: int = 1024
MAX_RESPONSE_LENGTH: int = 1024
BATCH_SIZE: int = 1024
MODEL: str = "Qwen/Qwen3-8B"
LEARNING_RATE: str = "1e-6"
MINI_BATCH_SIZE: int = 128
MICROBATCH_SIZE_PER_GPU: int = 16

# We also a define a volume for storing model checkpoints
checkpoints_volume = modal.Volume.from_name(
    "grpo-verl-example-heckpoints", create_if_missing=True
)

# Now, we write a modal function for kicking off the training run
# If you wish to use wandb, as we do in this code, you'll need to create a wandb [secret](https://modal.com/docs/guide/secrets#secrets)


@app.function(
    image=image,
    gpu="H200:8",
    volumes={
        CHECKPOINTS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret", environment_name="main")],
    timeout=86400,
)
def train() -> None:
    cmd: list[str] = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={TRAINING_FILES_PATH}",
        f"data.val_files={VALIDATION_FILES_PATH}",
        f"data.train_batch_size={BATCH_SIZE}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={MODEL}",
        f"actor_rollout_ref.actor.optim.lr={LEARNING_RATE}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={MINI_BATCH_SIZE}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=5",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=verl_grpo_example_qwq32b",
        "trainer.experiment_name=qwq32b_example",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=5",
        "trainer.test_freq=5",
        "trainer.total_epochs=15",
        f"trainer.default_local_dir={CHECKPOINTS_PATH}",
        "trainer.resume_mode=auto",
        # for the custom reward function
        f"custom_reward_function.path={PATH_TO_REWARD_FUNCTION}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]
    subprocess.run(cmd, check=True)


# We define a local entrypoint for kicking off the training function.
@app.local_entrypoint()
def main():
    train.remote()


# You can now run the training using `modal run <filename.py>`. By default, results are in wandb
