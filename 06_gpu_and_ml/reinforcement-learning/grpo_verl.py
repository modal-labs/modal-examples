# ---
# cmd: ["modal", "run", "06_gpu_and_ml/reinforcement-learning/grpo_verl.py::train"]
# ---

# # Train a model to solve math problems using GRPO and verl

# This example demonstrates how to train with [GRPO](https://arxiv.org/pdf/2402.03300) on Modal using the [verl](https://github.com/volcengine/verl) framework.
# GRPO is a reinforcement learning algorithm introduced by DeepSeek, and was used to train DeepSeek R1.
# verl is a reinforcement learning training library that is an implementation of [HybridFlow](https://arxiv.org/abs/2409.19256v2), an RLHF framework.

# The training process works as follows:
# - Each example in the dataset corresponds to a math problem.
# - In each training step, the model attempts to solve the math problems showing its steps.
# - We then compute a reward for the model's solution using the reward function defined below.
# - That reward value is then used to update the model's parameters according to the GRPO training algorithm.

# ## Setup

# Import the necessary modules for Modal deployment.
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

import modal

# ## Defining the image and app

app = modal.App("example-grpo-verl")

# We define an image where we clone the verl repo and install its dependencies. We use a base verl image as a starting point.

VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .pip_install("verl[vllm]==0.4.1")
)

# ## Defining the dataset

# In this example, we'll use reinforcement learning to train a model to solve math problems.
# We use the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of math problems and a [Modal Volume](https://modal.com/docs/guide/volumes#volumes) to store the data.

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name(
    "grpo-verl-example-data", create_if_missing=True
)


# We write a Modal Function to populate the Volume with the data. This downloads the dataset and stores it in the Volume.
# You will need to run this step if you don't already have data you'd like to use for this example.
@app.function(image=image, volumes={DATA_PATH: data_volume})
def prep_dataset() -> None:
    subprocess.run(
        [
            "python",
            VERL_REPO_PATH / "examples" / "data_preprocess" / "gsm8k.py",
            "--local_dir",
            DATA_PATH,
        ],
        check=True,
    )


# You can kick off the dataset download with
# `modal run <filename.py>::prep_dataset`

# ## Defining a reward function

# In reinforcement learning, we define a reward function for the model.
# We can define this in a separate file, or in the same file as in this case that we then pass as an argument to verl.
# We use a `default` reward function for GSM8K from the [verl repo](https://github.com/volcengine/verl/blob/v0.1/verl/utils/reward_score/gsm8k.py), modified to return 1.0 if it's a correct answer and 0 otherwise.


def extract_solution(
    solution_str: str, method: Literal["strict", "flexible"] = "strict"
) -> Optional[str]:
    assert method in ["strict", "flexible"]

    if method == "strict":
        # This also tests the formatting of the model
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
            # No reward if there is no answer.
            pass
        else:
            invalid_str: list[str] = ["", "."]
            # Find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


# Reward functions need to follow a [predefined signature.](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)


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


# We then define constants to pass into verl during the training run.
PATH_TO_REWARD_FUNCTION: Path = Path("/root/grpo_verl.py")
REWARD_FUNCTION_NAME: str = "compute_reward"

# ## Kicking off a training run

# We define some more constants for the training run.
MODELS_PATH: Path = Path("/models")
MINUTES: int = 60


# We also define a Volume for storing model checkpoints.
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "grpo-verl-example-checkpoints", create_if_missing=True
)

# Now, we write a Modal Function for kicking off the training run.
# If you wish to use Weights & Biases, as we do in this code, you'll need to create a Weights & Biases [Secret.](https://modal.com/docs/guide/secrets#secrets)


# verl uses Ray under the hood. It creates Ray workers for each step where each Ray worker is a python process and each step is a step in the RL dataflow pipeline.
# verl also keeps a separate control flow process that's independent of this, responsible for figuring out what step in the RL pipeline to execute.
# Each Ray worker gets mapped onto 1 or more GPUs. Depending on the number of GPUs available, Ray will decide what workers go where, or to hold off scheduling workers
# if there are no available GPUs. Generally, more VRAM = less hot-swapping of Ray workers, which means less waiting around for memory copying each iteration.
# In this example we have chosen a configuration that allows for easy automated testing, but you may wish to use more GPUs or more powerful GPU types.
# More details [here](https://verl.readthedocs.io/en/latest/hybrid_flow.html).


@app.function(
    image=image,
    gpu="H100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * MINUTES,
)
def train(*arglist) -> None:
    data_volume.reload()

    cmd: list[str] = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.train_batch_size=128",
        "data.max_prompt_length=64",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "actor_rollout_ref.model.path=Qwen/Qwen2-0.5B",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size=128",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.actor.checkpoint.save_contents='model,optimizer,extra,hf_model'",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=verl_grpo_example_qwen2-0.5b",
        "trainer.experiment_name=qwen2-0.5b_example",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.test_freq=5",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.resume_mode=auto",
        # Parameters chosen to ensure easy automated testing. Remove if needed.
        "trainer.save_freq=1",
        "trainer.total_training_steps=1",
        "trainer.total_epochs=1",
        # For the custom reward function.
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]
    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)


# You can now run the training using `modal run --detach grpo_verl.py::train`, or pass in any [additional args from the CLI](https://modal.com/docs/guide/apps#argument-parsing) like this `modal run --detach grpo.py::train -- trainer.total_epochs=20 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16`.

# ## Performing inference on the trained model

# We use vLLM to perform inference on the trained model.

VLLM_PORT: int = 8000


# Once you have the model checkpoints in your Modal Volume, you can load the weights and perform inference using vLLM. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).
# The weights path is as follows: `global_step_n/actor/huggingface` where n is the checkpoint you want (e.g. `global_step_5/actor/huggingface`).
# The `latest_checkpointed_iteration.txt` file stores the most recent checkpoint index.
def get_latest_checkpoint_file_path():
    with open(MODELS_PATH / "latest_checkpointed_iteration.txt") as f:
        latest_checkpoint_index = int(f.read())
    return str(
        MODELS_PATH / f"global_step_{latest_checkpoint_index}" / "actor" / "huggingface"
    )


# We provide the code for setting up an OpenAI compatible inference endpoint here. For more details re. serving models on vLLM, check out [this example.](https://modal.com/docs/examples/vllm_inference#deploy-the-server)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100:2",
    scaledown_window=15 * MINUTES,  # How long should we stay up with no requests?
    timeout=10 * MINUTES,  # How long should we wait for container start?
    volumes={"/root/.cache/vllm": vllm_cache_vol, MODELS_PATH: checkpoints_volume},
)
@modal.concurrent(
    max_inputs=32
)  # How many requests can one replica handle? Tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    latest_checkpoint_file_path = get_latest_checkpoint_file_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        latest_checkpoint_file_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "2",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# You can then deploy the server using `modal deploy grpo_verl.py`, which gives you a custom URL. You can then query it using the following curl command:

# ```bash
# curl -X POST <url>/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant for solving math problems."},
#       {"role": "user", "content": "James had 4 apples. Mary gave him 2 and he ate 1. How many does he have left?"}
#     ],
#     "temperature": 0.7
#   }'
# ```

# or in the [following ways](https://modal.com/docs/examples/vllm_inference#interact-with-the-server).
