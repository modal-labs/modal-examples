# ---
# args: ["--smoke-test"]
# pytest: false
# ---
#
# # Train coding agents
#
# This example trains a coding agent with reinforcement learning on
# [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) using
# [Spindle](https://modal.com/docs/guide/spindle), Modal's Tinker-compatible API.
#
# During rollouts, the agent reads a repository issue and uses a bash tool to
# inspect and edit files in a [Modal Sandbox](https://modal.com/docs/guide/sandboxes).
#
# This example is inspired by
# [ProRL-Agent-Server's SWE-Gym recipe](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/tree/6a1ead6bfac054fce6c1e62d1a77b330d96c58db/examples/swegym_slime_grpo).
# Accordingly, we'll train our model using Group Relative Policy Optimization (GRPO) on a 31-task subset of SWE-Gym for simplicity, but it's easy to expand the dataset when you're
# ready to train a more general coding agent!
#
# Spindle's multi-tenant backend enables multiple LoRA training jobs to share
# compute, which we will utilize to run concurrent training jobs on a single trainer node.
#
# Some plots from our SWE-Gym run using this recipe:
#
# ![SWE-Gym average reward and full batch time for eight clients](https://modal-cdn.com/examples/swe-gym/qwen3-5-9b-reward-time-17676ad259b6f588.png)
#
# ## Connect to the training server
#
# See the [setup guide](/docs/guide/spindle) for how to deploy your Spindle server.
# Once you have the server's URL and a generated API key, set them:
#
# ```shell
# export TINKER_BASE_URL='https://your-server-url.modal.run'
# export TINKER_API_KEY='your-api-key'
# ```
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "modal==1.5.5",
#   "tinker==0.24.1",
#   "numpy==2.4.6",
#   "transformers[chat-template]==5.17.0",
#   "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7",
# ]
# ///

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ## Choose the training recipe
#
# In our GRPO recipe, we'll admit 128k max context length across 20 turns for our multi-turn RL, with a batch size of 32 groups and 8 prompts per group.
# With DAPO-style filtering, groups where every attempt gets the same reward are replaced, since they provide no policy gradient signal.
#
# In our async RL recipe, rollout and grading workers produce data samples to a queue for the trainer loop to consume from. The rollout workers submit sampling
# requests to our sampling server, the grading workers spin up sandboxes to execute agent code for determining rewards, and the trainer loop calls our trainer
# server's endpoints (`forward_backward` and `optim_step`) to update the policy weights with these generated rollouts and rewards.
#
# The `rollout_workers` and `grading_workers` parameters can be tuned to provide more parallelism, and the `completed_group_buffer` denotes the size of the queue
# that the trainer loop consumes from. Note that these are all on the *client-side* (i.e., within the Tinker SDK), and unrelated to any server-side concurrency.
#
# Below we detail our full training config for this run:

MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

TRAINING = {
    "context_tokens": 131_072,
    "turn_tokens": 8192,
    "max_turns": 20,
    "group_size": 8,
    "groups_per_batch": 32,
    "minibatch_groups": 8,
    "temperature": 1.0,
    "kl_coef": 1e-4,
    "seed": 4242,
    "rollout_workers": 64,
    "inflight_groups": 32,
    "grading_workers": 64,
    "grading_queue_size": 64,
    "completed_group_buffer": 8,
    "filter_mixed_rewards": True,
    "prefetch": True,
}

SMOKE_TRAINING = {
    "context_tokens": 4096,
    "turn_tokens": 512,
    "max_turns": 2,
    "group_size": 2,
    "filter_mixed_rewards": False,
    "prefetch": False,
}

# ## Define a reward function
#
# The `load_tasks()` helper selects 31 issues from the SkyRL SWE-Gym dataset (to use a larger subset, change the task selection in `helpers/dataset.py`
# and `helpers/task_ids.json`). After each attempt, we apply the agent's patch in a fresh Sandbox and run the task's tests. The reward is 1 if the patch
# solves the issue, or 0 otherwise. You can find the grading code in
# [this helper file](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/swe_gym/helpers/rollouts.py).
#
# ## Create a LoRA training client
#
# Pass `"qwen35-9b-lora-128k"` as `base_model` to select the server's 128K-context preset.
# Each job gets its own rank-32 adapter and optimizer state but share the same multi-LoRA trainer node.


def create_training_client(service, base_model, checkpoint=None):
    trainer = service.create_lora_training_client(
        base_model=base_model, rank=32, train_unembed=False
    )
    if checkpoint:
        trainer.load_state_with_optimizer(checkpoint["checkpoint"]).result(timeout=3600)
    return trainer


# ## Publish weights for sampling
#
# Each group needs a fixed policy across all eight attempts and their tool turns.
# A named sampling checkpoint lets an in-progress attempt keep using those
# weights while training updates the adapter.


# ```python
# def publish_policy(trainer, service, name):
#     receipt = trainer.save_weights_for_sampler(name=name).result(timeout=3600)
#     return service.create_sampling_client(model_path=receipt.path)
# ```

# ## Async RL Loop
#
# Because our Spindle server and the Tinker SDK all support async operations as a first-class primitive, we'll show how to implement an async RL loop using the Tinker SDK, which will allow us to interleave
# long multi-turn code-execution rollouts with trainer updates. The high-level idea is to split up rollout producing, grading, and training into separate threads, all of which can run concurrently and communicate with each other via queues.

# ### Rollout generation
#
# In the "producer" side of the loop, we'll generate rollouts for each issue, generating eight attempts using the
# same sampling checkpoint for each task. `generate_rollout()` runs the full multi-turn agent loop:
# sample a command, run it in a Sandbox, and feed the output back to the model.
#
# ```python
# rollouts = [
#     rollout_pool.submit(
#         generate_rollout,
#         task,
#         sampler,
#         tokenizer,
#         environments,
#         cfg,
#         log,
#         {**identity, "attempt": attempt},
#     )
#     for attempt in range(cfg["group_size"])
# ]
# ```

# ### Rollout grading
#
# As each attempt finishes, we then send it to the grading pool to calculate rewards. `grade_rollout()` applies
# its patch in a fresh Sandbox and runs the tests. It gets a reward of 1 if it
# solves the issue, or 0 otherwise. Rollout workers can start new attempts while
# these tests run.
#
# ```python
# graded = [
#     graders.submit(grade_rollout, future.result(), task, environments, log)
#     for future in as_completed(rollouts)
# ]
# completed_groups.put(
#     {
#         "episodes": [future.result() for future in graded],
#         "policy_version": version,
#     }
# )
# ```

# ### Trainer policy update
#
# The trainer pulls groups from the `completed_groups` queue. We use DAPO-style filtering (only keep groups with mixed rewards), and one-step off policy lag (drop groups more than one publication behind the current policy).
#
# ```python
# chosen = []
# while len(chosen) < cfg["groups_per_batch"]:
#     group = completed_groups.get()
#     rewards = [episode["reward"] for episode in group["episodes"]]
#     if 0 < sum(rewards) < len(rewards) and version - group["policy_version"] <= 1:
#         chosen.append(group)
# ```
#
# Lastly, we split the batch into minibatches and update the adapter. `prepare_minibatch()`
# computes group-relative advantages and reference logprobs, masks prompt/tool
# tokens, and normalizes by the minibatch's generated-token count. Then we send to the trainer to update the policy, which completes the "loop"!
#
# ```python
# for minibatch in batched(chosen, cfg["minibatch_groups"]):
#     data = prepare_minibatch(minibatch, reference, cfg["kl_coef"])
#     forward = trainer.forward_backward(
#         data,
#         "ppo",
#         loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
#     )
#     optimizer = trainer.optim_step(
#         types.AdamParams(learning_rate=1e-6, grad_clip_norm=1.0)
#     )
#     forward.result(timeout=7200)
#     optimizer.result(timeout=7200)
#
# trainer.save_state(name).result(timeout=3600)
# sampler = publish_policy(trainer, service, name)
# version += 1
# ```
#
# New rollout groups use the updated sampler; groups already in flight keep their
# original checkpoint. These snippets show the core loop. We compile all the above sections into a single `TrainingJob` class, which also handles checkpointing, timeouts, and shutdown. You can find the implementation
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/swe_gym/helpers/job.py).

# ## Launch the training jobs
#
# The driver creates one `TrainingJob` per LoRA client and runs them concurrently.
# By default, eight jobs share the Spindle server and compute.
#
# Run the example with:
#
# ```shell
# uv run --python 3.12 --script 06_gpu_and_ml/swe_gym/swe_gym.py
# ```
#
# By default, this script runs eight clients with 100 batches each.
# Use `--clients` and `--steps` to change these defaults.
#
# Each completed batch prints the job, step, mean reward, and checkpoint path.
# Summaries are saved locally to `/tmp/swe-gym-results/events.jsonl` by default.
# Use `--output` to choose a different directory.
# To understand an individual attempt, you can also inspect its tool turns, patch, and test
# report under `/tmp/swe-gym-results/episodes/`.


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--groups-per-batch", type=int, default=TRAINING["groups_per_batch"]
    )
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--output", type=Path, default=Path("/tmp/swe-gym-results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.clients, args.steps, args.groups_per_batch = 1, 2, 1
    if min(args.clients, args.steps, args.groups_per_batch, args.hours) <= 0:
        parser.error("clients, steps, groups-per-batch, and hours must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error(
            "Use a new output directory or --resume to preserve existing results"
        )
    cfg = {
        **TRAINING,
        "steps": args.steps,
        "groups_per_batch": args.groups_per_batch,
        "inflight_groups": min(TRAINING["inflight_groups"], args.groups_per_batch),
        "completed_group_buffer": min(
            TRAINING["completed_group_buffer"], args.groups_per_batch
        ),
    }
    if args.smoke_test:
        cfg.update(SMOKE_TRAINING)
    base_model = "swe-gym-smoke" if args.smoke_test else "qwen35-9b-lora-128k"
    log = Log(args.output)
    atomic_json(
        args.output / "config.json",
        {
            **cfg,
            "model": MODEL,
            "revision": MODEL_REVISION,
            "preset": base_model,
            "clients": args.clients,
        },
    )
    tasks = load_tasks()
    if args.smoke_test:
        tasks = [t for t in tasks if t["instance_id"] == "facebookresearch__hydra-1560"]
    environments = Environments()

    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"], api_key=os.environ["TINKER_API_KEY"]
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    reference = service.create_sampling_client(base_model=base_model)
    jobs = []
    for client in range(args.clients):
        checkpoint_file = args.output / f"train-job{client}-checkpoint.json"
        checkpoint = (
            json.loads(checkpoint_file.read_text())
            if args.resume and checkpoint_file.exists()
            else None
        )
        start_step = checkpoint["step"] + 1 if checkpoint else 0
        if start_step >= args.steps:
            continue
        trainer = create_training_client(service, base_model, checkpoint)
        job = TrainingJob(
            client,
            trainer,
            service,
            tokenizer,
            environments,
            tasks,
            cfg,
            log,
            reference,
        )
        jobs.append((job, start_step))

    deadline = time.time() + args.hours * 3600

    phase = f"run-{time.time_ns()}"
    with ThreadPoolExecutor(args.clients) as pool:
        futures = [
            pool.submit(job.run, phase, deadline, start_step)
            for job, start_step in jobs
        ]
        for future in futures:
            future.result()


# ## Addenda
#
# We include the following code for testing purposes.

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "modal==1.5.5",
        "tinker==0.24.1",
        "numpy==2.4.6",
        "transformers[chat-template]==5.17.0",
        "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7",
    )
    .add_local_dir(Path(__file__).parent / "helpers", "/root/helpers")
)

with image.imports():
    import tinker
    from helpers.dataset import load_tasks
    from helpers.environments import Environments
    from helpers.job import TrainingJob
    from helpers.training import Log, atomic_json
    from transformers import AutoTokenizer


app = modal.App("example-swe-gym-training")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=25 * 60,
)
def run_training(
    clients: int, steps: int, groups_per_batch: int, smoke_test: bool = False
):
    argv = [
        "--clients",
        str(clients),
        "--steps",
        str(steps),
        "--groups-per-batch",
        str(groups_per_batch),
    ]
    if smoke_test:
        argv.append("--smoke-test")
    main(argv)


@app.local_entrypoint()
def test(
    clients: int = 1,
    steps: int = 1,
    groups_per_batch: int = 1,
    smoke_test: bool = False,
):
    secret = "spindle-synmon-smoke" if smoke_test else "spindle-synmon"
    run_training.with_options(secrets=[modal.Secret.from_name(secret)]).remote(
        clients, steps, groups_per_batch, smoke_test
    )


if __name__ == "__main__":
    main()
