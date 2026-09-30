# ---
# args: ["--smoke-test"]
# ---

# # Make LLMs better at math
#
# This example trains [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
# to solve problems from [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
# with [Miles](https://github.com/radixark/miles) and Group Relative Policy
# Optimization (GRPO).
#
# We use [Clustered Functions](https://modal.com/docs/guide/multi-node-training)
# to run distributed training on two nodes, each with eight H100 GPUs.
#
# ![Training reward, AIME correctness, step timing, and response truncation across 100 updates](https://modal-cdn.com/cdnbot/qwen3-4b-fsdp-8k-100-updates-labeled_4c365b52.png)
#
# Run this example with:
#
# ```bash
# modal run --detach 14_clusters/miles_grpo.py
# ```
#
# The default runs 100 training updates; our run took about three hours on
# two eight-GPU H200 nodes with the image and inputs cached.

import json
import os
import shlex
import socket
import subprocess
import time
import uuid
from pathlib import Path

import modal

app = modal.App("example-miles-grpo")
MINUTES = 60  # seconds
TRAINING_TIMEOUT = 12 * 60 * MINUTES
N_NODES = 2
GPUS_PER_NODE = 8
DATA = Path("/data")
RESULTS = Path("/results")
MILES = Path("/opt/miles")
MILES_COMMIT = "2806267d060d51b1d3b62f85a1f9b145047aeef9"  # v0.1.1

# ## Download the model and datasets
#
# We download the model and datasets before training so that we don't waste
# GPU time. A [Modal Volume](https://modal.com/docs/guide/volumes) caches them
# for subsequent runs. FSDP loads the Hugging Face checkpoint directly.
# The math verifier requires a boxed final answer. DAPO prompts request this
# format; we append the same instruction to the AIME evaluation prompts.

data_volume = modal.Volume.from_name("example-miles-grpo-data", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "example-miles-grpo-results-v2", create_if_missing=True, version=2
)
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.34.4")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
SOURCES = [
    ("Qwen/Qwen3-4B", "model", "1cfa9a7208912126459214e8b04321603b3df60c"),
    ("zhuzilin/dapo-math-17k", "dataset", "2e65612930298bde4c5d58fd97b3f23a483aaff9"),
    ("zhuzilin/aime-2024", "dataset", "1c625e328db94ec7ef7ff169016b097c468d60b9"),
]


@app.function(image=download_image, volumes={DATA: data_volume}, timeout=20 * MINUTES)
def download():
    from huggingface_hub import snapshot_download

    for repo_id, repo_type, revision in SOURCES:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=DATA / repo_id.split("/")[-1],
        )
    aime = DATA / "aime-2024"
    with (
        (aime / "aime-2024.jsonl").open() as source,
        (aime / "aime-2024-boxed.jsonl").open("w") as prepared,
    ):
        for line in source:
            sample = json.loads(line)
            sample["prompt"][-1]["content"] += (
                "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
            prepared.write(json.dumps(sample) + "\n")
    data_volume.commit()


# ## Define the container image
#
# A [Modal Image](https://modal.com/docs/guide/images) lets us use the Miles
# release image, which includes PyTorch, SGLang, FlashAttention, and Ray.

image = (
    modal.Image.from_registry(
        "radixark/miles:v0.1.1@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf"
    )
    .entrypoint([])
    .run_commands(
        f"git clone --depth 1 --branch v0.1.1 https://github.com/radixark/miles.git {MILES}",
        f"cd {MILES} && git checkout {MILES_COMMIT} && pip install --no-deps -e .",
    )
    .env(
        {
            "PYTHONPATH": f"{MILES}:/root/Megatron-LM",
            "PYTHONUNBUFFERED": "1",
            "NCCL_NVLS_ENABLE": "0",
        }
    )
)

# ## Configure our training recipe
#
# These settings adapt the Miles [Qwen3-4B recipe](https://github.com/radixark/miles/blob/v0.1.1/scripts/run_qwen3_4b.py)
# to FSDP with thinking enabled. SGLang generates eight answers per question;
# the `math` reward checks them against the labels. GRPO uses the relative
# rewards within each group to update the model with FSDP, then Miles sends
# the weights back to SGLang.
# Truncated importance sampling (TIS) corrects for differences between training
# and inference log probabilities.
#
# We cap generated responses at 8,192 tokens for both training and evaluation;
# prompt tokens are additional. This bounds generation time, but can cut off
# long solutions. The results below track correctness and truncation together.
#
# The colocated engines exchange CUDA tensors through IPC. We disable
# expandable allocator segments because their IPC path requires `pidfd_getfd`.


def training_args(
    run_dir: Path, num_rollout: int, eval_samples: int, smoke_test: bool = False
) -> list[str]:
    batch_size = 8 if smoke_test else 32
    return shlex.split(
        f"""
        --train-backend fsdp
        --hf-checkpoint {DATA}/Qwen3-4B
        --ref-load {DATA}/Qwen3-4B
        --prompt-data {DATA}/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt --label-key label
        --apply-chat-template --rollout-shuffle --balance-data
        --apply-chat-template-kwargs '{{"enable_thinking": true}}'
        --rm-type math
        --num-rollout {num_rollout}
        --rollout-batch-size {batch_size} --n-samples-per-prompt 8
        --rollout-max-response-len 8192 --rollout-temperature 1
        --global-batch-size {batch_size * 8}
        --eval-interval 10
        --eval-prompt-data aime {DATA}/aime-2024/aime-2024-boxed.jsonl
        --n-samples-per-eval-prompt {eval_samples}
        --eval-max-response-len 8192 --eval-top-p 1
        --advantage-estimator grpo --use-tis --tis-clip 2.0 --tis-clip-low 0.0
        --use-kl-loss --kl-loss-coef 0 --kl-loss-type low_var_kl
        --kl-coef 0 --entropy-coef 0
        --eps-clip 0.2 --eps-clip-high 0.28
        --optimizer adam --lr 1e-6 --lr-decay-style constant
        --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
        --rollout-num-gpus-per-engine 1
        --sglang-decode-log-interval 1000
        --sglang-mem-fraction-static 0.75
        --sglang-attention-backend fa3 --sglang-chunked-prefill-size 4096
        --update-weight-buffer-size 536870912
        --gradient-checkpointing --attn-implementation flash_attention_2
        --train-env-vars '{{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}}'
        --use-dynamic-batch-size --max-tokens-per-gpu 32768
        --actor-num-nodes {N_NODES} --actor-num-gpus-per-node {GPUS_PER_NODE}
        --colocate --use-fault-tolerance
        --save {run_dir}/checkpoints --save-interval 10
        --use-tensorboard --tb-project-name miles --tb-experiment-name {run_dir.name}
        """
    )


# ## Start training
#
# Clustered Functions schedule both nodes together and enable RDMA networking with a decorator.
# Rank zero starts the Ray head and launches training once all 16 GPUs join.
#
# `ray start` returns after starting background processes. The worker waits
# on a Queue so its Function stays alive until the head finishes training.
# Each run uses its own partition; single-use containers release Ray on exit.
# A Volume v2 shares the checkpoint files across all training ranks.

completion_queue = modal.Queue.from_name(
    "example-miles-grpo-completion", create_if_missing=True
)


@app.function(
    image=image,
    gpu=f"H100:{GPUS_PER_NODE}",
    cpu=32,
    memory=262144,
    volumes={DATA: data_volume, RESULTS: results_volume},
    timeout=TRAINING_TIMEOUT,
    single_use_containers=True,
)
@modal.clustered(size=N_NODES, rdma=True)
def train(
    run_id: str,
    num_rollout: int,
    eval_samples: int,
    smoke_test: bool = False,
):
    import ray

    cluster = modal.Cluster.from_context()
    rank = cluster.container_rank()
    ips = cluster.container_ips(family="ipv4")
    head_ip = ips[0]
    node_ip = ips[rank]
    run_dir = RESULTS / run_id
    env = {
        **os.environ,
        "MASTER_ADDR": head_ip,
        "RAY_ADDRESS": f"{head_ip}:6379",
        "TENSORBOARD_DIR": str(run_dir / "tensorboard"),
    }
    start = [
        "ray",
        "start",
        f"--node-ip-address={node_ip}",
        f"--num-gpus={GPUS_PER_NODE}",
        "--num-cpus=32",
        "--disable-usage-stats",
    ]
    succeeded = False
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        data_volume.reload()
        if rank == 0:
            subprocess.run(start + ["--head", "--port=6379"], env=env, check=True)
        else:
            deadline = time.monotonic() + 180
            while True:
                try:
                    with socket.create_connection((head_ip, 6379), timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Ray head did not start within 180 seconds")
                    time.sleep(1)
            subprocess.run(start + [f"--address={head_ip}:6379"], env=env, check=True)
            if not completion_queue.get(partition=run_id, timeout=TRAINING_TIMEOUT):
                raise RuntimeError(
                    "Miles training failed on the head node; see the App logs"
                )
            results_volume.commit()
            return

        ray.init(address=f"{head_ip}:6379")
        deadline = time.monotonic() + 180
        while sum(n["Alive"] for n in ray.nodes()) < N_NODES:
            if time.monotonic() > deadline:
                raise TimeoutError("The second Ray node did not join")
            time.sleep(1)
        resources = ray.cluster_resources()
        expected_gpus = N_NODES * GPUS_PER_NODE
        if resources.get("GPU") != expected_gpus:
            raise RuntimeError(
                f"Expected {expected_gpus} GPUs in the Ray cluster, got {resources}"
            )
        print(f"Ray cluster ready: {resources}", flush=True)
        ray.shutdown()

        subprocess.run(
            [
                "python",
                str(MILES / "train.py"),
                *training_args(run_dir, num_rollout, eval_samples, smoke_test),
            ],
            cwd=MILES,
            env=env,
            check=True,
        )
        results_volume.commit()
        succeeded = True
        return str(run_dir)
    finally:
        if rank == 0:
            completion_queue.put(succeeded, partition=run_id)


# Submit training asynchronously so it survives a detached CLI disconnection.
# Waiting on the result also surfaces errors while the terminal is connected.


@app.local_entrypoint()
def main(num_rollout: int = 100, eval_samples: int = 4, smoke_test: bool = False):
    if smoke_test:
        num_rollout, eval_samples = 1, 1
    if num_rollout < 1 or eval_samples < 1:
        raise ValueError("num-rollout and eval-samples must be positive")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    print(f"Run ID: {run_id}")
    download.remote()
    result = train.spawn(run_id, num_rollout, eval_samples, smoke_test).get()
    print(f"Results saved to {result} in Volume {results_volume.name}")


# ## Inspect the results
#
# Download the TensorBoard events using the run ID printed by the script:
#
# ```bash
# mkdir -p /tmp/miles-grpo/RUN_ID
# modal volume get example-miles-grpo-results-v2 RUN_ID/tensorboard /tmp/miles-grpo/RUN_ID
# uvx tensorboard --logdir /tmp/miles-grpo/RUN_ID
# ```
#
# Once you open the local TensorBoard URL printed by the command, you can inspect:
#
# - `rollout/episode_raw_reward`: fraction of correct answers in each training batch.
# - `eval/aime`: mean correctness on the fixed AIME problems.
# - `rollout/truncated_ratio`: fraction of training responses that reach the token limit.
# - `eval/aime-truncated_ratio`: fraction of evaluation responses that reach the token limit.
# - `perf/rollout_time`, `perf/train_time`: time spent generating and updating the model.
#
# ## Results
#
# The default 100-update run completed in 173 minutes on two nodes with eight
# H200s each, including startup, evaluation, checkpoint saving, and cleanup,
# with the image and inputs cached. Generation averaged 66 seconds per update;
# training averaged 19 seconds after the first update's 64-second compilation
# and training pass. Most of the runtime was spent generating solutions.
# These measurements used H200s; the Function requests H100s, so runtime on
# other allocations may differ.
#
# We evaluate on all 30 AIME 2024 problems before training and every 10 updates,
# sampling four answers per problem. The score is mean answer correctness,
# not the fraction of problems solved at least once (pass@4). The 120 answers
# share 30 problems, so they are not 120 independent test questions.
#
# | Completed updates | AIME correct answers | Responses cut off at 8K |
# | --- | --- | --- |
# | 0 | 42/120 (35.0%) | 87/120 (72.5%) |
# | 10 | 41/120 (34.2%) | 88/120 (73.3%) |
# | 20 | 49/120 (40.8%) | 86/120 (71.7%) |
# | 30 | 44/120 (36.7%) | 82/120 (68.3%) |
# | 40 | 50/120 (41.7%) | 78/120 (65.0%) |
# | 50 | 54/120 (45.0%) | 71/120 (59.2%) |
# | 60 | 49/120 (40.8%) | 78/120 (65.0%) |
# | 70 | 54/120 (45.0%) | 75/120 (62.5%) |
# | 80 | 54/120 (45.0%) | 71/120 (59.2%) |
# | 90 | 57/120 (47.5%) | 67/120 (55.8%) |
# | 100 | 56/120 (46.7%) | 68/120 (56.7%) |
#
# AIME correctness rose from 35.0% to 46.7%, with a peak of 47.5% at update 90.
# Training reward averaged 56.6% over the first ten batches and 70.9% over the
# last ten. Each training batch contains different problems, so individual
# batch rewards fluctuate.
#
# The response budget matters for this thinking model. AIME truncation fell
# from 72.5% to 56.7%; training truncation averaged 48.7% over the first ten
# batches and 25.2% over the last ten. These results show improvement within
# an 8K response budget in one run; they do not establish the effect at longer
# response lengths or across random seeds. More than half of the final AIME
# responses still reached the limit. Increasing the two response limits in
# `training_args` allows longer solutions at the cost of more generation time
# and training memory.

# ## Export a checkpoint for inference
#
# During training, checkpoints are saved using PyTorch's distributed checkpoint
# format. We convert the saved model shards into Hugging Face weights on CPU:
#
# ```bash
# modal run 14_clusters/miles_grpo.py::export --run-id RUN_ID
# modal run 14_clusters/miles_grpo.py::export --run-id RUN_ID --iteration 10
# ```


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    volumes={DATA: data_volume, RESULTS: results_volume},
    timeout=20 * MINUTES,
)
def export(run_id: str, iteration: int = 0):
    if iteration < 0:
        raise ValueError(
            "iteration must be zero (latest) or a positive saved iteration"
        )
    checkpoints = RESULTS / run_id / "checkpoints"
    if iteration == 0:
        iteration = int((checkpoints / "latest_checkpointed_iteration.txt").read_text())
    checkpoint = checkpoints / f"iter_{iteration:07d}"
    output = RESULTS / run_id / "huggingface" / checkpoint.name
    subprocess.run(
        [
            "python",
            str(MILES / "tools/convert_fsdp_to_hf.py"),
            "--input-dir",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--origin-hf-dir",
            str(DATA / "Qwen3-4B"),
        ],
        cwd=MILES,
        check=True,
    )
    results_volume.commit()
    print(f"Hugging Face model saved to {output}")
    return str(output)
