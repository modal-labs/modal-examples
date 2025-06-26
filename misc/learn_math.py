import modal

app = modal.App(name="math-rl")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

YAML_CONFIG = """
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

training_script = '''\
import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

# env 1: gsm8k
def gsm8k_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric1 = vf.Rubric(
    funcs=[
        gsm8k_answer_reward_func,
        parser.get_format_reward_func()
    ],
    weights=[1.0, 0.2]
)

dataset1 = load_example_dataset("gsm8k", split="train").select(range(1000))
env1 = vf.SingleTurnEnv(
    dataset=dataset1,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric1,
)

# env 2: math
def math_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric2 = vf.Rubric(
    funcs=[
        math_answer_reward_func,
        parser.get_format_reward_func()
    ],
    weights=[1.0, 0.2]
)

dataset2 = load_example_dataset("math", split="train").select(range(1000))
env2 = vf.SingleTurnEnv(
    dataset=dataset2,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric2,
)

vf_env = vf.EnvGroup([env1, env2], env_names=["gsm8k", "math"])

model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size     = 16
training_args.num_generations                = 16
training_args.gradient_accumulation_steps    = 8
training_args.num_iterations                 = 1
training_args.max_prompt_length              = 512
training_args.max_completion_length          = 2048
training_args.max_steps                      = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
'''

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "clang")
    .pip_install( 
        "ninja",
        "packaging",
        "wheel",
        "vllm==0.8.5",
    )
    .run_commands("pip install 'verifiers[all]'")
    .run_commands("pip install flash-attn --no-build-isolation")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def gsm8k_answer_reward_func(completion, answer, parser, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0


def math_answer_reward_func(completion, answer, parser, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

@app.function(gpu="H100:4", image=image, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },)
def math_group_verifier():
    import verifiers as vf
    from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer
    import subprocess
    import time
    import os

    os.makedirs("tmp", exist_ok=True)

    with open("./tmp/zero3.yaml", "w") as f:
        f.write(YAML_CONFIG)

    with open("./tmp/train.py", "w") as f:
        f.write(training_script)
    
    model_name = "willcb/Qwen3-0.6B"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  
    vllm_proc = subprocess.Popen(["vf-vllm", "--model", model_name, "--port", "8000", "--enforce-eager"], env=env)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    env["NCCL_DEBUG"] = "INFO"
    train_proc = subprocess.Popen([
        "accelerate", "launch",
        "--config-file",   "tmp/zero3.yaml",
        "tmp/train.py",
    ], env=env)


    train_proc.wait()
    vllm_proc.terminate()
    vllm_proc.wait()

@app.local_entrypoint()
def main():
    math_group_verifier.remote()