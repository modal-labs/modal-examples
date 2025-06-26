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
num_processes: 3
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

training_script = '''\
import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.

Example:
<think>
Let's submit the answer.
</think>
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=3
)

model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=2
training_args.per_device_train_batch_size=8
training_args.num_generations=8
training_args.gradient_accumulation_steps=2

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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

@app.function(gpu="H100:4", image=image, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
    )
def math_group_verifier():
    import subprocess
    import time
    import os

    import wandb

    wandb.init(project="math-rl")
    wandb.config = {"epochs": 10}


    os.makedirs("tmp", exist_ok=True)

    with open("./tmp/zero3.yaml", "w") as f:
        f.write(YAML_CONFIG)

    with open("./tmp/train.py", "w") as f:
        f.write(training_script)
    
    model_name = "willcb/Qwen3-0.6B"

    vllm_proc = subprocess.Popen(
    "export CUDA_VISIBLE_DEVICES=0 && "
    "export NCCL_CUMEM_ENABLE=0 && "
    f"vf-vllm --model {model_name} --port 8000 --enforce-eager",
    shell=True,
)
    train_proc = subprocess.Popen(
    "export CUDA_VISIBLE_DEVICES=1,2,3 && "
    "export NCCL_DEBUG=INFO && "
    "export NCCL_CUMEM_ENABLE=0 && "
    "accelerate launch --config-file tmp/zero3.yaml tmp/train.py",
    shell=True,
)



    train_proc.wait()
    vllm_proc.terminate()
    vllm_proc.wait()

@app.local_entrypoint()
def main():
    math_group_verifier.remote()