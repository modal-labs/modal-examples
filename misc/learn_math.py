import modal

app = modal.App(name="math-rl")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

run_vllm = """
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000
"""

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


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "clang")
    .pip_install( 
        "ninja",
        "packaging",
        "wheel",
        "vllm",
    )
    .run_commands("pip install 'verifiers[all]'")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env({"VLLM_TARGET_DEVICE": "cuda",
          "VLLM_USE_TRITON_FLASH_ATTN": "True",
          "VLLM_PORT": "8120",
    })
)

def gsm8k_answer_reward_func(completion, answer, parser, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0


def math_answer_reward_func(completion, answer, parser, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

@app.function(gpu="H100", image=image)
def math_group_verifier():
    import verifiers as vf
    from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer
    import os
    import subprocess
    import time
    

    model_name = "willcb/Qwen3-0.6B"

    vllm_process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(vllm_port),
        "--host", "0.0.0.0",
        "--dtype", "float16",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.5",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("Waiting for VLLM server to be ready...")
    time.sleep(10) 

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    rubric1 = vf.Rubric(funcs=[gsm8k_answer_reward_func, parser.get_format_reward_func()], weights=[1.0, 0.2])
    dataset1 = load_example_dataset("gsm8k", split="train").select(range(1000))
    env1 = vf.SingleTurnEnv(dataset=dataset1, system_prompt=system_prompt, parser=parser, rubric=rubric1)

    rubric2 = vf.Rubric(funcs=[math_answer_reward_func, parser.get_format_reward_func()], weights=[1.0, 0.2])
    dataset2 = load_example_dataset("math", split="train").select(range(1000))
    env2 = vf.SingleTurnEnv(dataset=dataset2, system_prompt=system_prompt, parser=parser, rubric=rubric2)

    vf_env = vf.EnvGroup([env1, env2], env_names=["gsm8k", "math"])
    model_name = "willcb/Qwen3-0.6B"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    run_name = "math-group_" + model_name.split("/")[-1].lower()

    training_args=vf.grpo_defaults(run_name=run_name)
    training_args.per_device_train_batch_size=16
    training_args.num_generations=16
    training_args.gradient_accumulation_steps=8
    training_args.num_iterations=1
    training_args.max_prompt_length=512
    training_args.max_completion_length=2048
    training_args.max_steps=100
    training_args.vllm_port=8120

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    
    trainer.train() 

@app.local_entrypoint()
def main():
    math_group_verifier.remote()