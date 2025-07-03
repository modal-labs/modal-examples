'''
Reward after setp 0 and reward at step n
use wanb 

'''

import modal

app = modal.App(name="math-rl")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


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
    .add_local_file("math_rl_trainer.py", "/root/math_rl_trainer.py")
    .add_local_file("math_rl.yaml", "/root/math_rl.yaml")
)

HF_CACHE_DIR = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)

WEIGHTS_DIR = "/root/math_weights"
WEIGHTS_VOL = modal.Volume.from_name("math-rl-weights", create_if_missing=True)

@app.function(gpu="H100:4", image=image, volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        VLLM_CACHE_DIR: VLLM_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret-rl")],
    )
def math_group_verifier():
    import subprocess
    import time
    import os
    import wandb

    wandb.init(project="math-rl")
    wandb.config = {"epochs": 10}

    model_name = "willcb/Qwen3-0.6B"

    vllm_proc = subprocess.Popen(
        "export CUDA_VISIBLE_DEVICES=0 && "
        "export NCCL_CUMEM_ENABLE=0 && "
        f"vf-vllm --model {model_name} --port 8000 --enforce-eager",
        shell=True,
    )
    
    # Wait a bit for VLLM to start
    time.sleep(30)
    
    train_proc = subprocess.Popen(
        "export CUDA_VISIBLE_DEVICES=1,2,3 && "
        "export NCCL_DEBUG=INFO && "
        "export NCCL_CUMEM_ENABLE=0 && "
        "cd /root && python math_rl_trainer.py --config math_rl.yaml",
        shell=True,
    )

    train_proc.wait()
    vllm_proc.terminate()
    vllm_proc.wait()

@app.local_entrypoint()
def main():
    math_group_verifier.remote()