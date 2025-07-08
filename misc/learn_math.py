# ---
# cmd: ["modal", "run", "misc/math_rl.py"]
# ---

# # GRPO Training for Mathematical Reasoning with a Multi-GPU Setup

# This example demonstrates a setup for training mathematical reasoning models using 
# [GRPO (Generalized Reinforcement Learning from Process Optimization)](https://arxiv.org/abs/2402.07647) 
# with multi-GPU coordination on Modal. This implementation uses the [verifiers library](https://github.com/willccbb/verifiers) which 
# is a set of tools and abstractions for training LLMs with reinforcement learning in verifiable multi-turn environments via GRPO.
# This example shows how to use the verifiers library to train a model to solve mathematical problems on modal.


# In this example, we will show how to:
# - Use multiple GPUs for training
# - Use VLLM for inference during training
# - Use Modal volumes for caching
# - Run inference after training
# - Use Weights & Biases for logging


import modal
from verifiers.utils import load_example_dataset
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE  

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

TOOL_DESCRIPTIONS = """
- sandbox_exec: Execute Python code to perform calculations
"""

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
    
    print("Training completed! Running a single inference from test set...")
    
    dataset = (
        load_example_dataset("math", split="test")
        .shuffle(seed=42)
        .select(range(1))
    )
    example = dataset[0]
    question = example["question"]
    prompt = DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS) + "\n\nProblem: " + question + "\n\n<think>\n\n<answer>"

    inference.remote(prompt)

@app.function(gpu="H100", image=image, volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=600,
    )
def inference(prompt: str):
    """Test the trained model with the same format as training"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import json
    import subprocess
    
    # Load the trained model
    print("Loading trained model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{WEIGHTS_DIR}", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"{WEIGHTS_DIR}", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        print("✓ Loaded trained model from weights volume")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Loading base model instead...")
        model_name = "willcb/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE_DIR, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def sandbox_exec(code):
        """Execute Python code in a subprocess"""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            
            output = result.stdout.strip() if result.stdout else ""
            if len(output) > 1000:
                output = output[:1000] + "... (truncated to 1000 chars)"
            
            return output
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"


    def generate_response(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt_text):].strip()
    
    print("="*50)
    print("TESTING TRAINED MODEL")
    print("="*50)
    
    # Initial generation
    model_response = generate_response(prompt + "\n\n<think>\n\n<answer>")
    print("MODEL RESPONSE:")
    print(model_response)
    print("-" * 30)
    return model_response


@app.local_entrypoint()
def main(mode: str = "train", prompt: str = None, prompt_file: str = None):
    if mode == "inference":
        if prompt_file:
            try:
                with open(prompt_file, 'r') as f:
                    prompt_text = f.read().strip()
                print(f"Using prompt from file: {prompt_file}")
            except FileNotFoundError:
                print(f"Error: File {prompt_file} not found")
                return
        elif prompt:
            prompt_text = prompt
            print("Using prompt from command line argument")
        else:
            default_question = "Find the value of x that satisfies the equation: 2x + 5 = 17"
            prompt_text = DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS) + "\n\nProblem: " + default_question + "\n\n<think>\n\n<answer>"
        
        inference.remote(prompt_text)
    elif mode == "train":
        math_group_verifier.remote()