# ---
# cmd: ["modal", "run", "06_gpu_and_ml/reinforcement-learning/learn_math.py", "--mode=train", "--run-id=test_run", "--trainer-script=06_gpu_and_ml/reinforcement-learning/trainer_script_grpo.py", "--config-file=06_gpu_and_ml/reinforcement-learning/config_grpo.yaml"]
# ---

# # Training a mathematical reasoning model using the verifiers library with sandboxed code execution

# This example demonstrates how to train mathematical reasoning models on Modal using the [verifiers library](https://github.com/willccbb/verifiers) with [Modal Sandboxes](https://modal.com/docs/guide/sandbox) for executing generated code.
# The [verifiers library](https://github.com/willccbb/verifiers) is a set of tools and abstractions for training LLMs with reinforcement learning in verifiable multi-turn environments via [GRPO](https://arxiv.org/abs/2402.03300).

# This example demonstrates how to:
# - Launch a distributed GRPO training job on Modal with 4× H100 GPUs.
# - Use vLLM for inference during training.
# - Cache HuggingFace, vLLM, and store the model weights in [Volumes](https://modal.com/docs/guide/volumes).
# - Run inference by loading the trained model from Volumes.

# ## Setup
# We start by importing modal and the dependencies from the verifiers library. Then, we create a Modal App and an image with a NVIDIA CUDA base image.
# We install the dependencies for the `verifiers` and `flash-attn` libraries, following the verifiers [README](https://github.com/willccbb/verifiers?tab=readme-ov-file#getting-started).

import modal

app = modal.App(name="example-learn-math")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/"
    "flash_attn-2.7.1.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)  # We use a pre-built binary for flash-attn to install it in the image.

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "clang")
    .uv_pip_install(
        "huggingface_hub[hf_transfer]==0.33.5",
        "setuptools==69.0.3",
        "wheel==0.45.1",
        "ninja==1.11.1.4",
        "packaging==25.0",
        "verifiers[all]==0.1.1",
        flash_attn_release,
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)

# ## Caching HuggingFace, vLLM, and storing model weights. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).
# We create Modal Volumes to persist:
# - HuggingFace downloads
# - vLLM cache
# - Model weights


# We define the model name and a tool that the model can use to execute Python code that it generates.
# See this [this training script](/docs/examples/trainer_script_grpo) for more details.

HF_CACHE_DIR = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)

WEIGHTS_DIR = "/root/math_weights"
WEIGHTS_VOL = modal.Volume.from_name(
    "example-trainer-script-grpo-weights", create_if_missing=True
)

MODEL_NAME = "willcb/Qwen3-0.6B"
TOOL_DESCRIPTIONS = """
- sandbox_exec: Execute Python code to perform calculations.
"""

# ## Training
# Following the [verifiers example](https://github.com/willccbb/verifiers/blob/main/verifiers/examples/math_python.py), we will need a training script and a config file.
# For sandboxed code execution, we will use [this training script](/docs/examples/trainer_script_grpo) and the config file defined [here](https://github.com/willccbb/verifiers/blob/main/configs/zero3.yaml).

# We create a function that uses 4 H100 GPUs and mounts the defined Volumes. Then, we write the training script and the config file to the `/root/` directory.
# We use the `willcb/Qwen3-0.6B` model from HuggingFace, setting up inference via a vLLM server. Once, the model is served, we will launch the training script using `accelerate`.
# We can use the App ID as a unique identifier for saving and loading the model weights.
# When the training is complete, we will run a single inference from the training set to test our training run.


@app.function(
    gpu="H100:4",
    image=image,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        VLLM_CACHE_DIR: VLLM_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret-rl")],
)
def math_group_verifier(trainer_script: str, config_file: str, run_id: str = None):
    import os
    import subprocess

    import wandb
    from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
    from verifiers.utils import load_example_dataset

    with open("/root/trainer_script.py", "w") as f:
        f.write(trainer_script)
    with open("/root/config.yaml", "w") as f:
        f.write(config_file)

    wandb.init(project="example-trainer-script-grpo")
    wandb.config = {"epochs": 10}

    vllm_proc = subprocess.Popen(
        ["vf-vllm", "--model", MODEL_NAME, "--port", "8000", "--enforce-eager"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "NCCL_CUMEM_ENABLE": "0"},
    )

    run_id = app.app_id if run_id is None else run_id

    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--config-file",
            "/root/config.yaml",
            "/root/trainer_script.py",
            "--run-id",
            run_id,
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1,2,3",
            "NCCL_DEBUG": "INFO",
            "NCCL_CUMEM_ENABLE": "0",
        },
    )
    vllm_proc.terminate()
    vllm_proc.wait()

    print("Training completed! Running a single inference from test set...")

    dataset = load_example_dataset(
        "math", split="train"
    ).select(
        range(1)
    )  # We use the first example from the training set for inference to test our training run.

    example = dataset[0]
    question = example["question"]
    prompt = (
        DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS)
        + "\n\nProblem: "
        + question
        + "\n\n<think>\n\n<answer>"
    )

    result = inference.remote(prompt, run_id)
    print(result)


# ## Inference
# We define an `inference` Modal function that runs on a single GPU and mounts the weights volume.
# Then, we load the trained model from the volume, falling back to the base model if necessary.
# To build the prompt, we apply `DEFAULT_TOOL_PROMPT_TEMPLATE` with `TOOL_DESCRIPTIONS` and the problem text.
# Finally, we tokenize the prompt, generate a response with sampling (temperature, top-p, repetition penalty), then decode and return the answer.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=60 * 10,
)
def inference(prompt: str, run_id: str = None):
    """Test the trained model with the same format as training"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

    prompt = (
        DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS)
        + "\n\nProblem: "
        + prompt
        + "\n\n<think>\n\n<answer>"
    )

    model_path = f"{WEIGHTS_DIR}/{app.app_id if run_id is None else run_id}"
    print(f"Loading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"✓ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Loading base model instead...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=HF_CACHE_DIR, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt_text) :].strip()

    model_response = generate_response(prompt + "\n\n<think>\n\n<answer>")
    return model_response


# ## Usage
# We create a main function that serves as the entrypoint for the app.
# It supports two modes:
# - train: kick off math_group_verifier with the provided training script and config file
# - inference: invoke inference with prompt string or prompt file

# To run the training, we can use the following command:
# ```bash
# modal run learn_math.py --mode=train --trainer-script=trainer_script_grpo.py --config-file=config_grpo.yaml
# ```
# To run the inference with a custom prompt, we can use the following command after setting the model path inside our volume:
# ```bash
# modal run learn_math.py --mode=inference --prompt "Find the value of x that satisfies the equation: 2x + 5 = 17" --model-path "test_run"
# ```
# To run the inference with a custom prompt from a file, we can use the following command:
# ```bash
# modal run learn_math.py --mode=inference --prompt-file "prompt.txt"
# ```


@app.local_entrypoint()
def main(
    mode: str = "train",
    prompt: str = None,
    prompt_file: str = None,
    run_id: str = None,
    trainer_script: str = "trainer_script_grpo.py",
    config_file: str = "config_grpo.yaml",
):
    if mode == "inference":
        if prompt_file:
            try:
                with open(prompt_file, "r") as f:
                    prompt_text = f.read().strip()
                print(f"Using prompt from file: {prompt_file}")
            except FileNotFoundError:
                print(f"Error: File {prompt_file} not found")
                return
        elif prompt:
            prompt_text = prompt
            print("Using prompt from command line argument")
        else:
            prompt_text = "Find the value of x that satisfies the equation: 2x + 5 = 17"

        print("=" * 50)
        print("Running inference...")
        print("=" * 50)
        print("PROMPT:")
        print(prompt_text)
        print("-" * 30)
        model_response = inference.remote(prompt_text, run_id)
        print("MODEL RESPONSE:")
        print(model_response)
        print("-" * 30)

    elif mode == "train":
        print(
            f"Training with trainer script:\n{trainer_script}\nand config file:\n{config_file}"
        )
        with open(trainer_script, "r") as f:
            trainer_content = f.read()
        with open(config_file, "r") as f:
            config_content = f.read()

        math_group_verifier.remote(trainer_content, config_content, run_id)
