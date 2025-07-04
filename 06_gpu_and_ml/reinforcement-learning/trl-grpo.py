# ---
# output-directory: "/tmp/playdiffusion"
# cmd: ["modal", "run", "06_gpu_and_ml/audio-editing/playdiffusion-model.py"]
# args: ["--audio-url", "https://modal-public-assets.s3.us-east-1.amazonaws.com/mono_44100_127389__acclivity__thetimehascome.wav", "--output-text", "November, '9 PM. I'm standing in alley. After waiting several hours, the time has come. A man with long dark hair approaches. I have to act and fast before he realizes what has happened. I must find out.", "--output-path", "/tmp/playdiffusion/output.wav"]
# ---

# # Run GRPO on Modal using TRL

# This example demonstrates how to run [GRPO](https://arxiv.org/pdf/2402.03300) on modal using the TRL [GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
# GRPO is a reinforcement learning algorithm introduced by DeepSeek, and was used to train DeepSeek R1.
# TRL is a reinforcement learning training library by Huggingface

## Basic TRL example

# First we import modal and then defining the app
import modal
app = modal.App("grpo-trl-example")

# We define an image where we install the TRL library
# We also install vllm for the next part of this example. We also use wandb for logging
image = modal.Image.debian_slim().pip_install("trl[vllm]==0.19.0", "datasets==3.5.1", "wandb==0.17.6")

# In reinforcement learning, we define a reward function for the model
# We define a simple reward function here
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


# We import the relevant libraries and kick off training. We use the tldr dataset of reddit posts
with image.imports():
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

# We use wandb for logging, hence we use a [modal secret](https://modal.com/docs/guide/secrets#secrets) with wandb credentials
@app.function(image = image, gpu = "H100", timeout = 86400, secrets = [modal.Secret.from_name("wandb-secret")])
def train():
    dataset = load_dataset("trl-lib/tldr", split="train")
    training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", report_to="wandb") # comment if don't want to use wandb
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

# To run: `modal run --detach trl-grpu.py``
    
## Speeding up training with vLLM

# vLLM can be used either in server mode (run vLLM server on separate gpu) or colocate mode (within the training process)

# @app.function(image = image, gpu = "H100", timeout = 86400, secrets = [modal.Secret.from_name("wandb-secret")])
# def train_vllm_server_mode():
#     dataset = load_dataset("trl-lib/tldr", split="train")
#     training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")
#     trainer = GRPOTrainer(
#         model="Qwen/Qwen2-0.5B-Instruct",
#         reward_funcs=reward_len,
#         args=training_args,
#         train_dataset=dataset,
#         use_vllm=True,
#         vllm_mode="server"
#     )


# @app.function(image = image, gpu = "H100", timeout = 86400, secrets = [modal.Secret.from_name("wandb-secret")])
# def train_vllm_colocate_mode():
#     dataset = load_dataset("trl-lib/tldr", split="train")
#     training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")
#     trainer = GRPOTrainer(
#         model="Qwen/Qwen2-0.5B-Instruct",
#         reward_funcs=reward_len,
#         args=training_args,
#         train_dataset=dataset,
#         use_vllm=True,
#         vllm_mode="colocate"
#     )


