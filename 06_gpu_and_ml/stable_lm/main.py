import base64
import os
import time
from pathlib import Path

import modal

with open("requirements.txt", "r") as f:
    requirements = base64.b64encode(f.read().encode("utf-8")).decode("utf-8")


def build_models():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = snapshot_download("stabilityai/stablelm-tuned-alpha-7b", ignore_patterns=["*.md"], revision="25071b093c15c0d1cb2b2876c6deb621b764fcf5")
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    m.save_pretrained(model_path, safe_serialization=True, max_shard_size="24GB")
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)
    [p.unlink() for p in Path(model_path).rglob("*.bin")]  # type: ignore


image = (
    modal.Image.conda()
    .apt_install("git", "software-properties-common", "wget")
    .conda_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        channels=["nvidia", "pytorch"],
    )
    .env({"HF_HOME": "/root", "SAFETENSORS_FAST_GPU": "1", "BITSANDBYTES_NOWELCOME": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_CACHE_DIR": "1"})
    .run_commands(
        f"echo '{requirements}' | base64 --decode > /root/requirements.txt",
        "pip install -r /root/requirements.txt",
        gpu="A10G"
    )
    .run_function(
        build_models,
        gpu=None,
        timeout=3600,
    )
)

stub = modal.Stub(name="example-stability-lm", image=image, secrets=[modal.Secret({"REPO_ID": "stabilityai/stablelm-tuned-alpha-7b"})])


@stub.cls(gpu="A10G")
class StabilityLM:
    def __init__(self, model_url: str = "stabilityai/stablelm-tuned-alpha-7b"):
        self.model_url = model_url

    def __enter__(self):
        """
        Container-lifeycle method for model setup.
        """
        import accelerate
        import torch
        from transformers import pipeline

        self.generator = pipeline(
            "text-generation",
            model=self.model_url,
            torch_dtype=torch.float16,
            device_map="auto",
        )


    @modal.method()
    def generate(
        self, text: str, temperature: float = 1.0, max_new_tokens: int = 1024
    ):
        text = format_prompt(text)
        result = self.generator(
            text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            num_beams=1,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            stopping_criteria=[] # TODO: add stopping criteria
        )
        return {"response": result[0]["generated_text"].replace(text, "")}


def format_prompt(instruction):
    return f"<|USER|>{instruction}<|ASSISTANT|>"


@stub.function()
@modal.web_endpoint(method="GET")
def ask(prompt: str, temperature: float = 1.0, max_new_tokens: int = 512):
    return StabilityLM().generate.call(
        prompt, temperature=temperature, max_new_tokens=max_new_tokens
    )


@stub.local_entrypoint()
def main():
    q_style, q_end = "\033[1m", "\033[0m"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world..",
        "How can I tell apart female and male red cardinals?",
    ]
    for q, a in zip(instructions, list(StabilityLM().generate.map(instructions))):
        print(f"{q_style}{q}{q_end}\n{a['response']}\n\n")
