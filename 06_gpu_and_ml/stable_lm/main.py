import base64
import os
import time
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel

import modal

with open("requirements.txt", "r") as f:
    requirements = base64.b64encode(f.read().encode("utf-8")).decode("utf-8")


def build_models():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = snapshot_download(
        "stabilityai/stablelm-tuned-alpha-7b",
        ignore_patterns=["*.md"],
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    m.save_pretrained(
        model_path, safe_serialization=True, max_shard_size="24GB"
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)
    [p.unlink() for p in Path(model_path).rglob("*.bin")]  # type: ignore


image = (
    modal.Image.conda()
    .apt_install("git", "software-properties-common", "wget")
    .conda_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        "rust=1.69.0",
        channels=["nvidia", "pytorch", "conda-forge"],
    )
    .env(
        {
            "HF_HOME": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "BITSANDBYTES_NOWELCOME": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    .run_commands(
        f"echo '{requirements}' | base64 --decode > /root/requirements.txt",
        "pip install -r /root/requirements.txt",
        gpu="A10G",
    )
    .run_function(
        build_models,
        gpu=None,
        timeout=3600,
    )
)

stub = modal.Stub(
    name="example-stability-lm",
    image=image,
    secrets=[modal.Secret({"REPO_ID": "stabilityai/stablelm-tuned-alpha-7b"})],
)


class CompletionRequest(BaseModel):
    prompt: str = ""
    model: Optional[str] = "stabilityai/stablelm-tuned-alpha-7b"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 16
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = [
        "<|USER|>",
        "<|ASSISTANT|>",
        "<|SYSTEM|>",
        "<|padding|>",
        "<|endoftext|>",
    ]
    top_k: Optional[int] = 1000
    do_sample: Optional[bool] = True


class CompletionResponse(BaseModel):
    text: str = ""


@stub.cls(gpu="A10G")
class StabilityLM:
    def __init__(self, model_url: str = "stabilityai/stablelm-tuned-alpha-7b"):
        self.model_url = model_url
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    def generate(self, completion_request: CompletionRequest):
        text = format_prompt(completion_request.prompt)
        result = self.generator(
            text,
            temperature=completion_request.temperature,
            max_new_tokens=completion_request.max_tokens,
            top_p=completion_request.top_p,
            top_k=completion_request.top_k,
            do_sample=completion_request.do_sample,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=self.generator.tokenizer.convert_tokens_to_ids(
                completion_request.stop
            ),
        )
        return {"text": result[0]["generated_text"].replace(text, "")}


def format_prompt(instruction: str) -> str:
    return f"<|USER|>{instruction}<|ASSISTANT|>"


@stub.function()
@modal.web_endpoint(method="POST")
def completions(completion_request: CompletionRequest) -> CompletionResponse:
    return StabilityLM().generate.call(completion_request=completion_request)


@stub.local_entrypoint()
def main():
    q_style, q_end = "\033[1m", "\033[0m"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world.",
        "How can I tell apart female and male red cardinals?",
    ]
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=512) for q in instructions
    ]
    for q, a in zip(
        instructions, list(StabilityLM().generate.map(instruction_requests))
    ):
        print(f"{q_style}{q}{q_end}\n{a['text']}\n\n")
