import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from rich import print
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

import modal


def fetch_models():
    MODEL_PATH = snapshot_download("stabilityai/stablelm-tuned-alpha-7b", ignore_patterns=["*.md"], revision="25071b093c15c0d1cb2b2876c6deb621b764fcf5")


def compile_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_PATH = snapshot_download("stabilityai/stablelm-tuned-alpha-7b", ignore_patterns=["*.md"], revision="25071b093c15c0d1cb2b2876c6deb621b764fcf5")
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, local_files_only=True).cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    m = torch.compile(m)
    m.save_pretrained(MODEL_PATH, safe_serialization=True, max_shard_size="24GB")
    tok.save_pretrained(MODEL_PATH)
    [p.unlink() for p in Path(MODEL_PATH).rglob("*.bin")]


image = (
    modal.Image.conda()
    .apt_install("git", "software-properties-common", "curl", "wget", "libopenblas-dev", "libblas-dev", "g++", "libboost-all-dev", "cmake", "ninja-build", "libpq-dev", "libatlas-base-dev", "gfortran", "libclblast-dev", "libprotobuf-dev", "protobuf-compiler")
    .conda_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        channels=["nvidia", "pytorch"],
    )
    .env({"HF_HOME": "/root", "SAFETENSORS_FAST_GPU": "1", "BITSANDBYTES_NOWELCOME": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_CACHE_DIR": "1"})
    .run_commands(
        "pip install transformers==4.28.1 accelerate==0.18.0 safetensors==0.3.0 bitsandbytes==0.38.1 sentencepiece==0.1.98",
        gpu="A10G"
    )
    .run_function(
        compile_models,
        gpu="A10G",
        timeout=3600,
    )
)

stub = modal.Stub(name="example-stability-lm", image=image)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


@stub.cls(gpu="A10G")
class StabilityLM:
    def __init__(self, model_url: str = "stabilityai/stablelm-tuned-alpha-7b"):
        self.model_url = model_url

    def __enter__(self):
        """
        Container-lifeycle method for model setup.
        """
        import accelerate
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_url,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"Loaded model in {time.time() - start:.2f} seconds.")

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 4096

        self.context_len = context_len
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
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
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
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
    q_style, q_end = "[bold bright_magenta]", "[/bold bright_magenta]"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world..",
        "How can I tell apart female and male red cardinals?",
    ]
    for q, a in zip(instructions, list(StabilityLM().generate.map(instructions))):
        print(f"{q_style}{q}{q_end}\n{a['response']}\n\n")
