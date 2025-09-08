# ---
# lambda-test: false  # deprecated
# ---
# # Structured output generation with Jsonformer
#
# [Jsonformer](https://github.com/1rgs/jsonformer) is a tool that generates structured synthetic data using LLMs.
# You provide a JSON spec and it generates a JSON object following the spec. It's a
# great tool for developing, benchmarking, and testing applications.


from typing import Any

import modal

# We will be using one of [Databrick's Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
# models, choosing for the smallest version with 3B parameters. Feel free to use any of the other models
# available from the [Huggingface Hub Dolly repository](https://huggingface.co/databricks).
MODEL_ID: str = "databricks/dolly-v2-3b"
CACHE_PATH: str = "/root/cache"


# ## Build image and cache model
#
# We'll download models from the Huggingface Hub and store them in our image.
# This skips the downloading of models during inference and reduces cold boot
# times.
def download_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, use_cache=True, device_map="auto"
    )
    model.save_pretrained(CACHE_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, use_cache=True)
    tokenizer.save_pretrained(CACHE_PATH, safe_serialization=True)


# Define our image; install dependencies.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "jsonformer==0.9.0",
        "transformers",
        "torch",
        "accelerate",
        "safetensors",
    )
    .run_function(download_model)
)
app = modal.App("example-jsonformer-generate")


# ## Generate examples
#
# The generate function takes two arguments `prompt` and `json_schema`, where
# `prompt` is used to describe the domain of your data (for example, "plants")
# and the schema contains the JSON schema you want to populate.
@app.function(gpu="A10G", image=image)
def generate(prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
    from jsonformer import Jsonformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        CACHE_PATH, use_cache=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, use_cache=True, device_map="auto"
    )

    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()

    return generated_data


# Add Modal entrypoint for invoking your script, and done!
@app.local_entrypoint()
def main():
    prompt = "Generate random plant information based on the following schema:"
    json_schema = {
        "type": "object",
        "properties": {
            "height_cm": {"type": "number"},
            "bearing_fruit": {"type": "boolean"},
            "classification": {
                "type": "object",
                "properties": {
                    "species": {"type": "string"},
                    "kingdom": {"type": "string"},
                    "family": {"type": "string"},
                    "genus": {"type": "string"},
                },
            },
        },
    }

    result = generate.remote(prompt, json_schema)
    print(result)
