# ---
# pytest: false
# ---

# # Detect PII with a fine-tuned Phi-3 Mini on Modal

# This example shows how to serve
# [ab-ai/PII-Model-Phi3-Mini](https://huggingface.co/ab-ai/PII-Model-Phi3-Mini)
# on Modal for detecting personally identifiable information (PII) in text.

# The model is a fine-tuned version of
# [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
# trained to recognize 50+ PII entity types including names, emails, SSNs,
# credit card numbers, addresses, and more.
# Given an input text, it returns a JSON object mapping entity types to their detected values.

# We serve it with [vLLM](https://docs.vllm.ai/) behind an OpenAI-compatible API
# and include a `local_entrypoint` for quick testing from the command line.

# ## Set up the container image

# vLLM can be installed with `uv pip`
# since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).
# The model requires `trust_remote_code`, so we also install a compatible
# version of `transformers`.

import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "transformers>=4.51",
    )
)

# ## Configure the model

# The PII model is ~4B parameters (BF16), so it fits comfortably on a single T4 GPU.
# We cache the model weights in a [Modal Volume](https://modal.com/docs/guide/volumes)
# to avoid re-downloading them on each cold start.

MODEL_NAME = "ab-ai/PII-Model-Phi3-Mini"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## The PII detection prompt

# The model was fine-tuned with a specific prompt structure
# that lists all 50+ detectable entity types and asks for JSON output.

PII_ENTITIES = ", ".join(
    [
        "companyname",
        "pin",
        "currencyname",
        "email",
        "phoneimei",
        "litecoinaddress",
        "currency",
        "eyecolor",
        "street",
        "mac",
        "state",
        "time",
        "vehiclevin",
        "jobarea",
        "date",
        "bic",
        "currencysymbol",
        "currencycode",
        "age",
        "nearbygpscoordinate",
        "amount",
        "ssn",
        "ethereumaddress",
        "zipcode",
        "buildingnumber",
        "dob",
        "firstname",
        "middlename",
        "ordinaldirection",
        "jobtitle",
        "bitcoinaddress",
        "jobtype",
        "phonenumber",
        "height",
        "password",
        "ip",
        "useragent",
        "accountname",
        "city",
        "gender",
        "secondaryaddress",
        "iban",
        "sex",
        "prefix",
        "ipv4",
        "maskednumber",
        "url",
        "username",
        "lastname",
        "creditcardcvv",
        "county",
        "vehiclevrm",
        "ipv6",
        "creditcardissuer",
        "accountnumber",
        "creditcardnumber",
    ]
)

SYSTEM_PROMPT = (
    "### Instruction:\n\n"
    "Identify and extract the following PII entities from the text, "
    f"if present: {PII_ENTITIES}. Return the output in JSON format."
)


def build_pii_prompt(text: str) -> str:
    """Format input text into the model's expected prompt structure."""
    return f"{SYSTEM_PROMPT}\n\n### Input:\n{text}\n\n### Output:\n"


# ## Serve the model

# We use [`@modal.web_server`](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to expose vLLM's OpenAI-compatible API on the internet.

app = modal.App("example-pii-detection")

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="T4",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--enforce-eager",
    ]

    print(*cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy

# To deploy a persistent API endpoint:
# ```bash
# modal deploy pii_detection.py
# ```

# Once deployed, you can hit the OpenAI-compatible API at the URL printed in the terminal.
# The model expects a raw text prompt via the `/v1/completions` endpoint
# (not `/v1/chat/completions`), since the PII prompt format
# doesn't use chat-style messages.

# ## Test locally

# The `local_entrypoint` spins up a fresh replica on Modal and sends
# a sample PII detection request from your local machine:
# ```bash
# modal run pii_detection.py
# ```

# You can also pass custom text:
# ```bash
# modal run pii_detection.py --text "Call John Smith at john@example.com"
# ```

SAMPLE_TEXT = (
    "Hi Abner, just a reminder that your next primary care appointment "
    "is on 23/03/1926. Please confirm by replying to this email "
    "Nathen15@hotmail.com or call 555-123-4567."
)


@app.local_entrypoint()
async def main(text: str = SAMPLE_TEXT, test_timeout: int = 10 * MINUTES):
    url = await serve.get_web_url.aio()

    prompt = build_pii_prompt(text)

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Checking server health at {url} ...")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Server is healthy.\n")

        print(f"Detecting PII in:\n  {text}\n")

        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.1,
        }

        async with session.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

        output = result["choices"][0]["text"]
        print("Detected PII:")
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(output)
