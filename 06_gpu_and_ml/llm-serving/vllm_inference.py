# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/llm-serving/vllm_inference.py"]
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# You can find a video walkthrough of this example and the related scripts on the Modal YouTube channel
# [here](https://www.youtube.com/watch?v=QmY_7ePR1hM).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `pip`.

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

# ## Download the model weights

# We'll be running a pretrained foundation model -- Meta's LLaMA 3.1 8B
# in the Instruct variant that's trained to chat and follow instructions,
# quantized to 4-bit by [Neural Magic](https://neuralmagic.com/) and uploaded to Hugging Face.

# You can read more about the `w4a16` "Machete" weight layout and kernels
# [here](https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/).

MODELS_DIR = "/llamas"
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"

# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes)
# as a cache, which act as a "shared disk" that all Functions can access.
hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ## Build a vLLM engine and serve it

# We start a vLLM server as a subprocess, which Modal has [first-class support for](https://modal.com/docs/reference/modal.web_server). We configure Modal
# to forward requests to port 8000.

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model. vLLM will authenticate requests
# using the API key we provide it.

app = modal.App("example-vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "super-secret-key"  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_inference.py
# ```

# This will create a new app on Modal, build the container image for it, and deploy.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands. They also demonstrate authentication.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

# To interact with the API programmatically, you can use the Python `openai` library.

# See the `client.py` script in the examples repository
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible)
# to take it for a spin:

# ```bash
# # pip install openai==1.13.3
# python openai_compatible/client.py
# ```

# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatibl):

# ```bash
# modal run openai_compatible/load_test.py
# ```
