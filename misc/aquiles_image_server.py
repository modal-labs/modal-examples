# Run Aquiles-Image API server with FLUX.1-Krea-dev on Modal

"""
[Aquiles-Image](https://github.com/Aquiles-ai/Aquiles-Image) is a production-ready API server that brings state-of-the-art image generation
models to your applications. Built on FastAPI and Diffusers, it provides an OpenAI-compatible
interface for generating and editing images using models like FLUX, Stable Diffusion 3.5, and more.

This example shows how to deploy an Aquiles-Image server on Modal using the FLUX.1-Krea-dev model,
providing a simple REST API for generating images from text prompts on Modal's GPU infrastructure.
"""

import os

import modal

# ## Set up the container image
#
# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
#
# We start with an NVIDIA CUDA base image and install the necessary dependencies:
# - Git and build tools for installing packages from source
# - PyTorch 2.8
# - Diffusers
# - Transformers and tokenizers for text processing
# - Aquiles-Image from GitHub for the optimized API server
#
# Aquiles-Image provides 3x faster inference compared to vanilla implementations
# through advanced optimizations and efficient model loading strategies.

aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential",)
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .uv_pip_install(
        "torch==2.8",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==4.57.3",
        "tokenizers==0.22.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers from Hugging Face
        "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "") # HuggingFace token to download the models if you don't have them available in Modal secrets
    })
)

# ## Select the model
#
# We'll be running the FLUX.1-Krea-dev model from Black Forest Labs.
# This is a powerful text-to-image diffusion model that produces high-quality images.
#
# You can swap this model out for any of the compatible models below by changing the string.
#
# Note: Larger models may require more VRAM. A single H100 GPU has 80GB of VRAM,
# which is sufficient for most models listed above.

MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"

# ## Cache model weights and configuration
#
# Although Aquiles-Image will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
#
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access
# like it's a regular disk. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

# ## Build the Aquiles-Image server and serve it
#
# The function below spawns an Aquiles-Image instance listening at port 5500.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.
#
# The server runs in an independent process, via `subprocess.Popen`, and only starts
# accepting requests once the model is loaded and the `serve` function returns.

app = modal.App("aquiles-image-server")

N_GPU = 1
MINUTES = 60  # seconds
AQUILES_PORT = 5500


@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"H100:{N_GPU}",
    scaledown_window=6 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=5
)
@modal.web_server(port=AQUILES_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    # Configure the Aquiles-Image server command with:
    # - Host and port settings for network access
    # - Model name and inference steps
    # - API key for authentication
    # - Device map to use CUDA GPU
    cmd = [
        "aquiles-image",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(AQUILES_PORT),
        "--model",
        MODEL_NAME,
        "--set-steps", "35",  # number of diffusion steps (higher = better quality, slower)
        "--api-key", "dummy-api-key",  # set your own API key for production
        "--device-map", "cuda",  # use GPU acceleration
    ]

    print(f"Starting Aquiles-Image with the model: {MODEL_NAME}")
    print(f"Command: {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server
#
# To deploy the API on Modal, just run:
# ```bash
# modal deploy aquiles_image_server.py
# ```
#
# This will create a new app on Modal, build the container image for it
# if it hasn't been built yet, and deploy the app.

# ## Interact with the server
#
# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--aquiles-image-server-serve.modal.run`.
#
# The server provides an **OpenAI-compatible API**, making it a drop-in replacement for
# OpenAI's image generation endpoints. You can use the official `openai` library
# to interact with it, making integration seamless with existing code.
#
# ### Using the OpenAI Python library
#
# ```python
# # pip install openai
# from openai import OpenAI
# import base64
#
# client = OpenAI(
#     base_url="https://your-workspace-name--aquiles-image-server-serve.modal.run",
#     api_key="dummy-api-key"  # use the same key configured in the server
# )
#
# prompt = "A vast futuristic city curving upward into the sky, its buildings bending and connecting overhead in a continuous loop."
#
# result = client.images.generate(
#     model="black-forest-labs/FLUX.1-Krea-dev",
#     prompt=prompt,
#     size="1024x1024",
#     response_format="b64_json"
# )
#
# # Save the generated image
# image_bytes = base64.b64decode(result.data[0].b64_json)
# with open("output.png", "wb") as f:
#     f.write(image_bytes)
# ```
#
# ### Using curl
#
# You can also send POST requests directly using curl:
# ```bash
# curl -X POST https://your-url.modal.run/images/generations \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer dummy-api-key" \
#   -d '{
#     "model": "black-forest-labs/FLUX.1-Krea-dev",
#     "prompt": "A beautiful sunset over mountains",
#     "size": "1024x1024"
#   }'
# ```
#
# ### Additional Endpoints
#
# Aquiles-Image provides multiple endpoints depending on your use case:
# - `/images/generations` - Generate new images from text prompts
# - `/images/edits` - Edit existing images with text guidance
# - `/videos` - Generate videos from text prompts (experimental, Wan2.2 model)
#
# For full API documentation, visit the `/docs` route of your deployed server:
# `https://your-workspace-name--aquiles-image-server-serve.modal.run/docs`

# ## Testing the server
#
# To make it easier to test the server setup, we include a `local_entrypoint`
# that generates a test image and saves it locally.
#
# If you execute the command:
# ```bash
# modal run aquiles_image_server.py
# ```
#
# a fresh replica of the server will be spun up on Modal while
# the code executes on your local machine.
#
# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!

@app.local_entrypoint()
async def test():
    import base64

    from openai import OpenAI

    url = serve.get_web_url()

    print(f"Server is available at: {url}\n")

    # Create OpenAI client pointing to our Modal server
    client = OpenAI(base_url=url, api_key="dummy-api-key")

    prompt = """A vast futuristic city curving upward into the sky, its buildings bending
        and connecting overhead in a continuous loop. Gravity shifts seamlessly along
        the curve, with sunlight streaming across inverted skyscrapers. The scene feels
        serene and awe-inspiring—earthlike fields and rivers running along the inner
        surface of a colossal rotating structure."""

    print(f"Generating image with prompt:\n{prompt}\n")

    # Generate image using OpenAI-compatible API
    result = client.images.generate(
        model=MODEL_NAME,
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json"
    )

    print("Downloading image...\n")

    # Save the generated image
    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("output.png", "wb") as f:
        f.write(image_bytes)

    print("Image saved successfully as 'output.png'!")
