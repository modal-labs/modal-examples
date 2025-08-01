# # Deploy Ollama service on Modal
#
# This example shows how to deploy Ollama (https://ollama.com/) as a Modal web service,
# allowing you to run and interact with open source large language models with GPU acceleration.
#
# ## Overview
#
# This script creates a Modal application that:
#
# 1. Sets up an Ollama server as a web service
# 2. Uses a persistent volume to store model weights
# 3. Provides a method to pull new models to the service
# 4. Automatically scales down when not in use
#
# ## Usage
#
# To run this example:
#
# ```bash
# modal deploy misc/ollama_deployment.py
# ```
#
# To pull a model (e.g., llama3):
#
# ```bash
# modal run misc/ollama_deployment.py::OllamaService.pull_model --model-name llama3
# ```
#
# You can then interact with the Ollama API using standard HTTP requests.

import subprocess
import time

import modal

# Define the base image with Ollama installed
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "systemctl")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
    )
    .pip_install("httpx", "loguru")
    .env(
        {
            "OLLAMA_HOST": "0.0.0.0:11434",  # Configure Ollama to listen on all interfaces
            "OLLAMA_MODELS": "/usr/share/ollama/.ollama/models",  # Set models directory
        }
    )
)

# Create a persistent volume to store model weights. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).
volume = modal.Volume.from_name("ollama-model-weights", create_if_missing=True)

# Create the Modal application
app = modal.App(name="ollama-service", image=image)


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready.

    :param timeout: Maximum time to wait in seconds
    :param interval: Time between checks in seconds
    :raises TimeoutError: If the service doesn't start within the timeout period
    """
    import httpx
    from loguru import logger

    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            logger.info(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


@app.cls(
    scaledown_window=10,  # Automatically scale down after 10 seconds of inactivity
    volumes={
        "/usr/share/ollama/.ollama/models": volume
    },  # Mount volume for model storage
    memory=1024 * 1,  # Allocate 1GB of memory
    gpu="A10G",  # Use A10G GPU for model inference
)
class OllamaService:
    """Main service class that runs Ollama within Modal."""

    @modal.enter()
    def enter(self):
        """Start the Ollama server when the container is created."""
        subprocess.Popen(["ollama", "serve"])

    @modal.method()
    def pull_model(self, model_name: str):
        """Pull a model from Ollama's model hub.

        :param model_name: Name of the model to pull (e.g., "llama3", "mistral")
        """
        subprocess.run(["echo", "pulling model", model_name])
        subprocess.run(["ollama", "pull", model_name])

    @modal.web_server(11434)
    def server(self):
        """Expose the Ollama API as a web server on port 11434."""
        pass
