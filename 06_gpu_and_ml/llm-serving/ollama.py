# ---
# ---

# # Run open-source LLMs with Ollama on Modal

# [Ollama](https://ollama.com/) is a popular tool for running open-source large language models (LLMs) locally.
# It provides a simple API, including OpenAI compatibility, allowing you to interact with various models like
# Llama, Mistral, Phi, and more.

# In this example, we demonstrate how to run Ollama on Modal's cloud infrastructure, leveraging:
#
# 1. Modal's powerful GPU resources that far exceed what's available on most local machines
# 2. Serverless design that scales to zero when not in use (saving costs)
# 3. Persistent model storage using Modal Volumes
# 4. Web-accessible endpoints that expose Ollama's OpenAI-compatible API

# Since the Ollama server provides its own REST API, we use Modal's web_server decorator
# to expose these endpoints directly to the internet.

import asyncio
import subprocess
from typing import List

import modal

# ## Configuration and Constants

# Directory for Ollama models within the container and volume
MODEL_DIR = "/ollama_models"

# Define the models we want to work with
# You can specify different model versions using the format "model:tag"
MODELS_TO_DOWNLOAD = ["llama3.1:8b", "llama3.3:70b"]  # Downloaded at startup
MODELS_TO_TEST = ["llama3.1:8b", "llama3.3:70b"]  # Tested in our example

# Ollama version to install - you may need to update this for the latest models
OLLAMA_VERSION = "0.6.5"
# Ollama's default port - we'll expose this through Modal
OLLAMA_PORT = 11434

# ## Building the Container Image

# First, we create a Modal Image that includes Ollama and its dependencies.
# We use the official Ollama installation script to set up the Ollama binary.

ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .pip_install(
        "fastapi==0.115.8",
        "uvicorn[standard]==0.34.0",
        "openai~=1.30",  # Pin OpenAI version for compatibility
    )
    .run_commands(
        "echo 'Installing Ollama...'",
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        "echo 'Ollama installed at $(which ollama)'",
        f"mkdir -p {MODEL_DIR}",
    )
    .env(
        {
            # Configure Ollama to serve on its default port
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,  # Tell Ollama where to store models
        }
    )
)

# Create a Modal App, which groups our functions together
app = modal.App("example-ollama", image=ollama_image)

# ## Persistent Storage for Models

# We use a Modal Volume to cache downloaded models between runs.
# This prevents needing to re-download large model files each time.

model_volume = modal.Volume.from_name("ollama-models-store", create_if_missing=True)

# ## The Ollama Server Class

# We define an OllamaServer class to manage the Ollama process.
# This class handles:
# - Starting the Ollama server
# - Downloading required models
# - Exposing the API via Modal's web_server
# - Running test requests against the served models


@app.cls(
    gpu="H100",  # Use H100 GPUs for best performance
    volumes={MODEL_DIR: model_volume},  # Mount our model storage
    timeout=60 * 5,  # 5 minutes max input runtime
    min_containers=1,  # Keep at least one container running for fast startup
)
class OllamaServer:
    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Starts the Ollama server and ensures required models are downloaded."""
        print("Starting Ollama setup...")

        print(f"Starting Ollama server on port {OLLAMA_PORT}...")
        cmd = ["ollama", "serve"]
        self.ollama_process = subprocess.Popen(cmd)
        print(f"Ollama server started with PID: {self.ollama_process.pid}")

        # Wait for server to initialize
        await asyncio.sleep(10)
        print("Ollama server should be ready.")

        # --- Model Management ---
        # Check which models are already downloaded, and pull any that are missing
        loop = asyncio.get_running_loop()
        models_pulled = False  # Track if we pulled any model

        # Get list of currently available models
        ollama_list_proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        )

        if ollama_list_proc.returncode != 0:
            print(f"Error: 'ollama list' failed: {ollama_list_proc.stderr}")
            raise RuntimeError(
                f"Failed to list Ollama models: {ollama_list_proc.stderr}"
            )

        current_models_output = ollama_list_proc.stdout
        print("Current models detected:", current_models_output)

        # Download each requested model if not already present
        for model_name in MODELS_TO_DOWNLOAD:
            print(f"Checking for model: {model_name}")
            model_tag_to_check = (
                model_name if ":" in model_name else f"{model_name}:latest"
            )

            if model_tag_to_check not in current_models_output:
                print(
                    f"Model '{model_name}' not found. Pulling (output will stream directly)..."
                )
                models_pulled = True  # Mark that a pull is happening

                # Pull the model - this can take a while for large models
                pull_process = await asyncio.create_subprocess_exec(
                    "ollama",
                    "pull",
                    model_name,
                )

                # Wait for the pull process to complete
                retcode = await pull_process.wait()

                if retcode != 0:
                    print(f"Error pulling model '{model_name}': exit code {retcode}")
                else:
                    print(f"Model '{model_name}' pulled successfully.")
            else:
                print(f"Model '{model_name}' already exists.")

            # Commit the volume only if we actually pulled new models
            if models_pulled:
                print("Committing model volume...")
                await loop.run_in_executor(None, model_volume.commit)
                print("Volume commit finished.")

        print("Ollama setup complete.")

    @modal.exit()
    def stop_ollama(self):
        """Terminates the Ollama server process on shutdown."""
        print("Shutting down Ollama server...")
        if self.ollama_process and self.ollama_process.poll() is None:
            print(f"Terminating Ollama server (PID: {self.ollama_process.pid})...")
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=10)
                print("Ollama server terminated.")
            except subprocess.TimeoutExpired:
                print("Ollama server kill required.")
                self.ollama_process.kill()
                self.ollama_process.wait()
            except Exception as e:
                print(f"Error shutting down Ollama server: {e}")
        else:
            print("Ollama server process already exited or not found.")
        print("Shutdown complete.")

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=180)
    def serve(self):
        """
        Exposes the Ollama server's API endpoints through Modal's web_server.

        This is the key function that makes Ollama's API accessible over the internet.
        The web_server decorator maps Modal's HTTPS endpoint to Ollama's internal port.
        """
        print(f"Serving Ollama API on port {OLLAMA_PORT}")

    # ## Running prompt tests
    #
    # The following method allows us to run test prompts against our Ollama models.
    # This is useful for verifying that the models are working correctly and
    # to see how they respond to different types of prompts.

    @modal.method()
    async def run_tests(self):
        import openai
        from openai.types.chat import ChatCompletionMessageParam

        """
        Tests the Ollama server by sending various prompts to each configured model.
        Returns a dictionary of results organized by model.
        """
        print("Running tests inside OllamaServer container...")
        all_results = {}  # Store results per model

        # Configure OpenAI client to use our Ollama server
        base_api_url = f"http://localhost:{OLLAMA_PORT}/v1"
        print(f"Configuring OpenAI client for: {base_api_url}")
        client = openai.AsyncOpenAI(
            base_url=base_api_url,
            api_key="not-needed",  # Ollama doesn't require API keys
        )

        # Define some test prompts
        test_prompts = [
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about a cat watching rain.",
            "What are the main benefits of using Python?",
        ]

        # Test each model with each prompt
        for model_name in MODELS_TO_TEST:
            print(f"\n===== Testing Model: {model_name} =====")
            model_results = []
            all_results[model_name] = model_results

            for prompt in test_prompts:
                print(f"\n--- Testing Prompt ---\n{prompt}\n----------------------")

                # Create message in OpenAI format
                messages: List[ChatCompletionMessageParam] = [
                    {"role": "user", "content": prompt}
                ]

                try:
                    # Call the Ollama API through the OpenAI client
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                    )
                    assistant_message = response.choices[0].message.content
                    print(f"Assistant Response:\n{assistant_message}")
                    model_results.append(
                        {
                            "prompt": prompt,
                            "status": "success",
                            "response": assistant_message,
                        }
                    )
                except Exception as e:
                    print(
                        f"Error during API call for model '{model_name}', prompt '{prompt}': {e}"
                    )
                    model_results.append(
                        {"prompt": prompt, "status": "error", "error": str(e)}
                    )

        print("Internal tests finished.")
        return all_results


# ## Running the Example

# This local entrypoint function provides a simple way to test the Ollama server.
# When you run `modal run ollama.py`, this function will:
# 1. Start an OllamaServer instance in the cloud
# 2. Run test prompts against each configured model
# 3. Print a summary of the results


@app.local_entrypoint()
async def local_main():
    """
    Tests the Ollama server with sample prompts and prints the results.

    Run with: `modal run ollama.py`
    """
    print("Triggering test suite on the OllamaServer...")
    all_test_results = await OllamaServer().run_tests.remote.aio()
    print("\n--- Test Suite Summary ---")

    if all_test_results:
        for model_name, results in all_test_results.items():
            print(f"\n===== Results for Model: {model_name} =====")
            successful_tests = 0
            if results:
                for result in results:
                    print(f"Prompt: {result['prompt']}")
                    print(f"Status: {result['status']}")
                    if result["status"] == "error":
                        print(f"Error: {result['error']}")
                    else:
                        successful_tests += 1
                    print("----")
                print(
                    f"\nSummary for {model_name}: Total tests: {len(results)}, Successful: {successful_tests}"
                )
            else:
                print("No results returned for this model.")
    else:
        print("No results returned from test function.")

    print("\nTest finished. Your Ollama server is ready to use!")


# ## Deploying to Production
#
# While the local entrypoint is great for testing, for production use you'll want to deploy
# this application persistently. You can do this with:
#
# ```bash
# modal deploy ollama.py
# ```
#
# This creates a persistent deployment that:
#
# 1. Provides a stable URL endpoint for your Ollama API
# 2. Keeps at least one container warm for fast responses
# 3. Scales automatically based on usage
# 4. Preserves your models in the persistent volume between invocations
#
# After deployment, you can find your endpoint URL in your Modal dashboard.
#
# You can then use this endpoint with any OpenAI-compatible client by setting:
#
# ```
# OPENAI_API_BASE=https://your-endpoint-url
# OPENAI_API_KEY=any-value  # Ollama doesn't require authentication
# ```
