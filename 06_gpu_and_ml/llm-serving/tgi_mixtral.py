# ---
# deploy: true
# ---
# # Hosting Mixtral 8x7B with Text Generation Inference (TGI)
#
# In this example, we show how to run an optimized inference server using [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
# with performance advantages over standard text generation pipelines including:
# - continuous batching, so multiple generations can take place at the same time on a single container
# - PagedAttention, which applies memory paging to the attention mechanism's key-value cache, increasing throughput
#
# This example deployment, [accessible here](https://modal-labs--tgi-mixtral.modal.run), can serve Mixtral 8x7B on two 80GB A100s, with
# up to 500 tokens/s of throughput and per-token latency of 78ms.

# ## Setup
#
# First we import the components we need from `modal`.

import subprocess
from pathlib import Path

from modal import App, Image, Mount, asgi_app, enter, exit, gpu, method

# Next, we set which model to serve, taking care to specify the GPU configuration required
# to fit the model into VRAM, and the quantization method (`bitsandbytes` or `gptq`) if desired.
# Note that quantization does degrade token generation performance significantly.
#
# Any model supported by TGI can be chosen here.

GPU_CONFIG = gpu.A100(size="40GB", count=4)
MODEL_ID = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
MODEL_REVISION = "286ae6737d048ad1d965c2e830864df02db50f2f"
# Add `["--quantize", "gptq"]` for TheBloke GPTQ models.
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--revision",
    MODEL_REVISION,
    "--port",
    "8000",
]

# ## Define a container image
#
# We want to create a Modal image which has the Hugging Face model cache pre-populated.
# The benefit of this is that the container no longer has to re-download the model from Huggingface -
# instead, it will take advantage of Modal's internal filesystem for faster cold starts.
# The 95GB model can be loaded in as little as 70 seconds.
#
# ### Download the weights
# We can use the included utilities to download the model weights (and convert to safetensors, if necessary)
# as part of the image build.


def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ]
    )


# ### Image definition
# We’ll start from a Docker Hub image recommended by TGI, and override the default `ENTRYPOINT` for
# Modal to run its own which enables seamless serverless deployments.
#
# Next we run the download step to pre-populate the image with our model weights.
#
# Finally, we install the `text-generation` client to interface with TGI's Rust webserver over `localhost`.

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.3.3")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        timeout=60 * 20,
    )
    .pip_install("text-generation")
)

app = App("example-tgi-mixtral")


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).
# The class syntax is a special representation for a Modal function which splits logic into two parts:
# 1. the `@enter()` function, which runs once per container when it starts up, and
# 2. the `@method()` function, which runs per inference request.
#
# This means the model is loaded into the GPUs, and the backend for TGI is launched just once when each
# container starts, and this state is cached for each subsequent invocation of the function.
# Note that on start-up, we must wait for the Rust webserver to accept connections before considering the
# container ready.
#
# Here, we also
# - specify how many A100s we need per container
# - specify that each container is allowed to handle up to 10 inputs (i.e. requests) simultaneously
# - keep idle containers for 10 minutes before spinning down
# - lift the timeout of each request.


@app.cls(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        self.template = "[INST] {user} [/INST]"

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        webserver_ready = False
        while not webserver_ready:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                webserver_ready = True
                print("Webserver ready!")
            except (socket.timeout, ConnectionRefusedError):
                # If launcher process exited, a connection can never be made.
                if retcode := self.launcher.poll():
                    raise RuntimeError(f"launcher exited with code {retcode}")
                time.sleep(1.0)

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def generate(self, question: str):
        prompt = self.template.format(user=question)
        result = await self.client.generate(prompt, max_new_tokens=1024)

        return result.generated_text

    @method()
    async def generate_stream(self, question: str):
        prompt = self.template.format(user=question)

        async for response in self.client.generate_stream(
            prompt, max_new_tokens=1024
        ):
            if not response.token.special:
                yield response.token.text


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to invoke
# our remote function. You can run this script locally with `modal run text_generation_inference.py`.
@app.local_entrypoint()
def main():
    print(
        Model().generate.remote(
            "Implement a Python function to compute the Fibonacci numbers."
        )
    )


# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal-labs--tgi-mixtral.modal.run).

frontend_path = Path(__file__).parent.parent / "llm-frontend"


@app.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app(label="tgi-mixtral")
def tgi_mixtral():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().generate_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": MODEL_ID + " (TGI)",
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().generate_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


# ## Invoke the model from other apps
# Once the model is deployed, we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-mixtral", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```
