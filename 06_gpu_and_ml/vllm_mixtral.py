# # Fast inference with vLLM (Mixtral 8x7B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# We are running the [Mixtral 8x7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model here, which is a mixture-of-experts model finetuned for conversation.
# You can expect 3 minute second cold starts
# For a single request, the throughput is about 11 tokens/second, but there are upcoming `vLLM` optimizations to improve this.
# The larger the batch of prompts, the higher the throughput (up to about 300 tokens/second).
# For example, with the 60 prompts below, we can produce 30k tokens in 100 seconds.
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

from modal import Image, Stub, gpu, method

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GPU_CONFIG = gpu.A100(memory=80, count=2)


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns="*.pt", # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Dockerhub image recommended by `vLLM`, and use
# run_function to run the function defined above to ensure the weights of
# the model are saved within the container image.

VLLM_HASH = "89523c8293bc02a4dfaaa80079a5347dc3952464a33a501d5de329921eea7ec7"

image = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.2.5", "huggingface_hub==0.19.4", "hf-transfer==0.1.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, timeout=60 * 20)
)

stub = Stub("example-vllm-inference", image=image)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean. There are, however, some
# outstanding issues and performance improvements that we patch here, such as multi-GPU setup and
# suboptimal Ray CPU pinning.
@stub.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        if GPU_CONFIG.count > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = "<s> [INST] {user} [/INST] "

        # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
        if GPU_CONFIG.count > 1:
            import subprocess

            RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
            subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)

    @method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=1024,
            repetition_penalty=1.1,
        )

        t0 = time.time()
        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        async for output in result_generator:
            if "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta

        print(f"Generated {num_tokens} tokens in {time.time() - t0:.2f}s")


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q vllm_mixtral.py`. The `q` flag
# enables the text to stream in your local terminal.
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    for question in questions:
        print("Sending new request:", question)
        for text in model.completion_stream.remote_gen(question):
            print(text, end="", flush=True)


# ## Deploy and invoke the model
# Once we deploy this model with `modal deploy text_generation_inference.py`,
# we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-Mixtral-8x7B-Instruct-v0.1", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```

# ## Coupling a frontend web application
#
# We can stream inference from a FastAPI backend, also deployed on Modal.
#
# You can try our deployment [here](https://modal-labs--vllm-mixtral.modal.run).

from pathlib import Path

from modal import Mount, asgi_app

frontend_path = Path(__file__).parent / "llm-frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app(label="vllm-mixtral")
def app():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().completion_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": BASE_MODEL + " (vLLM)",
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().completion_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app
