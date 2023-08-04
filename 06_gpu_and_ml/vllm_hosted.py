import time
import os
import json

from modal import (
    Stub,
    Mount,
    Image,
    Secret,
    Dict,
    asgi_app,
    web_endpoint,
    method,
    gpu,
)

from pathlib import Path

MODEL_DIR = "/model"


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-13b-chat-hf",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


vllm_image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to 07/21/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@d7a1c6d614756b3072df3e8b52c0998035fb453f"
    )
    .run_function(
        download_model_to_folder, secret=Secret.from_name("huggingface")
    )
)

stub = Stub("llama-demo")
stub.dict = Dict.new()


# vLLM class
@stub.cls(
    gpu=gpu.A100(),
    image=vllm_image,
    allow_concurrent_inputs=60,
    concurrency_limit=1,
    container_idle_timeout=600,
)
class Engine:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        # tokens generated since last report
        self.last_report, self.generated_tokens = time.time(), 0

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            # Only uses 90% of GPU memory by default
            gpu_memory_utilization=0.95,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST] """

    def generated(self, n: int):
        # Log that n tokens have been generated
        t = time.time()
        self.generated_tokens += n
        # Save to dict every second
        if t - self.last_report > 1.0:
            stub.app.dict.update(
                tps=self.generated_tokens / (t - self.last_report),
                t=self.last_report,
            )
            self.last_report, self.generated_tokens = t, 0

    @method()
    async def completion(self, question: str):
        if not question:
            return

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            presence_penalty=0.8,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(
            self.template.format(question), sampling_params, request_id
        )

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            self.generated(new_tokens - tokens)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)


# Front-end functionality
frontend_path = Path(__file__).parent / "vllm-hosted"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=3,
    concurrency_limit=6,
    allow_concurrent_inputs=24,
    timeout=600,
)
@asgi_app()
def app():
    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = Engine().completion.get_current_stats()
        try:
            tps, t = stub.app.dict.get("tps"), stub.app.dict.get("t")
        except KeyError:
            tps, t = 0, 0
        return {
            "backlog": stats.backlog,
            "num_active_runners": stats.num_active_runners,
            "num_total_runners": stats.num_total_runners,
            "tps": tps if t > time.time() - 4.0 else 0,
        }

    @web_app.get("/completion/{question}")
    async def get(question: str):
        from urllib.parse import unquote

        print("Web server received request for", unquote(question))

        # FastAPI will run this in a separate thread
        def generate():
            for chunk in Engine().completion.call(unquote(question)):
                yield f"data: {json.dumps(dict(text=chunk), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app
