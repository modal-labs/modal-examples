# # Fast serving with vLLM (Llama 2 13B)
#


import os

from modal import Image, Secret, Stub, method, gpu, web_endpoint

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-13b-chat-hf",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


MODEL_DIR = "/model"

image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to 08/15/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
        "typing-extensions==4.5.0",  # >=4.6 causes typing issues
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub("vllm-serving", image=image)

@stub.cls(gpu=gpu.A100(), secret=Secret.from_name("huggingface"), allow_concurrent_inputs=12, container_idle_timeout=300)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=1,
            # Only uses 90% of GPU memory by default
            gpu_memory_utilization=0.95,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] """

    @method()
    async def generate(self, question: str):
        from vllm import SamplingParams

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        import time

        sampling_params = SamplingParams(
            presence_penalty=0.8,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        prompt = self.template.format(system="", user=question)
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )
        # print(prompt)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            tokens = len(request_output.outputs[0].token_ids)

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        # print(request_output.outputs[0].text)

@stub.function(timeout=60 * 10, allow_concurrent_inputs=12)
@web_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        chain(
            ("Loading model. This usually takes around 30s ...\n\n"),
            Model().generate.remote_gen(question),
        ),
        media_type="text/event-stream",
    )


@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
    ]

    for question in questions:
        tokens = []
        for token_text in model.generate.remote_gen(question):
            tokens.append(token_text)
