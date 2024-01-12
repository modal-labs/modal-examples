# # Generic model inference with vLLM


import os
import time

from modal import Image, Stub, method, Secret


vllm_image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    )
    .pip_install("vllm==0.2.6", "huggingface_hub==0.20.2", "hf-transfer==0.1.4")
    .env(dict(HUGGINGFACE_HUB_CACHE="/hf-cache", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

stub = Stub("example-vllm-generic", image=vllm_image)


@stub.cls(
    timeout=60 * 60,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=15,
)
class Model:
    def __init__(self, model_name: str):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import torch

        n_gpus = torch.cuda.device_count()

        if n_gpus > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=n_gpus)

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=n_gpus,
            gpu_memory_utilization=0.90,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = "<s> [INST] {user} [/INST] "

    @method()
    async def completion_stream(self, prompt: str):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.85,
            max_tokens=2048,
            repetition_penalty=1.1,
        )

        t0 = time.time()
        request_id = random_uuid()
        result_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        async for output in result_generator:
            if (
                output.outputs[0].text
                and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta

        print(f"Generated {num_tokens} tokens in {time.time() - t0:.2f}s")
