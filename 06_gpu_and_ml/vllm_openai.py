import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
from modal import Image, Secret, Stub, method, asgi_app
from uuid import uuid4
import time
from fastapi import FastAPI

app = FastAPI()

MODEL_DIR = "/model"
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# borrowed from vLLM, but using dataclasses instead of pydantic
@dataclass
class ChatCompletionRequest:
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True

@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

@dataclass
class ChatCompletionResponse:
    choices: List[ChatCompletionResponseChoice]
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid4()}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = BASE_MODEL
    usage: Optional[UsageInfo] = None

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
    )

image = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.2.6")
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub("example-vllm-inference", image=image)


@stub.cls(gpu="A100", secret=Secret.from_name("huggingface"), allow_concurrent_inputs=12, container_idle_timeout=300)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise TemplateError(message)
        
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.template = jinja_env.from_string(chat_template)

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=1,
            # Only uses 90% of GPU memory by default
            gpu_memory_utilization=0.95,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @method()
    async def generate(
        self, 
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, float]
    ):
        import time
        from vllm import SamplingParams
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(**sampling_params)
        request_id = random_uuid()
        prompt = self.template.render(
            messages=messages,
            add_generation_prompt=True,
        )
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )
        print(prompt)

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

@app.post("/chat/completions")
def create_completion(
    request: ChatCompletionRequest,
):
    model = Model()
    sampling_params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens if request.max_tokens is not None else 800,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
    }
    messages = request.messages
    tokens = []
    for i, token_text in enumerate(model.generate.remote_gen(
        messages=messages,
        sampling_params=sampling_params
    )):
        tokens.append(token_text)

    return ChatCompletionResponse(
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="".join(tokens)),
                finish_reason="length" if i == len(messages) - 1 else "stop sequence",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(messages),
            total_tokens=len(tokens),
            completion_tokens=len(tokens) - len(messages),
        ),
    )

@stub.function(timeout=60 * 10, allow_concurrent_inputs=12)
@asgi_app()
def fastapi_app():
    return app
    
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
    ]
    sampling_params = {
        "temperature": 0.75,
        "top_p": 1,
        "max_tokens": 800,
        "presence_penalty": 1.15
    }

    # for question in questions:
    #     messages = [
    #         {"role": "user", "content": question},
    #     ]
    #     tokens = []
    #     for token_text in model.generate.remote_gen(
    #         messages=messages,
    #         sampling_params=sampling_params
    #     ):
    #         tokens.append(token_text)
    #     print("".join(tokens))

    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url="https://gongy--example-vllm-inference-fastapi-app-dev.modal.run/",
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model=BASE_MODEL,
    )
    print("Completion result:", chat_completion)
