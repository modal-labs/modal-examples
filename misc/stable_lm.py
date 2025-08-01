# # Run StableLM text completion model

# This example shows how you can run [`stabilityai/stablelm-tuned-alpha-7b`](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b) on Modal

import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Union

import modal
from pydantic import BaseModel
from typing_extensions import Annotated, Literal


def build_models():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = snapshot_download(
        "stabilityai/stablelm-tuned-alpha-7b",
        ignore_patterns=["*.md"],
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    m.save_pretrained(model_path, safe_serialization=True, max_shard_size="24GB")
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)
    [p.unlink() for p in Path(model_path).rglob("*.bin")]  # type: ignore


image = (
    modal.Image.micromamba()
    .apt_install("git", "software-properties-common", "wget")
    .micromamba_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        "rust=1.69.0",
        channels=["nvidia", "pytorch", "conda-forge"],
    )
    .env(
        {
            "HF_HOME": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "BITSANDBYTES_NOWELCOME": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    .pip_install(
        "transformers~=4.28.1",
        "safetensors==0.3.0",
        "accelerate==0.18.0",
        "bitsandbytes==0.38.1",
        "msgspec==0.18.6",
        "sentencepiece==0.1.98",
        "hf-transfer==0.1.3",
        gpu="any",
    )
    .run_function(
        build_models,
        gpu=None,
        timeout=3600,
    )
)

app = modal.App(
    name="example-stability-lm",
    image=image,
    secrets=[
        modal.Secret.from_dict({"REPO_ID": "stabilityai/stablelm-tuned-alpha-7b"})
    ],
)


class CompletionRequest(BaseModel):
    prompt: Annotated[str, "The prompt for text completion"]
    model: Annotated[
        Literal["stabilityai/stablelm-tuned-alpha-7b"],
        "The model to use for text completion",
    ] = "stabilityai/stablelm-tuned-alpha-7b"
    temperature: Annotated[
        float,
        "Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.",
    ] = 0.8
    max_tokens: Annotated[
        int, "Maximum number of new tokens to generate for text completion."
    ] = 16
    top_p: Annotated[
        float,
        "Probability threshold for the decoder to use in sampling next most likely token.",
    ] = 0.9
    stream: Annotated[
        bool, "Whether to stream the generated text or return it all at once."
    ] = False
    stop: Annotated[Union[str, List[str]], "Any additional stop words."] = []
    top_k: Annotated[
        int,
        "Limits the set of tokens to consider for next token generation to the top k.",
    ] = 40
    do_sample: Annotated[
        bool, "Whether to use sampling or greedy decoding for text completion."
    ] = True


@app.cls(gpu="A10G")
class StabilityLM:
    stop_tokens = [
        "<|USER|>",
        "<|ASSISTANT|>",
        "<|SYSTEM|>",
        "<|padding|>",
        "<|endoftext|>",
    ]
    model_url: str = modal.parameter(default="stabilityai/stablelm-tuned-alpha-7b")

    @modal.enter()
    def setup_model(self):
        """
        Container-lifeycle method for model setup.
        """
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoTokenizer, TextIteratorStreamer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_url, local_files_only=True)
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_tokens)
        self.streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model_url,
            tokenizer=tokenizer,
            streamer=self.streamer,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={"local_files_only": True},
        )
        self.generator.model = torch.compile(self.generator.model)

    def get_config(self, completion_request: CompletionRequest) -> Dict[str, Any]:
        return dict(
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=list(
                set(
                    self.generator.tokenizer.convert_tokens_to_ids(
                        self.generator.tokenizer.tokenize(
                            "".join(completion_request.stop)
                        )
                    )
                    + self.stop_ids
                )
            ),
            max_new_tokens=completion_request.max_tokens,
            **completion_request.dict(
                exclude={"prompt", "model", "stop", "max_tokens", "stream"}
            ),
        )

    def generate_completion(
        self, completion_request: CompletionRequest
    ) -> Generator[str, None, None]:
        import re
        from threading import Thread

        from transformers import GenerationConfig

        text = format_prompt(completion_request.prompt)
        gen_config = GenerationConfig(**self.get_config(completion_request))
        stop_words = self.generator.tokenizer.convert_ids_to_tokens(
            gen_config.eos_token_id
        )
        stop_words_pattern = re.compile("|".join(map(re.escape, stop_words)))
        thread = Thread(
            target=self.generator.__call__,
            kwargs=dict(text_inputs=text, generation_config=gen_config),
        )
        thread.start()
        for new_text in self.streamer:
            if new_text.strip():
                new_text = stop_words_pattern.sub("", new_text)
                yield new_text
        thread.join()

    @modal.method()
    def generate(self, completion_request: CompletionRequest) -> str:
        return "".join(self.generate_completion(completion_request))

    @modal.method()
    def generate_stream(self, completion_request: CompletionRequest) -> Generator:
        for text in self.generate_completion(completion_request):
            yield text


def format_prompt(instruction: str) -> str:
    return f"<|USER|>{instruction}<|ASSISTANT|>"


with app.image.imports():
    import uuid

    import msgspec

    class Choice(msgspec.Struct):
        text: str
        index: Union[int, None] = 0
        logprobs: Union[int, None] = None
        finish_reason: Union[str, None] = None

    class CompletionResponse(msgspec.Struct, kw_only=True):  # type: ignore
        id: Union[str, None] = None
        object: str = "text_completion"
        created: Union[int, None] = None
        model: str
        choices: List[Choice]

        def __post_init__(self):
            if self.id is None:
                self.id = str(uuid.uuid4())
            if self.created is None:
                self.created = int(time.time())


@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)  # Interactive docs at /docs
async def completions(completion_request: CompletionRequest):
    from fastapi import Response, status
    from fastapi.responses import StreamingResponse

    response_id = str(uuid.uuid4())
    response_utc = int(time.time())

    if not completion_request.stream:
        return Response(
            content=msgspec.json.encode(
                CompletionResponse(
                    id=response_id,
                    created=response_utc,
                    model=completion_request.model,
                    choices=[
                        Choice(
                            index=0,
                            text=StabilityLM().generate.remote(
                                completion_request=completion_request
                            ),
                        )
                    ],
                )
            ),
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    def wrapped_stream():
        for new_text in StabilityLM().generate_stream.remote(
            completion_request=completion_request
        ):
            yield (
                msgspec.json.encode(
                    CompletionResponse(
                        id=response_id,
                        created=response_utc,
                        model=completion_request.model,
                        choices=[Choice(index=0, text=new_text)],
                    )
                )
                + b"\n\n"
            )

    return StreamingResponse(
        content=wrapped_stream(),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
    )


@app.local_entrypoint()
def main():
    q_style, q_end = "\033[1m", "\033[0m"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world.",
        "How can I tell apart female and male red cardinals?",
    ]
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in instructions
    ]
    print("Running example non-streaming completions:\n")
    for q, a in zip(
        instructions, list(StabilityLM().generate.map(instruction_requests))
    ):
        print(f"{q_style}{q}{q_end}\n{a}\n\n")

    print("Running example streaming completion:\n")
    for part in StabilityLM().generate_stream.remote_gen(
        CompletionRequest(
            prompt="Generate a list of ten sure-to-be unicorn AI startup names.",
            max_tokens=128,
            stream=True,
        )
    ):
        print(part, end="", flush=True)


# ```bash
# curl $MODEL_APP_ENDPOINT \
#   -H "Content-Type: application/json" \
#   -d '{
#     "prompt": "Generate a list of 20 great names for sentient cheesecakes that teach SQL",
#     "stream": true,
#     "max_tokens": 64
#   }'
# ```
