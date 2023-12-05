# ---
# lambda-test: false
# ---

import itertools
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from modal import Function, Image, Mount, Secret, Stub, asgi_app, gpu, method

model = "meta-llama/Llama-2-7b-chat-hf"


def prepare_int4_quantized():
    # TODO(irfansharif): Replace with run_command once we fix
    # https://linear.app/modal-labs/issue/MOD-1955/respect-gpu-specs-for-run-command.
    subprocess.run(
        [
            "python",
            "quantize.py",
            "--checkpoint_path",
            f"checkpoints/{model}/model.pth",
            "--mode",
            "int4",
            "--groupsize",
            "32",
        ],
        check=True,
        cwd="/gpt-fast",
    )


image = (
    Image.from_registry(
        "nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu121",
    )
    .pip_install(
        # Use the barebones hf-transfer package for maximum download speeds. No
        # progress bar, but expect 700MB/s. This combines with the
        # HF_HUB_ENABLE_HF_TRANSFER env var below, see:
        # https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads.
        "hf-transfer~=0.1",
        "huggingface-hub",
        "sentencepiece",
    )
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/pytorch-labs/gpt-fast && cd /gpt-fast && git checkout 3bcaaaf0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        f"cd /gpt-fast && ./scripts/prepare.sh {model}",
        secrets=[Secret.from_name("huggingface")],
        gpu=gpu.A100(memory=80),
    )
    .run_function(
        prepare_int4_quantized,
        gpu=gpu.A100(memory=80),
    )
)

stub = Stub("gpt-fast", image=image)

with stub.image.run_inside():
    import torch
    from sentencepiece import SentencePieceProcessor

    from . import generate
    from .generate import (
        B_INST,
        E_INST,
        _load_model,
        encode_tokens,
    )
    from .tp import maybe_init_dist


@stub.cls(
    gpu=gpu.A100(memory=80),
    timeout=10 * 60,  # 10m
    keep_warm=1,
    container_idle_timeout=20 * 60,  # 20m
)
class Model:
    def __init__(
        self,
        compile_model: bool = True,
        compile_prefill: bool = False,
        use_base_model: bool = False,
        use_speculative_sampling: bool = False,  # NB: takes >10m to initialize, tripping up runners
    ):
        checkpoint = "model.pth" if use_base_model else "model_int8.pth"
        checkpoint_path: Path = Path(
            f"/gpt-fast/checkpoints/{model}/{checkpoint}"
        )
        draft_checkpoint_path: Optional[Path] = None
        if use_speculative_sampling:
            if use_base_model:
                draft_checkpoint_path = Path(
                    f"/gpt-fast/checkpoints/{model}/model_int8.pth"
                )
            else:
                draft_checkpoint_path = Path(
                    f"/gpt-fast/checkpoints/{model}/model_int4.g32.pth"
                )

        self.compile_model = compile_model
        self.compile_prefill = compile_prefill
        self.checkpoint_path = checkpoint_path
        self.draft_checkpoint_path = draft_checkpoint_path

    def __enter__(self):
        assert self.checkpoint_path.is_file(), self.checkpoint_path
        if self.draft_checkpoint_path is not None:
            assert (
                self.draft_checkpoint_path.is_file()
            ), self.draft_checkpoint_path

        global print
        rank = maybe_init_dist()
        use_tp = rank is not None
        if use_tp:
            torch.cuda.set_device(rank)
            if rank != 0:
                # only print on rank 0
                def print(*args, **kwargs):
                    return None

        self.device = "cuda"
        precision = torch.bfloat16
        is_speculative = self.draft_checkpoint_path is not None

        t0 = time.time()
        print("Loading model weights ...")
        model = _load_model(
            self.checkpoint_path, self.device, precision, use_tp
        )

        if is_speculative:
            draft_model = _load_model(
                self.draft_checkpoint_path, self.device, precision, use_tp
            )
        else:
            draft_model = None

        torch.cuda.synchronize()
        print(f"Loading model weights took {time.time() - t0:.02f} seconds")

        if self.compile_model:
            if is_speculative and use_tp:
                torch._inductor.config.triton.cudagraph_trees = (
                    False  # Bug with cudagraph trees in this case
                )

            if is_speculative:
                generate.model_forward = torch.compile(
                    generate.model_forward,
                    mode="reduce-overhead",
                    fullgraph=True,
                )

            generate.decode_one_token = torch.compile(
                generate.decode_one_token,
                mode="reduce-overhead",
                fullgraph=True,
            )

            if self.compile_prefill:
                generate.prefill = torch.compile(
                    generate.prefill, fullgraph=True, dynamic=True
                )

        self.model = model
        self.draft_model = draft_model

        if self.compile_model:
            print("Running warmup inference ...")
            t0 = time.time()
            self.generate_inner(
                "How to print 'hello world' in python?",
                num_samples=1,
                max_new_tokens=100,
                speculate_k=5,
                temperature=0.8,
                top_k=200,
                interactive=True,
                q=queue.Queue(),
                sentinel=object(),
            )
            self.warmed_up = True
            print(f"Warmup inference took {time.time() - t0:.02f} seconds")

    @method()
    def generate(
        self,
        prompt: str,
        num_samples: int = 1,
        max_new_tokens: int = 100,
        speculate_k: int = 5,
        temperature: float = 0.8,
        top_k: int = 200,
        interactive: bool = True,
    ):
        q = queue.Queue()
        sentinel = object()

        # Use a separate thread to generate responses in order to stream them
        # back to the client.
        generation_thread = threading.Thread(
            target=self.generate_inner,
            args=(
                prompt,
                num_samples,
                max_new_tokens,
                speculate_k,
                temperature,
                top_k,
                interactive,
                q,
                sentinel,
            ),
        )
        generation_thread.start()

        # NB: There are bugs in either pytorch or the gpt-fast repo. Inference
        # occasionally hangs and server-logs show things like:
        #
        #   <frozen runpy>:198: _run_code: block: [5,0,0], thread: [62,0,0] Assertion `index out of bounds: 0 <= tmp84 < 120` failed.
        #
        # Use a timeout and poll frequently. We kill the entire container when
        # an input times out, which is annoying given the large model
        # compilation times.
        print(
            f"[{prompt=},{num_samples=}] Waiting for inference to complete (timeout=30s) ...",
        )

        start_time = time.time()
        while True:
            time.sleep(0.1)

            if not generation_thread.is_alive():
                return

            if time.time() - start_time > 30:  # > 30s
                print(
                    f"[{prompt=},{num_samples=}] Timed out waiting for inference to complete"
                )
                return

            try:
                data = q.get_nowait()
                if data is sentinel:
                    break
                yield data
            except queue.Empty:
                pass

    def generate_inner(
        self,
        prompt: str,
        num_samples: int,  # if 0, we compile the model
        max_new_tokens: int,
        speculate_k: int,
        temperature: float,
        top_k: int,
        interactive: bool,
        q: queue.Queue,
        sentinel: object,
    ):
        tokenizer_path = self.checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        is_chat = "chat" in str(self.checkpoint_path)
        if is_chat:
            prompt = f"{B_INST} {prompt.strip()} {E_INST}"

        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=self.device)
        prompt_length = encoded.size(0)

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self.model.parameters(), self.model.buffers()
                )
            ]
        )
        aggregate_metrics = {
            "tokens_per_sec": [],
            "accept_counts": [],
        }

        start = -1 if self.compile_model else 0
        for i in range(start, num_samples):
            torch.cuda.synchronize()

            if i == 0:
                print(f"Starting inference for prompt = '{prompt}'")

            if interactive and i >= 0:
                buffer = []
                period_id = tokenizer.encode(".")[0]
                done_generating = False

                def callback(x):
                    nonlocal done_generating
                    if done_generating:
                        return

                    xlist = [
                        item
                        for sublist in [x.tolist()]
                        for item in (
                            sublist if isinstance(sublist, list) else [sublist]
                        )
                    ]
                    buffer.append(tokenizer.decode([period_id] + xlist)[1:])
                    if x.item() == tokenizer.eos_id():
                        done_generating = True

                    if len(buffer) == 4 or done_generating:
                        q.put("".join(buffer))
                        buffer.clear()

            else:

                def callback(x):
                    return x

            t0 = time.perf_counter()

            try:
                y, metrics = generate.generate(
                    self.model,
                    encoded,
                    max_new_tokens,
                    interactive=interactive,
                    draft_model=self.draft_model,
                    speculate_k=speculate_k,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                )
            except Exception as e:
                print("Exception encountered during inference", e)
                break

            aggregate_metrics["accept_counts"].append(metrics["accept_counts"])

            if i == -1 and not self.warmed_up:
                print(
                    f"Model compilation time: {time.perf_counter() - t0:.2f} seconds"
                )
                continue

            torch.cuda.synchronize()
            t = time.perf_counter() - t0

            if not interactive:
                generated = tokenizer.decode(y.tolist())
                q.put(generated)
            else:
                q.put("\n")

            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            aggregate_metrics["tokens_per_sec"].append(tokens_sec)
            print(
                f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
            )
            print(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

        is_speculative = self.draft_checkpoint_path is not None
        if is_speculative:
            counts_aggregated = [
                sum(i) for i in zip(*aggregate_metrics["accept_counts"])
            ]
            acceptance_probs = [
                i / sum(counts_aggregated) for i in counts_aggregated
            ]
            print(f"Acceptance probs: {acceptance_probs}")
            print(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
            )

        if num_samples > 0:
            print(
                f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
            )
            print(
                f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        q.put(sentinel)


@stub.local_entrypoint()
def main(
    # Lookup an already deployed model. If False, we'll deploy a new one
    # constructed from the following args.
    lookup_existing: bool = False,
    # Model construction args.
    compile_model: bool = True,  # Compile the model through pytorch (makes for slower cold starts but much faster inference).
    compile_prefill: bool = False,  # Compile the prefill function (only used if compile_model is True, and ).
    use_base_model: bool = False,  # Use the base model (instead of the int8 quantized one).
    use_speculative_sampling: bool = False,  # Use speculative sampling.
    # Inference args.
    prompt: str = "",  # Input prompt.
    num_samples: int = 1,  # How many responses to generate for each prompt.
    max_new_tokens: int = 100,  # Size of each generated response.
    speculate_k: int = 5,  # Speculative execution depth.
    temperature: float = 0.8,  # Temperature for sampling.
    top_k: int = 200,  # Top-k for sampling.
    interactive: bool = True,  # Whether to stream response.
):
    if lookup_existing:
        fn = Function.lookup("gpt-fast", "Model.generate")
    else:
        fn = Model(
            compile_model=compile_model,
            compile_prefill=compile_prefill,
            use_base_model=use_base_model,
            use_speculative_sampling=use_speculative_sampling,
        ).generate

    prompts = [prompt]
    if not prompt:
        prompts = [
            "Implement fibonacci in python.",
            "Write a Rust function that performs binary exponentiation.",
            "How do I allocate memory in C?",
        ]

    for prompt in prompts:
        for generated in fn.remote_gen(
            prompt=prompt,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            speculate_k=speculate_k,
            temperature=temperature,
            top_k=top_k,
            interactive=interactive,
        ):
            print(generated, end="")


app = Stub("gpt-fast-app", image=Image.debian_slim())


@app.function(
    mounts=[
        Mount.from_local_dir(
            Path(__file__).parent.parent / "llm-frontend",
            remote_path="/assets",
        ),
    ],
    allow_concurrent_inputs=10,
    timeout=10 * 60,
)
@asgi_app(label="gpt-fast-app")
def modal_app():
    import json
    from urllib.parse import unquote

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/model")
    async def model():
        return {"name": "Llama-2-7b-chat-hf"}

    @web_app.get("/stats")
    async def stats():
        stats = await Function.lookup(
            "gpt-fast", "Model.generate"
        ).get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        async def generate():
            fn = Function.lookup("gpt-fast", "Model.generate")
            for generated in fn.remote_gen(unquote(question)):
                yield f"data: {json.dumps(dict(text=generated), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app
