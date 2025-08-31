# ---
# deploy: true
# ---

# # Run state-of-the-art RLMs on Blackwell GPUs with TensorRT-LLM (DeepSeek-R1-0528-FP4)

# In this example, we demonstrate how to use the TensorRT-LLM framework to serve NVIDIA's DeepSeek-R1-0528-FP4 model,
# a [state-of-the-art reasoning language model](https://lmarena.ai/leaderboard),
# on Modal's Blackwell GPUs (8 x B200s).

# Because this model is so large, our focus will be on optimizing the cold start and the model's inference latencies.
# We use [NVIDIA's recommendations](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4#minimum-latency-server-deployment)
# for matching expected performance and minimizing inference latency.

# ## Overview

# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the DeepSeek-R1-0528-FP4 model.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# To run code on Modal, we define [container images](https://modal.com/docs/guide/images).
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` container image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

import os
import webbrowser  # for opening generated HTML files in browser
from pathlib import Path

import modal

# We first install PyTorch with CUDA 12.8 support (required for Blackwell).
# We also add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and install packages using [uv](https://docs.astral.sh/uv/)
# to speed up the installation process.

tensorrt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.12
    )
    .entrypoint([])  # silence base-image entrypoint
    .apt_install(
        "git",
        "openmpi-bin",
        "libopenmpi-dev",
    )
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "mpi4py",
        "tensorrt_llm==1.0.0rc0",
    )
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes a few minutes on first run.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run` so that it runs in the background while you read the rest.

# ## Downloading the model

# Next, we'll set up a few things to download the model to persistent storage and do it quickly.
# For persistent, distributed storage, we use
# [Modal Volumes](https://modal.com/docs/guide/volumes), which can be accessed from any container
# with read speeds in excess of a gigabyte per second.

# We also set the `HF_HOME` environment variable to point to the Volume so that the model
# is cached there. And we install `hf-transfer` to get maximum download throughput from
# the Hugging Face Hub, in the hundreds of megabytes per second.

app_name = "example-trtllm-deepseek"

hf_cache_vol = modal.Volume.from_name(f"{app_name}-hf-cache", create_if_missing=True)
HF_CACHE_PATH = Path("/hf_cache")
volumes = {HF_CACHE_PATH: hf_cache_vol}

MODEL_NAME = "nvidia/DeepSeek-R1-0528-FP4-v2"
MODEL_REVISION = "d12ff8db9876124d533b26bc24523c27907ce386"  # in case repo updates!
MODELS_PATH = HF_CACHE_PATH / "models"
MODEL_PATH = MODELS_PATH / MODEL_NAME


# We use the function below to download the model from the Hugging Face Hub.


def download_model():
    from huggingface_hub import snapshot_download

    print(f"downloading base model to {MODEL_PATH} if necessary")
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )


# Just defining that function doesn't actually download the model, though.
# We can run it by adding it to the image's build process with `run_function`.
# The download process has its own dependencies, which we add here.

MINUTES = 60  # seconds
tensorrt_image = (
    tensorrt_image.uv_pip_install("hf-transfer==0.1.9", "huggingface_hub==0.33.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(MODELS_PATH),
        }
    )
    .run_function(download_model, volumes=volumes, timeout=40 * MINUTES)
)

with tensorrt_image.imports():
    from tensorrt_llm import SamplingParams
    from tensorrt_llm._tensorrt_engine import LLM


# ## Setting up the engine

# ### Configure plugins

# TensorRT-LLM is an LLM inference framework built on top of NVIDIA's TensorRT,
# which is a generic inference framework for neural networks.

# TensorRT includes a "plugin" extension system that allows you to adjust behavior,
# like configuring the [CUDA kernels](https://modal.com/gpu-glossary/device-software/kernel)
# used by the engine.
# The [General Matrix Multiply (GEMM)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
# plugin, for instance, adds heavily-optimized matrix multiplication kernels
# from NVIDIA's [cuBLAS library of linear algebra routines](https://docs.nvidia.com/cuda/cublas/).

# We'll specify the `paged_kv_cache` plugin which enables a
# [paged attention algorithm](https://arxiv.org/abs/2309.06180)
# for the key-value (KV) cache.


def get_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig

    return PluginConfig.from_dict(
        {
            "paged_kv_cache": True,
        }
    )


# ### Configure speculative decoding

# Speculative decoding is a technique for generating multiple tokens per step,
# avoiding the auto-regressive bottleneck in the Transformer architecture.
# Generating multiple tokens in parallel exposes more parallelism to the GPU.
# It works best for text that has predicable patterns, like code,
# but it's worth testing for any workload where latency is critical.

# Speculative decoding can use any technique to guess tokens, including running another,
# smaller language model. Here, we'll use a simple, but popular and effective
# speculative decoding strategy called "multi-token prediction (MTP) decoding",
# which essentially uses a smaller model to generate the next token.


def get_speculative_config():
    from tensorrt_llm.llmapi import MTPDecodingConfig

    return MTPDecodingConfig(
        num_nextn_predict_layers=3,  # number of layers to predict next n tokens
        use_relaxed_acceptance_for_thinking=True,  # draft token accepted when it's a candidate
        relaxed_topk=10,  # first k candidates are considered
        relaxed_delta=0.6,  # delta for relaxed acceptance
    )


# ### Set the build config

# Finally, we'll specify the overall build configuration for the engine. This includes
# more obvious parameters such as the maximum input length, the maximum number of tokens
# to process at once before queueing occurs, and the maximum number of sequences
# to process at once before queueing occurs.

# To minimize latency, we set the maximum number of sequences (the "batch size")
# to 4. We enforce this maximum by setting the number of inputs that the
# Modal Function is allowed to process at once -- `max_concurrent_inputs`.

MAX_BATCH_SIZE = MAX_CONCURRENT_INPUTS = 4


def get_build_config():
    from tensorrt_llm import BuildConfig

    return BuildConfig(
        plugin_config=get_plugin_config(),
        max_input_len=8192,
        max_num_tokens=16384,
        max_batch_size=MAX_BATCH_SIZE,
    )


# ## Serving inference

# Now that we have written the code to compile the engine, we can
# serve it with Modal!

# We start by creating an `App`.

app = modal.App(app_name)

# Thanks to our [custom container runtime system](https://modal.com/blog/jono-containers-talk),
# even this large container boots in seconds.

# On the first container start, we mount the Volume and build the engine,
# which takes a few minutes. Subsequent starts will be much faster,
# as the engine is cached in the Volume and loaded in seconds.

# Container starts are triggered when Modal scales up your Function,
# like the first time you run this code or the first time a request comes in after a period of inactivity.
# For details on optimizing container start latency, see
# [this guide](https://modal.com/docs/guide/cold-start).

# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to separate out the engine startup (`enter`) and engine execution (`generate`).
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).


N_GPU = 8


@app.cls(
    image=tensorrt_image,
    gpu=f"B200:{N_GPU}",
    scaledown_window=60 * MINUTES,
    timeout=60 * MINUTES,
    volumes=volumes,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class Model:
    def build_engine(self, engine_path, engine_kwargs):
        llm = LLM(model=MODEL_PATH, **engine_kwargs)
        # llm.save(engine_path)
        return llm

    @modal.enter()
    def enter(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        engine_kwargs = {
            "build_config": get_build_config(),
            "speculative_config": get_speculative_config(),
            "tensor_parallel_size": N_GPU,
            "moe_backend": "TRTLLM",
            "use_cuda_graph": True,
            "backend": "pytorch",
            "max_batch_size": MAX_BATCH_SIZE,
            "trust_remote_code": True,
        }

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,  # max generated tokens
        )

        engine_path = MODEL_PATH / "trtllm_engine"
        if not os.path.exists(engine_path):
            print(f"building new engine at {engine_path}")
            self.llm = self.build_engine(engine_path, engine_kwargs)
        else:
            print(f"loading engine from {engine_path}")
            self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.method()
    async def generate_async(self, prompt):
        text = self.text_from_prompt(prompt)
        async for output in self.llm.generate_async(
            text, self.sampling_params, streaming=True
        ):
            yield output.outputs[0].text_diff

    def text_from_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @modal.method()
    def boot(self):
        pass  # no-op to start up containers

    @modal.exit()
    def shutdown(self):
        self.llm.shutdown()
        del self.llm


# ## Calling our inference function

# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.

# We wrap that logic in a `local_entrypoint` so you can run it from the command line with

# ```bash
# modal run trtllm_deepseek.py
# ```

# For simplicity, we ask the model to generate a game of tic-tac-toe in HTML and open it in the browser.
# But the code in the `local_entrypoint` is just regular Python code
# that runs on your machine -- we wrap it in a CLI automatically --
# so feel free to customize it to your liking.


@app.local_entrypoint()
def main():
    print("🏎️  creating container")
    model = Model()

    print("🏎️  cold booting container")
    model.boot.remote()

    prompt = """
    Create an HTML page implementing a simple game of tic-tac-toe.
    Only output the HTML in English, no other text or language.
    """

    print("🏎️ creating game of tic-tac-toe")
    resp = ""
    for out in model.generate_async.remote_gen(prompt):
        print(out, end="", flush=True)
        resp += out
    print("\n")

    # post-process
    html_content = (
        resp.split("</think>")[-1].split("```html")[-1].split("```")[0].strip()
    )

    html_filename = Path(__file__).parent / "tic_tac_toe.html"
    with open(html_filename, "w") as f:
        f.write(html_content)

    file_path = html_filename.absolute()
    file_url = f"file://{file_path}"
    print(f"\nHTML saved to: {file_path}")
    print(f"Opening in browser: {file_url}")

    print("🏎️  opening in browser")
    success = webbrowser.open(file_url)
    if not success:
        print(f"Failed to open browser, please manually open: {file_url}")


# Once deployed with `modal deploy`, this `Model.generate` function
# can be called from other Python code. It can also be converted to an HTTP endpoint
# for invocation over the Internet by any client.
# For details, see [this guide](https://modal.com/docs/guide/trigger-deployed-functions).

# As a quick demo, we've included some sample chat client code in the
# Python main entrypoint below. To use it, first deploy with

# ```bash
# modal deploy trtllm_deepseek.py
# ```

# and then run the client with

# ```python notest
# python trtllm_deepseek.py
# ```


if __name__ == "__main__":
    import sys

    try:
        Model = modal.Cls.from_name(app_name, "Model")
        print("🏎️  connecting to model")
        model = Model()
        model.boot.remote()
    except modal.exception.NotFoundError as e:
        raise SystemError("Deploy this app first with modal deploy") from e

    print("🏎️  starting chat. exit with :q, ctrl+C, or ctrl+D")
    try:
        prompt = []
        while (nxt := input("🏎️  > ")) != ":q":
            prompt.append({"role": "user", "content": nxt})
            resp = ""
            for out in model.generate_async.remote_gen(prompt):
                print(out, end="", flush=True)
                resp += out
            print("\n")
            prompt.append({"role": "assistant", "content": resp})
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        print("\n")
        sys.exit(0)
