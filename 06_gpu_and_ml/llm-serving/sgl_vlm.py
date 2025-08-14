# # Run Qwen2-VL on SGLang for Visual QA

# Vision-Language Models (VLMs) are like LLMs with eyes:
# they can generate text based not just on other text,
# but on images as well.

# This example shows how to run a VLM on Modal using the
# [SGLang](https://github.com/sgl-project/sglang) library.

# Here's a sample inference, with the image rendered directly (and at low resolution) in the terminal:

# ![Sample output answering a question about a photo of the Statue of Liberty](https://modal-public-assets.s3.amazonaws.com/sgl_vlm_qa_sol.png)

# ## Setup

# First, we'll import the libraries we need locally
# and define some constants.

import os
import time
import warnings
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# VLMs are generally larger than LLMs with the same cognitive capability.
# LLMs are already hard to run effectively on CPUs, so we'll use a GPU here.
# We find that inference for a single input takes about 3-4 seconds on an A10G.

# You can customize the GPU type and count using the `GPU_TYPE` and `GPU_COUNT` environment variables.
# If you want to see the model really rip, try an `"a100-80gb"` or an `"h100"`
# on a large batch.

GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

# We use the [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
# model by Alibaba.

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"
TOKENIZER_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"

# We download it from the Hugging Face Hub using the Python function below.
# We'll store it in a [Modal Volume](https://modal.com/docs/guide/volumes)
# so that it's not downloaded every time the container starts.

MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )


# Modal runs Python functions on containers in the cloud.
# The environment those functions run in is defined by the container's `Image`.
# The block of code below defines our example's `Image`.
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])  # removes chatty prints on entry
    .apt_install("libnuma-dev")  # Add NUMA library for sgl_kernel
    .pip_install(  # add sglang and some Python dependencies
        "transformers==4.54.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.7.1",
        "sglang[all]==0.4.10.post2",
        "sgl-kernel==0.2.8",
        "hf-xet==1.1.5",
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_function(  # download the model by running a Python function
        download_model, volumes=volumes
    )
    .pip_install(  # add an optional extra that renders images in the terminal
        "term-image==0.7.1"
    )
)

# ## Defining a Visual QA service

# Running an inference service on Modal is as easy as writing inference in Python.

# The code below adds a modal `Cls` to an `App` that runs the VLM.

# We define a method `generate` that takes a URL for an image and a question
# about the image as inputs and returns the VLM's answer.

# By decorating it with `@modal.fastapi_endpoint`, we expose it as an HTTP endpoint,
# so it can be accessed over the public Internet from any client.

app = modal.App("example-sgl-vlm")


@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=vlm_image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=GPU_COUNT,  # t_ensor p_arallel size, number of GPUs to split the model over
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> str:
        from pathlib import Path

        import requests
        import sglang as sgl
        from term_image.image import from_file

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = request.get("image_url")
        if image_url is None:
            image_url = (
                "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
            )

        response = requests.get(image_url)
        response.raise_for_status()

        image_filename = image_url.split("/")[-1]
        image_path = Path(f"/tmp/{uuid4()}-{image_filename}")
        image_path.write_bytes(response.content)

        @sgl.function
        def image_qa(s, image_path, question):
            s += sgl.user(sgl.image(str(image_path)) + question)
            s += sgl.assistant(sgl.gen("answer"))

        question = request.get("question")
        if question is None:
            question = "What is this?"

        state = image_qa.run(
            image_path=image_path, question=question, max_new_tokens=128
        )
        # show the question and image in the terminal for demonstration purposes
        print(Colors.BOLD, Colors.GRAY, "Question: ", question, Colors.END, sep="")
        terminal_image = from_file(image_path)
        terminal_image.draw()
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        return state["answer"]

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()


# ## Asking questions about images via POST

# Now, we can send this Modal Function a POST request with an image and a question
# and get back an answer.

# The code below will start up the inference service
# so that it can be run from the terminal as a one-off,
# like a local script would be, using `modal run`:

# ```bash
# modal run sgl_vlm.py
# ```

# By default, we hit the endpoint twice to demonstrate how much faster
# the inference is once the server is running.


@app.local_entrypoint()
def main(
    image_url: Optional[str] = None, question: Optional[str] = None, twice: bool = True
):
    import json
    import urllib.request

    model = Model()

    payload = json.dumps(
        {
            "image_url": image_url,
            "question": question,
        },
    )

    req = urllib.request.Request(
        model.generate.get_web_url(),
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        assert response.getcode() == 200, response.getcode()
        print(json.loads(response.read().decode()))

    if twice:
        # second response is faster, because the Function is already running
        with urllib.request.urlopen(req) as response:
            assert response.getcode() == 200, response.getcode()
            print(json.loads(response.read().decode()))


# ## Deployment

# To set this up as a long-running, but serverless, service, we can deploy it to Modal:

# ```bash
# modal deploy sgl_vlm.py
# ```

# And then send requests from anywhere. See the [docs](https://modal.com/docs/guide/webhook-urls)
# for details on the `web_url` of the function, which also appears in the terminal output
# when running `modal deploy`.

# You can also find interactive documentation for the endpoint at the `/docs` route of the web endpoint URL.

# ## Addenda

# The rest of the code in this example is just utility code.

warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
