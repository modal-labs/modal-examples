# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/llm-serving/vllm_inference.py"]
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.
# You can find a video walkthrough of this example on our YouTube channel [here](https://www.youtube.com/watch?v=QmY_7ePR1hM).

# Note that the vLLM server is a FastAPI app, which can be configured and extended just like any other.
# Here, we use it to add simple authentication middleware, following the
# [implementation in the vLLM repository](https://github.com/vllm-project/vllm/blob/v0.5.3post1/vllm/entrypoints/openai/api_server.py).

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# You can find a video walkthrough of this example and the related scripts on the Modal YouTube channel
# [here](https://www.youtube.com/watch?v=QmY_7ePR1hM).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `pip`.

import modal

vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.3post1", "fastapi[standard]==0.115.4"
)

# ## Download the model weights

# We'll be running a pretrained foundation model -- Meta's LLaMA 3.1 8B
# in the Instruct variant that's trained to chat and follow instructions,
# quantized to 4-bit by [Neural Magic](https://neuralmagic.com/) and uploaded to Hugging Face.

# You can read more about the `w4a16` "Machete" weight layout and kernels
# [here](https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/).

MODELS_DIR = "/llamas"
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"

# We need to make the weights of that model available to our Modal Functions.

# So to follow along with this example, you'll need to download those weights
# onto a Modal Volume by running another script from the
# [examples repository](https://github.com/modal-labs/modal-examples).

try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")


# ## Build a vLLM engine and serve it

# vLLM's OpenAI-compatible server is exposed as a [FastAPI](https://fastapi.tiangolo.com/) router.

# FastAPI is a Python web framework that implements the [ASGI standard](https://en.wikipedia.org/wiki/Asynchronous_Server_Gateway_Interface),
# much like [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) is a Python web framework
# that implements the [WSGI standard](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface).

# Modal offers [first-class support for ASGI (and WSGI) apps](https://modal.com/docs/guide/webhooks). We just need to decorate a function that returns the app
# with `@modal.asgi_app()` (or `@modal.wsgi_app()`) and then add it to the Modal app with the `app.function` decorator.

# The function below first imports the FastAPI router from the vLLM library, then adds authentication compatible with OpenAI client libraries. You might also add more routes here.

# Then, the function creates an `AsyncLLMEngine`, the core of the vLLM server. It's responsible for loading the model, running inference, and serving responses.

# After attaching that engine to the FastAPI app via the `api_server` module of the vLLM library, we return the FastAPI app
# so it can be served on Modal.

app = modal.App("example-vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(
    image=vllm_image,
    gpu=modal.gpu.H100(count=N_GPU),
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext

    volume.reload()  # ensure we have the latest version of the weights

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    # security: CORS middleware for external requests
    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # security: inject dependency on authed routes
    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_inference.py
# ```

# This will create a new app on Modal, build the container image for it, and deploy.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands. They also demonstrate authentication.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

# To interact with the API programmatically, you can use the Python `openai` library.

# See the `client.py` script in the examples repository
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible)
# to take it for a spin:

# ```bash
# # pip install openai==1.13.3
# python openai_compatible/client.py
# ```

# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatibl):

# ```bash
# modal run openai_compatible/load_test.py
# ```

# ## Addenda

# The rest of the code in this example is utility code.


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config
