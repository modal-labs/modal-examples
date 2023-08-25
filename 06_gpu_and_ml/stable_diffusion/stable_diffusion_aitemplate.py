# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/stable_diffusion/stable_diffusion_aitemplate.py"]
# ---
# # Stable Diffusion (AITemplate Edition)
#
# Example by [@maxscheel](https://github.com/maxscheel)
#
# This example shows the Stable Diffusion 2.1 compiled with [AITemplate](https://github.com/facebookincubator/AITemplate) to run faster on Modal.
# There is also a [Stable Diffusion CLI example](/docs/guide/ex/stable_diffusion_cli).
#
# #### Upsides
#  - Image generation improves over the CLI example to about 550ms per image generated (A10G, 10 steps, 512x512, png).
#
# #### Downsides
#  - Width and height as well as batch size must be configured prior to compilation which takes about 15 minutes.
#  - In this example the compilation is done at docker image creation.
#  - Cold start time are also increased to up-to ~30s from ~10s.

# ## Setup
import io
import os
import sys

import modal
from fastapi import FastAPI, Response
from pydantic import BaseModel

# Set cache path, size of output image, and stable diffusion version.
HF_CACHE_DIR: str = "/root/.cache/huggingface"
AIT_BUILD_CACHE_DIR: str = "/root/.cache/aitemplate"
GPU_TYPE: str = "A10G"
MODEL_ID: str = "stabilityai/stable-diffusion-2-1"
WIDTH: int = 512
HEIGHT: int = 512
BATCH_SIZE: int = 1
MODEL_PATH: str = "./tmp/diffusers/"


def set_paths():
    ait_sd_example_path = "/app/AITemplate/examples/05_stable_diffusion"
    os.chdir(ait_sd_example_path)
    sys.path.append(ait_sd_example_path)


# Download and compile model during image creation. This will store both the
# original HuggingFace model and the AITemplate compiled artifacts in the image,
# making startup times slightly faster.
def download_and_compile():
    import diffusers
    import torch

    set_paths()

    os.environ["AIT_BUILD_CACHE_DIR"] = AIT_BUILD_CACHE_DIR

    # Download model and scheduler
    diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    ).save_pretrained(MODEL_PATH, safe_serialization=True)

    diffusers.EulerDiscreteScheduler.from_pretrained(
        MODEL_PATH,
        subfolder="scheduler",
    ).save_pretrained(MODEL_PATH, safe_serialization=True)

    # Compilation
    from src.compile_lib.compile_clip import compile_clip
    from src.compile_lib.compile_unet import compile_unet
    from src.compile_lib.compile_vae import compile_vae

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_PATH, revision="fp16", torch_dtype=torch.float16
    )

    compile_clip(
        pipe.text_encoder,
        batch_size=BATCH_SIZE,
        seqlen=77,
        use_fp16_acc=True,
        convert_conv_to_gemm=True,
        depth=pipe.text_encoder.config.num_hidden_layers,
        num_heads=pipe.text_encoder.config.num_attention_heads,
        dim=pipe.text_encoder.config.hidden_size,
        act_layer=pipe.text_encoder.config.hidden_act,
    )
    compile_unet(
        pipe.unet,
        batch_size=BATCH_SIZE * 2,
        width=WIDTH // 8,
        height=HEIGHT // 8,
        use_fp16_acc=True,
        convert_conv_to_gemm=True,
        hidden_dim=pipe.unet.config.cross_attention_dim,
        attention_head_dim=pipe.unet.config.attention_head_dim,
        use_linear_projection=pipe.unet.config.get(
            "use_linear_projection", False
        ),
    )
    compile_vae(
        pipe.vae,
        batch_size=BATCH_SIZE,
        width=WIDTH // 8,
        height=HEIGHT // 8,
        use_fp16_acc=True,
        convert_conv_to_gemm=True,
    )


def _get_pipe():
    set_paths()
    os.environ["AIT_BUILD_CACHE_DIR"] = AIT_BUILD_CACHE_DIR

    import torch
    from diffusers import EulerDiscreteScheduler
    from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    scheduler = EulerDiscreteScheduler.from_pretrained(
        MODEL_PATH,
        subfolder="scheduler",
        device_map="auto",
    )

    pipe = StableDiffusionAITPipeline.from_pretrained(
        MODEL_PATH,
        scheduler=scheduler,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def _inference(
    pipe,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    format: str = "webp",
):
    from torch import autocast, inference_mode

    with inference_mode():
        with autocast("cuda"):
            single_image = pipe(
                prompt=[prompt] * BATCH_SIZE,
                HEIGHT=HEIGHT,
                WIDTH=WIDTH,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=[negative_prompt] * BATCH_SIZE,
            ).images[0]
            with io.BytesIO() as buf:
                single_image.save(buf, format=format)
                return Response(
                    content=buf.getvalue(), media_type=f"image/{format}"
                )


# ## Build image
#
# Install AITemplate from source, download, and compile the configured model. We
# will use an official NVIDIA image from [Docker Hub](https://hub.docker.com/r/nvidia/cuda)
# which include all required drivers.

image = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get install -y git python3-pip",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
            "WORKDIR /app",
            "RUN git clone --recursive https://github.com/facebookincubator/AITemplate.git",
            "WORKDIR /app/AITemplate/python",
            # Set hash for reproducibility
            "RUN git checkout 6305588af76eeec987762c5b5ee373a61f8a7fb3",
            # Build and install aitemplate library
            "RUN python setup.py bdist_wheel && pip install dist/aitemplate-*.whl && rm -rf dist",
            "WORKDIR /app/AITemplate/examples/05_stable_diffusion",
            # Patch deprecated access of unet.in_channels (silence warning) in AIT pipeline example implementation
            "RUN sed -i src/pipeline_stable_diffusion_ait.py -e 's/unet.in_channels/unet.config.in_channels/g'",
        ],
    )
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "transformers~=4.25.1",
        "safetensors",
        "torch>=2.0",
        "triton",
        "xformers==0.0.20",
    )
    .run_function(
        download_and_compile,
        secret=modal.Secret.from_name("huggingface-secret"),
        timeout=60 * 30,
        gpu=GPU_TYPE,
    )
)

function_params = {
    "image": image,
    "concurrency_limit": 1,
    "container_idle_timeout": 60,
    "timeout": 60,
    "gpu": GPU_TYPE,
}

stub = modal.Stub("example-stable-diffusion-aitemplate")

# ## Inference as asgi app
#
# We load the pipe and serve the example via the `/inference` endpoint. You can interact
# the endpoint via a `POST` request with a JSON payload containing parameters defined
# in `InferenceRequest`.


class InferenceRequest(BaseModel):
    prompt: str = "photo of a wolf in the snow, blue eyes, highly detailed, 8k, 200mm canon lens, shallow depth of field"
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    negative_prompt: str = "deformed, extra legs, no tail"
    format: str = "webp"  # png or webp; webp is slightly faster


@stub.function(**function_params)
@modal.asgi_app(
    label=f'{GPU_TYPE.lower()}-{WIDTH}-{HEIGHT}-{BATCH_SIZE}-{MODEL_ID.replace("/","--")}'
)
def inference_asgi():
    pipe = _get_pipe()
    app = FastAPI()

    @app.post("/inference")
    def inference(request: InferenceRequest):
        return _inference(
            pipe,
            request.prompt,
            request.num_inference_steps,
            request.guidance_scale,
            request.negative_prompt,
            request.format,
        )

    return app


# Serve your app using `modal serve` as follows:
#
# ```bash
# modal serve stable_diffusion_aitemplate.py
# ```
#
# Grab the Modal app URL then query the  API with curl:
#
# ```bash
# curl --location --request POST '$ENDPOINT_URL/inference' \
#      --header 'Content-Type: application/json' \
#      --data-raw '{
#         "prompt": "photo of a wolf in the snow, blue eyes, highly detailed, 8k, 200mm canon lens, shallow depth of field",
#         "num_inference_steps": 10,
#         "guidance_scale": 10.0,
#         "negative_prompt": "deformed, extra legs, no tail",
#         "format": "webp"
#      }'
# ```
