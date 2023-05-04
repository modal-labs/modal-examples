# # Stable Diffusion (AITemplate Edition)
#
# This example shows the Stable Diffusion 1.5 compiled with [AITemplate](https://github.com/facebookincubator/AITemplate) to run faster on Modal.
# There is also a [Stable Diffusion CLI example](/docs/guide/ex/stable_diffusion_cli).
# 
# #### Upsides
#  - Image generation improves over the CLI example to about 550ms per image generated (A10G, 10 steps, 512x512, png).
# 
# #### Downsides
#  - Width and height as well as batch size must be configured prior to compilation which takes about 15 minutes.
#  - In this example the compilation is done at docker image creation.
#  - Cold start time are also increased to upto ~30s.

# ## Setup
import io
import os
import sys
import modal
from fastapi import FastAPI, Response
from pydantic import BaseModel

hf_dir = "/root/.cache/huggingface"
hf_volume = modal.SharedVolume().persist("hf-cache-vol")

# ## Define the GPU Type and model, dimensions and batch size
gpu_type = "A10G"
model_id = "runwayml/stable-diffusion-v1-5"
width = 512
height = 512
batch_size = 1

# ## Set the paths to the AITemplate example. This is needed to import the compiler and pipeline.

def set_paths():
    ait_sd_example_path = "/app/AITemplate/examples/05_stable_diffusion"
    os.chdir(ait_sd_example_path)
    sys.path.append(ait_sd_example_path)

def download_and_compile():
    import diffusers
    import torch
    set_paths()

    # Download model and scheduler
    model_pt = "./tmp/diffusers/"
    diffusers.StableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    ).save_pretrained(model_pt, safe_serialization=True)

    diffusers.DPMSolverMultistepScheduler.from_pretrained(
        model_pt, subfolder="scheduler",
    ).save_pretrained(model_pt, safe_serialization=True)

    # Compilation
    from src.compile_lib.compile_clip import compile_clip
    from src.compile_lib.compile_unet import compile_unet
    from src.compile_lib.compile_vae import compile_vae

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_pt, revision="fp16", torch_dtype=torch.float16
    )

    compile_clip(
        pipe.text_encoder,
        batch_size=batch_size,
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
        batch_size=batch_size * 2,
        width=width // 8,
        height=height // 8,
        use_fp16_acc=True,
        convert_conv_to_gemm=True,
        hidden_dim=pipe.unet.config.cross_attention_dim,
        attention_head_dim=pipe.unet.config.attention_head_dim,
        use_linear_projection=pipe.unet.config.get("use_linear_projection", False),
    )
    compile_vae(
        pipe.vae,
        batch_size=batch_size,
        width=width // 8,
        height=height // 8,
        use_fp16_acc=True,
        convert_conv_to_gemm=True,
    )

def _get_pipe():
    set_paths()
    import torch
    from diffusers import DPMSolverMultistepScheduler
    from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model_pt = "./tmp/diffusers/"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_pt,
        subfolder="scheduler",
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        denoise_final=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    pipe = StableDiffusionAITPipeline.from_pretrained(
        model_pt,
        scheduler=scheduler,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def _inference(pipe, prompt: str, num_inference_steps: int, guidance_scale: float, negative_prompt: str, format: str = "webp"):
    from torch import autocast, inference_mode
    with inference_mode():
        with autocast("cuda"):
            single_image = pipe(
                prompt=[prompt] * batch_size,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=[negative_prompt] * batch_size,
            ).images[0]
            with io.BytesIO() as buf:
                single_image.save(buf, format=format)
                return Response(content=buf.getvalue(), media_type=f"image/{format}")

# ## Define Image from Dockerhub
# Install AITemplate from source, download and compile the configured model.

image = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get install -y git python3-pip",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
            "WORKDIR /app",
            "RUN git clone --recursive https://github.com/facebookincubator/AITemplate.git",
            "WORKDIR /app/AITemplate/python",
            "RUN git checkout b041e0e",
            "RUN python setup.py bdist_wheel && pip install dist/aitemplate-*.whl",
            "WORKDIR /app/AITemplate/examples/05_stable_diffusion",
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
        "xformers==0.0.20.dev526",
    )
    .run_function(
        download_and_compile,
        secret=modal.Secret.from_name("my-huggingface-secret"),
        shared_volumes={hf_dir: hf_volume},
        timeout=60 * 30,
        gpu=gpu_type
    )
    .run_commands(
        # clean up some files we don't need
        "find /app/AITemplate/examples/05_stable_diffusion/tmp/CLIPTextModel -type f ! -name '*.so' -delete ",
        "find /app/AITemplate/examples/05_stable_diffusion/tmp/UNet2DConditionModel -type f ! -name '*.so' -delete ",
        "find /app/AITemplate/examples/05_stable_diffusion/tmp/AutoencoderKL -type f ! -name '*.so' -delete ",
        "rm -rf /app/AITemplate/examples/05_stable_diffusion/tmp/profiler",
        "rm -rf /app/AITemplate/examples/05_stable_diffusion/tmp/diffusers/models--runwayml--stable-diffusion-v1-5",
        "rm -rf /app/AITemplate/python/dist",
    )
)

function_params = {
    "image": image,
    "concurrency_limit": 1,
    "container_idle_timeout": 60,
    "timeout": 60,
    "gpu": gpu_type,
}

class InferenceRequest(BaseModel):
    prompt: str = "photo of a cat in the snow, blue eyes, highly detailed, 8k, 200mm canon lens, shallow depth of field"
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    negative_prompt: str = "deformed, extra legs, no tail"
    format: str = "webp"

stub = modal.Stub()

# ## Inference as asgi app
@stub.function(**function_params)
@modal.asgi_app(label=f'{gpu_type.lower()}-{width}-{height}-{batch_size}-{model_id.replace("/","--")}')
def baremetal_asgi():
    pipe = _get_pipe()
    app = FastAPI()
    @app.post("/inference")
    def inference(request: InferenceRequest):
        # A10G inference time for 10 steps 512x512 webp: execution time: 505.0 ms, total latency: 585.8 ms
        # A10G inference time for 10 steps 512x512 png: execution time: 557.6 ms, total latency: 629.9 ms
        return _inference(pipe, request.prompt, request.num_inference_steps, request.guidance_scale, request.negative_prompt, request.format)
    return app

# ## Cold Start: [14..33] seconds
# '''
#    90%|█████████ | 9/10 [00:00<00:00, 25.00it/s]100%|██████████| 10/10 [00:00<00:00, 24.84it/s]
#    Request finished with status 200. (execution time: 2324.5 ms, total latency: 33779.0 ms)
#    90%|█████████ | 9/10 [00:00<00:00, 25.60it/s]100%|██████████| 10/10 [00:00<00:00, 25.59it/s]
#    Request finished with status 200. (execution time: 508.6 ms, total latency: 592.6 ms)
#    90%|█████████ | 9/10 [00:00<00:00, 25.67it/s]100%|██████████| 10/10 [00:00<00:00, 25.67it/s]
#    Request finished with status 200. (execution time: 507.2 ms, total latency: 598.3 ms)
#    90%|█████████ | 9/10 [00:00<00:00, 25.67it/s]100%|██████████| 10/10 [00:00<00:00, 25.67it/s]
#    Request finished with status 200. (execution time: 507.5 ms, total latency: 648.9 ms)
# '''

