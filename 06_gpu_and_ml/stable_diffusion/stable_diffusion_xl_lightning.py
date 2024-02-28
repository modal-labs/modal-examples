from pathlib import Path

import modal

stub = modal.Stub("stable-diffusion-xl-lightning")

image = modal.Image.debian_slim().pip_install(
    "diffusers", "transformers", "accelerate"
)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"


with image.imports():
    import io

    import torch
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from fastapi import Response
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file


@stub.cls(image=image, gpu="a100")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            "cuda", torch.float16
        )
        unet.load_state_dict(
            load_file(hf_hub_download(repo, ckpt), device="cuda")
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

    @modal.web_endpoint()
    def inference(
        self,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ):
        image = self.pipe(
            prompt, num_inference_steps=4, guidance_scale=0
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpeg")


frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@stub.function(
    image=web_image,
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI
    from jinja2 import Template

    web_app = FastAPI()

    with open("/assets/index.html", "r") as f:
        template_html = f.read()

    template = Template(template_html)

    with open("/assets/index.html", "w") as f:
        html = template.render(
            inference_url=Model.inference.web_url,
            model_name="SDXL Lightning",
            default_prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        )
        f.write(html)

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
