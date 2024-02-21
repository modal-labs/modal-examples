from pathlib import Path

import modal

stub = modal.Stub("sd-demo")

image = modal.Image.debian_slim().pip_install(
    "diffusers", "transformers", "accelerate"
)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.pth"  # Use the correct ckpt for your step setting!


with image.imports():
    import io

    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download


@stub.cls(image=image, gpu="a100")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.pipe.unet.load_state_dict(
            torch.load(hf_hub_download(repo, ckpt), map_location="cuda")
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

    @modal.method()
    def generate(
        self,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ):
        image = self.pipe(
            prompt, num_inference_steps=4, guidance_scale=0
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return buffer.getvalue()


frontend_path = Path(__file__).parent / "frontend"


@stub.function(
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI
    from fastapi.responses import Response

    web_app = FastAPI()

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        image_bytes = Model().generate.remote(prompt)

        return Response(image_bytes, media_type="image/jpeg")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
