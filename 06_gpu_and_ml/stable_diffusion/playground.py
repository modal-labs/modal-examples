from pathlib import Path

import modal

stub = modal.Stub("playground-2-5")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate",
        "safetensors",
    )
)


with image.imports():
    import io

    import torch
    from diffusers import DiffusionPipeline


@stub.cls(image=image, gpu="a100")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
        # from diffusers import EDMDPMSolverMultistepScheduler
        # pipe.scheduler = EDMDPMSolverMultistepScheduler()

    @modal.method()
    def generate(
        self,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ):
        image = self.pipe(
            prompt, num_inference_steps=50, guidance_scale=3
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
