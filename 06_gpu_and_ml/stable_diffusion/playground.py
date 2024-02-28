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
    from fastapi import Response


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

    @modal.web_endpoint()
    def inference(
        self,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ):
        image = self.pipe(
            prompt, num_inference_steps=50, guidance_scale=3
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(buffer.getvalue(), media_type="image/jpeg")


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
            model_name="Playground 2.5",
        )
        f.write(html)

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
