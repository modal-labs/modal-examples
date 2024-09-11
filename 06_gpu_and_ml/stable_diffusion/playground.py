# ---
# deploy: true
# output-directory: "/tmp/playground-2-5"
# args: ["--prompt", "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe."]
# ---

from pathlib import Path

import fastapi.staticfiles
import modal
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates

app = modal.App("playground-2-5")

DIFFUSERS_GIT_SHA = "2e31a759b5bd8ca2b288b5c61709636a96c4bae9"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_GIT_SHA}",
        "transformers~=4.38.1",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)


with image.imports():
    import io

    import torch
    from diffusers import DiffusionPipeline


@app.cls(image=image, gpu="H100")
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

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        image = self.pipe(
            prompt,
            negative_prompt="disfigured, ugly, deformed",
            num_inference_steps=50,
            guidance_scale=3,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return buffer

    @modal.method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()

    @modal.web_endpoint(docs=True)
    def web_inference(
        self, prompt: str, n_steps: int = 24, high_noise_frac: float = 0.8
    ):
        return Response(
            content=self._inference(
                prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
            ).getvalue(),
            media_type="image/jpeg",
        )


frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@app.function(
    image=web_image,
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def ui():
    web_app = FastAPI(docs_url="/docs")
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Model.web_inference.web_url,
                "model_name": "Playground 2.5",
                "default_prompt": "Astronaut in the ocean, cold color palette, muted colors, detailed, 8k",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


@app.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/playground-2-5")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
