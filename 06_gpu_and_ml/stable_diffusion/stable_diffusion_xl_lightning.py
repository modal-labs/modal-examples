from pathlib import Path

import modal

app = modal.App("stable-diffusion-xl-lightning")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "diffusers==0.26.3", "transformers~=4.37.2", "accelerate==0.27.2"
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


@app.cls(image=image, gpu="a100")
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

    def _inference(self, prompt, n_steps=4):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.pipe(
            prompt=prompt,
            guidance_scale=0,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @modal.method()
    def inference(self, prompt, n_steps=4):
        return self._inference(
            prompt,
            n_steps=n_steps,
        ).getvalue()

    @modal.web_endpoint(docs=True)
    def web_inference(self, prompt, n_steps=4):
        return Response(
            content=self._inference(
                prompt,
                n_steps=n_steps,
            ).getvalue(),
            media_type="image/jpeg",
        )


# And this is our entrypoint; where the CLI is invoked. Run this example
# with: `modal run stable_diffusion_xl_lightning.py --prompt "An astronaut riding a green horse"`


@app.local_entrypoint()
def main(
    prompt: str = "in the style of Dali, a surrealist painting of a weasel in a tuxedo riding a bicycle in the rain",
):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl-lightning")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl_lightning.py`.
#
# Because the `web_endpoint` decorator on our `web_inference` function has the `docs` flag set to `True`,
# we also get interactive documentation for our endpoint at `/docs`.

frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@app.function(
    image=web_image,
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.templating import Jinja2Templates

    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Model.web_inference.web_url,
                "model_name": "Stable Diffusion XL Lightning",
                "default_prompt": "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app
