import base64
from pathlib import Path
import json

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from modal import Image, Mount, Stub, asgi_app, gpu, method


def download_models():
    from huggingface_hub import snapshot_download

    # Ignore files that we don't need to speed up download time.
    ignore = [
        "*.bin",
        "*.onnx_data",
        "*/diffusion_pytorch_model.safetensors",
    ]

    snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)


image = (
    Image.debian_slim()
    .pip_install(
        "Pillow~=10.1.0",
        "diffusers~=0.24",
        "transformers~=4.35",
        "accelerate~=0.25",
        "safetensors~=0.4",
    )
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl-turbo", image=image)


@stub.cls(gpu=gpu.A100(memory=40), container_idle_timeout=240)
class Model:
    def __enter__(self):
        import torch
        from diffusers import AutoPipelineForImage2Image

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
        )

    @method()
    async def inference(self, image_bytes, prompt):
        from io import BytesIO

        from diffusers.utils import load_image
        from PIL import Image

        init_image = load_image(Image.open(BytesIO(image_bytes))).resize((512,512))
        num_inference_steps = 2
        strength = 0.50
        assert num_inference_steps * strength >= 1

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
            use_safetensors=True,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


web_app = FastAPI()
static_path = Path(__file__).with_name("webcam").resolve()


@web_app.post("/generate")
async def generate(request: Request):
    body = await request.body()
    body_json = json.loads(body)
    img_data_in = base64.b64decode(body_json["image"].split(",")[1])  # read data-uri
    prompt = body_json["prompt"]
    img_data_out = await Model().inference.remote.aio(img_data_in, prompt)
    output_data = b"data:image/png;base64," + base64.b64encode(img_data_out)
    return Response(content=output_data)


@stub.function(
    mounts=[Mount.from_local_dir(static_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app
