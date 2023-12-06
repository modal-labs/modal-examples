import base64
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from modal import Image, Mount, Stub, asgi_app, gpu, web_endpoint


def download_models():
    from huggingface_hub import snapshot_download

    # Ignore files that we don't need to speed up download time.
    ignore = [
        "*.bin",
        "*.onnx_data",
        "*/diffusion_pytorch_model.safetensors",
    ]

    snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)


stub = Stub("stable-diffusion-xl-turbo")

web_image = Image.debian_slim().pip_install("jinja2")

inference_image = (
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

with inference_image.run_inside():
    from io import BytesIO

    import torch
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image
    from PIL import Image


@stub.cls(
    gpu=gpu.A100(memory=40), container_idle_timeout=240, image=inference_image
)
class Model:
    def __enter__(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
            add_watermarker=False,
        )

    @web_endpoint(method="POST")
    async def inference(self, request: Request):
        t0 = time.time()
        body = await request.body()
        body_json = json.loads(body)
        img_data_in = base64.b64decode(
            body_json["image"].split(",")[1]
        )  # read data-uri
        prompt = body_json["prompt"]

        init_image = load_image(Image.open(BytesIO(img_data_in))).resize(
            (512, 512)
        )
        num_inference_steps = 2
        strength = 0.65
        assert num_inference_steps * strength >= 1

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
            use_safetensors=True,
            seed=42,
        ).images[0]

        print("infer time:", time.time() - t0)

        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        img_data_out = byte_stream.getvalue()

        print("total time:", time.time() - t0)

        output_data = b"data:image/jpeg;base64," + base64.b64encode(
            img_data_out
        )

        return Response(content=output_data)


static_path = Path(__file__).with_name("webcam").resolve()


@stub.function(
    mounts=[Mount.from_local_dir(static_path, remote_path="/assets")],
    image=web_image,
)
@asgi_app()
def fastapi_app():
    web_app = FastAPI()

    from jinja2 import Environment, FileSystemLoader

    jinja_env = Environment(loader=FileSystemLoader("/assets"))

    with open("/assets/index.html", "w") as f:
        html = jinja_env.get_template("index.html.template").render(
            inference_url=Model.inference.web_url
        )
        f.write(html)

    web_app.mount("/", StaticFiles(directory="/assets", html=True))

    return web_app
