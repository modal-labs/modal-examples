# # Stable Diffusion XL 1.0 Img2Img
#
# This example is similar to the xl file in the folder, but also includes an image input
#
# The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, gpu, method

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers",
        "invisible_watermark",
        "transformers",
        "accelerate",
        "safetensors",
        "gradio",
        "torch",
    )
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl", image=image)

# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        device = "cuda"
        self.base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
        # Load refiner model
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        ).to(device)

        # These suggested compile commands actually increase inference time, but may be mis-used.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt, input_image):
        from diffusers.utils import load_image
        import torch

        init_image = load_image(input_image).convert("RGB")
        negative_prompt = "disfigured, ugly, deformed"

        generator = torch.Generator(device="cuda").manual_seed(1024)
        image = self.base(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            strength=0.8,
            guidance_scale=10.5,
            generator=generator
        ).images
        image = self.refiner(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=0.8,
            guidance_scale=10.5,
            generator=generator
        ).images[0]

        return image


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

from fastapi import FastAPI
from PIL import Image

web_app = FastAPI()

assets_path = Path(__file__).parent / "assets"
@stub.function(
    image=image,
    concurrency_limit=3,
    mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call to the GPU inference function on Modal.
    def go(text, image):
        return Model().inference.remote(text, image)

    # set up AppConfig
    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/ex/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

      ### Learn how to make your own [here]({modal_example_url}).
    """

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs=[
          gr.Textbox(label="Text Input"),
          gr.Image(shape=(768,768), type="pil")  # Adjust the shape as needed
        ],
        outputs=gr.Image(shape=(768, 768)),
        title=f"Generate images from sample and query",
        description=description,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )
