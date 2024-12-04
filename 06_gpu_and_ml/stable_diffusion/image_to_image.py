# ---
# output-directory: "/tmp/stable-diffusion"
# ---

# # Transform images with SDXL Turbo

# In this example, we run the SDXL Turbo model in _image-to-image_ mode:
# the model takes in a prompt and an image and transforms the image to better match the prompt.

# For example, the model transformed the image on the left into the image on the right based on the prompt
# _dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k_.

# ![](https://modal-cdn.com/cdnbot/sd-im2im-dog-8sanham3_915c7d4c.webp)

# SDXL Turbo is a distilled model designed for fast, interactive image synthesis.
# Learn more about it [here](https://stability.ai/news/stability-ai-sdxl-turbo).

# ## Define a container image

# First, we define the environment the model inference will run in,
# the [container image](https://modal.com/docs/guide/custom-container).

from io import BytesIO
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate~=0.25.0",  # Allows `device_map="auto"``, for computation of optimized device_map
        "diffusers~=0.24.0",  # Provides model libraries
        "huggingface-hub[hf-transfer]~=0.25.2",  # Lets us download models from Hugging Face's Hub
        "Pillow~=10.1.0",  # Image manipulation in Python
        "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
        "transformers~=4.35.2",  # This is needed for `import torch`
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # allow faster model downloads
)

app = modal.App("image-to-image", image=image)

with image.imports():
    import torch
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download
    from PIL import Image


# ## Downloading, setting up, and running SDXL Turbo

# The Modal `Cls` defined below contains all the logic to download, set up, and run SDXL Turbo.

# The [container lifecycle](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta) decorators
# `@build` and `@enter` ensure we download the model when building our container image and load it into memory
# when we start up a new instance of our `Cls`.

# The `inference` method runs the actual model inference. It takes in an image as a collection of `bytes` and a string `prompt` and returns
# a new image (also as a collection of `bytes`).

# To avoid excessive cold-starts, we set the `container_idle_timeout` to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down.


@app.cls(gpu=modal.gpu.A10G(), container_idle_timeout=240)
class Model:
    @modal.build()
    def download_models(self):
        # Ignore files that we don't need to speed up download time.
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]

        snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)

    @modal.enter()
    def enter(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
        )

    @modal.method()
    def inference(
        self, image_bytes: bytes, prompt: str, strength: float = 0.9
    ) -> bytes:
        init_image = load_image(Image.open(BytesIO(image_bytes))).resize(
            (512, 512)
        )
        num_inference_steps = 4
        # "When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1"
        # See: https://huggingface.co/stabilityai/sdxl-turbo
        assert num_inference_steps * strength >= 1

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# ## Running the model from the command line

# You can run the model from the command line with

# ```bash
# modal run image_to_image.py
# ```

# Use `--help` for additional details.


@app.local_entrypoint()
def main(
    image_path=Path(__file__).parent / "demo_images/dog.png",
    prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
    strength=0.9,  # increase to favor the prompt over the baseline image
):
    print(f"🎨 reading input image from {image_path}")
    input_image_bytes = Path(image_path).read_bytes()
    print(f"🎨 editing image with prompt {prompt}")
    output_image_bytes = Model().inference.remote(input_image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion")
    dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"🎨 saving output image to {output_path}")
    output_path.write_bytes(output_image_bytes)
