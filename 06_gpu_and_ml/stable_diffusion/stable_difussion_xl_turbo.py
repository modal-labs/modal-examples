from pathlib import Path

from modal import Image, Stub, gpu, method


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)


image = (
    Image.debian_slim()
    .pip_install(
        "Pillow~=10.1.0",
        "diffusers~=0.24",
        "transformers~=4.35",
        "accelerate~=0.25",
        "safetensors~=0.4",
    ).run_function(download_models)
)

stub = Stub("stable-diffusion-xl-turbo", image=image)


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
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
    def inference(self, image_bytes, prompt):
        from io import BytesIO

        from diffusers.utils import load_image
        from PIL import Image

        init_image = load_image(Image.open(BytesIO(image_bytes))).resize(
            (512, 512)
        )
        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=4,
            strength=0.9,
            guidance_scale=0.0,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


@stub.local_entrypoint()
def main(
    image_path="demo_images/dog.png",
    prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
):
    with open(image_path, "rb") as image_file:
        input_image_bytes = image_file.read()
        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion-xl-turbo")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(output_image_bytes)
