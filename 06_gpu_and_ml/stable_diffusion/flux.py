# ---
# output-directory: "/tmp/flux"
# ---
# example originally contributed by [@Arro](https://github.com/Arro)
from io import BytesIO
from pathlib import Path

import modal

VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF

diffusers_commit_sha = "1fcb811a8e6351be60304b1d4a4a749c36541651"

flux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .run_commands(
        f"pip install git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} 'numpy<2'"
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
    )
)

app = modal.App("example-flux")

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline


@app.cls(
    gpu=modal.gpu.A100(size="40GB"),
    container_idle_timeout=100,
    image=flux_image,
)
class Model:
    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download

        snapshot_download(f"black-forest-labs/FLUX.1-{VARIANT}")

        self.pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

    @modal.method()
    def inference(self, prompt):
        print("Generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,  # use a larger number if you are using [dev], smaller for [schnell]
        ).images[0]
        print("Generated.")

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


@app.local_entrypoint()
def main(
    prompt: str = "a computer screen showing ASCII terminal art of the word 'Modal' in neon green. two programmers are pointing excitedly at the screen.",
):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/flux")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.jpg"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
