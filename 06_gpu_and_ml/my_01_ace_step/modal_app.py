import modal

stub = modal.Stub("ace-step-deploy")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1")
    .pip_install(
        "torch", "transformers", "diffusers[torch]", "accelerate", "safetensors", "pillow", "fastapi", "uvicorn[standard]"
    )
)

volume = modal.SharedVolume().persist("ace-step-cache")

@stub.function(image=image, shared_volumes={"/root/.cache": volume}, gpu="A10G", timeout=600)
def generate(prompt: str) -> bytes:
    from diffusers import DiffusionPipeline
    import torch
    from io import BytesIO

    pipe = DiffusionPipeline.from_pretrained(
        "ACE-Step/ACE-Step-v1-3.5B",
        torch_dtype=torch.float16,
    ).to("cuda")

    image = pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()