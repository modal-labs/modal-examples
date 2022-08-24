# ---
# integration-test: false
# ---

import io
import os
from typing import Optional

from fastapi import Request

import modal

stub = modal.Stub(
    "stable-diff-bot", image=modal.DebianSlim().pip_install(["diffusers", "transformers", "scipy", "ftfy"])
)

volume = modal.SharedVolume().persist("stable-diff-model-vol")

CACHE_PATH = "/root/model_cache"


@stub.function(image=modal.DebianSlim().pip_install(["slack-sdk"]), secret=modal.ref("stable-diff-secret"))
def post_to_slack(prompt: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(channels=channel_name, title=prompt, content=image_bytes)


@stub.function(gpu=True, shared_volumes={CACHE_PATH: volume}, secret=modal.ref("stable-diff-secret"))
async def run_stable_diffusion(prompt: str, channel_name: Optional[str] = None):
    from diffusers import StableDiffusionPipeline
    from torch import autocast

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        cache_dir=CACHE_PATH,
    ).to("cuda")

    with autocast("cuda"):
        image = pipe(prompt, num_inference_steps=100)["sample"][0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    if channel_name:
        post_to_slack(prompt, channel_name, img_bytes)

    return img_bytes


# python-multipart is needed for fastapi form parsing.
@stub.webhook(method="POST", image=modal.DebianSlim().pip_install(["python-multipart"]))
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    # Start the function call but return immediately.
    run_stable_diffusion.submit(prompt, body["channel_name"])
    return f"Running stable diffusion for {prompt}."


OUTPUT_DIR = "/tmp/render"

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "oil painting of a shiba"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with stub.run():
        for idx, img_bytes in enumerate(run_stable_diffusion.map([prompt] * 3)):
            with open(os.path.join(OUTPUT_DIR, "output_%d.png" % idx), "wb") as f:
                f.write(img_bytes)
