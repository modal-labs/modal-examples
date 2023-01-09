import io
import os
from typing import Optional

from fastapi import Request

import modal

stub = modal.Stub("example-dalle-bot", image=modal.Image.debian_slim().pip_install("min-dalle"))

volume = modal.SharedVolume().persist("dalle-model-vol")

CACHE_PATH = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("slack-sdk"),
    secret=modal.Secret.from_name("dalle-bot-slack-secret"),
)
def post_to_slack(prompt: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(channels=channel_name, title=prompt, content=image_bytes)


@stub.function(gpu="A10G", shared_volumes={CACHE_PATH: volume})
async def run_minidalle(prompt: str, channel_name: Optional[str]):
    import torch
    from min_dalle import MinDalle

    model = MinDalle(
        models_root=CACHE_PATH,
        dtype=torch.float32,
        device="cuda",
        is_mega=True,
        is_reusable=True,
    )

    image = model.generate_image(
        text=prompt,
        seed=-1,
        grid_size=3,
        is_seamless=False,
        temperature=1,
        top_k=256,
        supercondition_factor=16,
        is_verbose=False,
    )

    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        if channel_name:
            post_to_slack.call(prompt, channel_name, buf.getvalue())
        return buf.getvalue()


# python-multipart is needed for fastapi form parsing.
@stub.webhook(method="POST", image=modal.Image.debian_slim().pip_install("python-multipart"))
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    # Deferred call to function.
    run_minidalle.spawn(prompt, body["channel_name"])
    return f"Running text2im for {prompt}."


# Entrypoint code so this can be easily run from the command line

OUTPUT_DIR = "/tmp/render"


@stub.local_entrypoint
def main(prompt: str = "martha stewart at burning man"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "output.png")
    img_bytes = run_minidalle.call(prompt, None)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Done! Your DALL-E output image is at '{output_path}'")
