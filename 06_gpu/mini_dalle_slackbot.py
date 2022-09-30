import io
import os

import modal
from fastapi import Request

stub = modal.Stub(
    "dalle-bot", image=modal.Image.debian_slim().pip_install(["min-dalle"])
)

volume = modal.SharedVolume().persist("dalle-model-vol")

CACHE_PATH = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install(["slack-sdk"]),
    secret=modal.ref("dalle-bot-slack-secret"),
)
def post_to_slack(prompt: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(channels=channel_name, title=prompt, content=image_bytes)


@stub.function(gpu=True, shared_volumes={CACHE_PATH: volume})
async def run_minidalle(prompt: str, channel_name: str):
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

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    post_to_slack(prompt, channel_name, buf.getvalue())


# python-multipart is needed for fastapi form parsing.
@stub.webhook(
    method="POST", image=modal.Image.debian_slim().pip_install(["python-multipart"])
)
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    # Deferred call to function.
    run_minidalle.submit(prompt, body["channel_name"])
    return f"Running text2im for {prompt}."
