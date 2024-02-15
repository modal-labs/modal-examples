import io
import os
from typing import Optional

from fastapi import Request
from modal import Image, Secret, Stub, web_endpoint

CACHE_PATH = "/root/model_cache"


def load_model(device=None):
    import torch
    from min_dalle import MinDalle

    # Instantiate the model, which has the side-effect of persisting
    # the model to disk if it does not already exist.
    return MinDalle(
        models_root=CACHE_PATH,
        dtype=torch.float32,
        device=device,
        is_mega=True,
        is_reusable=True,
    )


stub = Stub(
    "example-dalle-bot",
    image=Image.debian_slim().pip_install("min-dalle").run_function(load_model),
)


@stub.function(
    image=Image.debian_slim().pip_install("slack-sdk"),
    secrets=[Secret.from_name("dalle-bot-slack-secret")],
)
def post_to_slack(prompt: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(
        channels=channel_name, title=prompt, content=image_bytes
    )


@stub.function(gpu="A10G")
async def run_minidalle(prompt: str, channel_name: Optional[str]):
    model = load_model(device="cuda")

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
            post_to_slack.remote(prompt, channel_name, buf.getvalue())
        return buf.getvalue()


# python-multipart is needed for fastapi form parsing.
@stub.function(
    image=Image.debian_slim().pip_install("python-multipart"),
)
@web_endpoint(
    method="POST",
)
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    # Deferred call to function.
    run_minidalle.spawn(prompt, body["channel_name"])
    return f"Running text2im for {prompt}."


# Entrypoint code so this can be easily run from the command line

OUTPUT_DIR = "/tmp/render"


@stub.local_entrypoint()
def main(prompt: str = "martha stewart at burning man"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "output.png")
    img_bytes = run_minidalle.remote(prompt, None)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Done! Your DALL-E output image is at '{output_path}'")
