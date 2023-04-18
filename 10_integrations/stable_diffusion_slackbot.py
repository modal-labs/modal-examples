# ---
# cmd: ["python", "10_integrations/stable_diffusion_slackbot.py", "a photo of an astronaut riding a horse on mars"]
# output-directory: "/tmp/stable-diffusion"
# ---
# # Stable diffusion slackbot
#
# This tutorial shows you how to build a Slackbot that uses
# [stable diffusion](https://stability.ai/blog/stable-diffusion-public-release)
# to produce realistic images from text prompts on demand.
#
# ![stable diffusion slackbot](./stable_diff_screenshot.jpg)

# ## Basic setup

import io
import os
from typing import Optional

from modal import Image, Secret, SharedVolume, Stub, web_endpoint

# All Modal programs need a [`Stub`](/docs/reference/modal.Stub) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

stub = Stub("example-stable-diff-bot")

# ## Inference Function
#
# ### HuggingFace token
#
# We're going to use the pre-trained
# [stable diffusion model](https://github.com/runwayml/stable-diffusion-v1-5) in
# HuggingFace's `diffusers` library. To gain access, you need to sign in to your
# HuggingFace account ([sign up here](https://huggingface.co/join)) and request
# access on the [model card page](https://huggingface.co/runwayml/stable-diffusion-v1-5).
#
# Next, [create a HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the
# [secrets page](https://modal.com/secrets). Let's use the environment variable
# named `HUGGINGFACE_TOKEN`. Functions that inject this secret will have access
# to the environment variable.
#
# ![create a huggingface token](./huggingface_token.png)
#
# ### Model cache
#
# The `diffusers` library downloads the weights for a pre-trained model to a local
# directory, if those weights don't already exist. To decrease start-up time, we want
# this download to happen just once, even across separate function invocations.
# To accomplish this, we use a [`SharedVolume`](/docs/guide/shared-volumes), a
# writable volume that can be attached to Modal functions and persisted across function runs.

volume = SharedVolume().persist("stable-diff-model-vol")

# ### The actual function
#
# Now that we have our token and `SharedVolume` set up, we can put everything together.
#
# Let's define a function that takes a text prompt and an optional channel name
# (so we can post results to Slack if the value is set) and runs stable diffusion.
# The `@stub.function()` decorator declares all the resources this function will
# use: we configure it to use a GPU, run on an image that has all the packages we
# need to run the model, mount the `SharedVolume` to a path of our choice, and
# also provide it the secret that contains the token we created above.
#
# By setting the `cache_dir` argument for the model to the mount path of our
# `SharedVolume`, we ensure that the model weights are downloaded only once.

CACHE_PATH = "/root/model_cache"


@stub.function(
    gpu="A10G",
    image=(
        Image.debian_slim()
        .run_commands(
            "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"
        )
        .pip_install("diffusers", "transformers", "scipy", "ftfy", "accelerate")
    ),
    shared_volumes={CACHE_PATH: volume},
    secret=Secret.from_name("huggingface-secret"),
)
async def run_stable_diffusion(prompt: str, channel_name: Optional[str] = None):
    from diffusers import StableDiffusionPipeline
    from torch import float16

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        revision="fp16",
        torch_dtype=float16,
        cache_dir=CACHE_PATH,
        device_map="auto",
    )

    image = pipe(prompt, num_inference_steps=100).images[0]

    # Convert PIL Image to PNG byte array.
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

    if channel_name:
        # `post_image_to_slack` is implemented further below.
        post_image_to_slack.call(prompt, channel_name, img_bytes)

    return img_bytes


# ## Slack webhook
#
# Now that we wrote our function, we'd like to trigger it from Slack. We can do
# this with [slash commands](https://api.slack.com/interactivity/slash-commands)
# — a feature that lets you register prefixes (such as `/run-my-bot`) to
# trigger webhooks of your choice.
#
# To serve our model as a web endpoint, we apply the
# [`@stub.web_endpoint`](/docs/guide/webhooks#web_endpoint) decorator in addition to
# `@stub.function()`. Modal webhooks are [FastAPI](https://fastapi.tiangolo.com/)
# endpoints by default (though we accept any ASGI web framework). This webhook
# retrieves the form body passed from Slack.
#
# Instead of blocking on the result of the stable diffusion model (which could
# take some time), we want to notify the user immediately that their request
# is being processed. Modal Functions let you
# [`spawn`](/docs/reference/modal.Function#spawn) an input without waiting for
# the results, which we use here to kick off model inference as a background task.

from fastapi import Request


@stub.function()
@web_endpoint(method="POST")
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    run_stable_diffusion.spawn(prompt, body["channel_name"])
    return f"Running stable diffusion for {prompt}."


# ## Post to Slack
#
# Finally, let's define a function to post images to a Slack channel.
#
# First, we need to create a Slack app and store the token for our app as a
# Modal secret. To do so, visit the the Modal [Secrets](/secrets) page and click
# "create a Slack secret". Then, you will find instructions on how to create a
# Slack app, give it OAuth permissions, and get a token. Note that you need to
# add the `file:write` OAuth scope to the created app.
#
# ![create a slack secret](./slack_secret.png)
#
# Below, we use the secret and `slack-sdk` to post to a Slack channel.


@stub.function(
    image=Image.debian_slim().pip_install("slack-sdk"),
    secret=Secret.from_name("stable-diff-slackbot-secret"),
)
def post_image_to_slack(title: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(channels=channel_name, title=title, content=image_bytes)


# ## Deploy the Slackbot
#
# That's all the code we need! To deploy your application, run
#
# ```shell
# modal deploy stable_diffusion_slackbot.py
# ```
#
# If successful, this will print a URL for your new webhook. To point your Slack
# app at it:
#
# - Go back to the [Slack apps page](https://api.slack.com/apps/).
# - Find your app and navigate to "Slash Commands" under "Features" in the left
#   sidebar.
# - Click on "Create New Command" and paste the webhook URL from Modal into the
#   "Request URL" field.
# - Name the command whatever you like, and hit "Save".
# - Reinstall the app to your workspace.
#
# We're done! 🎉 Install the app to any channel you're in, and you can trigger it
# with the command you chose above.
#
# ## Run Manually
#
# We can also trigger `run_stable_diffusion` manually for easier debugging.


@stub.local_entrypoint()
def run(
    prompt: str = "oil painting of a shiba",
    output_dir: str = "/tmp/stable-diffusion",
):
    os.makedirs(output_dir, exist_ok=True)
    img_bytes = run_stable_diffusion.call(prompt)
    output_path = os.path.join(output_dir, "output.png")
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Wrote data to {output_path}")


# This code lets us call our script as follows:
#
# ```shell
# modal run stable_diffusion_slackbot.py "a photo of an astronaut riding a horse on mars"
# ```
#
# The resulting image can be found in `/tmp/stable-diffusion/output.png`.
