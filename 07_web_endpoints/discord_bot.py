# ---
# deploy: true
# ---

# # Serve a Discord Bot on Modal

# In this example we will demonstrate how to use Modal to build and serve a Discord bot that uses
# [slash commands](https://discord.com/developers/docs/interactions/application-commands).

# Slash commands send information from Discord server members to a service at a URL.
# Here, we set up a simple [FastAPI app](https://fastapi.tiangolo.com/)
# to run that service and deploy it easily  Modal’s
# [`@asgi_app`](https://modal.com/docs/guide/webhooks#serving-asgi-and-wsgi-apps) decorator.

# As our example service, we hit a simple free API:
# the [Free Public APIs API](https://www.freepublicapis.com/api),
# a directory of free public APIs.

# [Try it out on Discord](https://discord.gg/PmG7P47EPQ)!

# ## Set up our App and its Image

# First, we define the [container image](https://modal.com/docs/guide/images)
# that all the pieces of our bot will run in.

# We set that as the default image for a Modal [App](https://modal.com/docs/guide/apps).
# The App is where we'll attach all the components of our bot.

import json
from enum import Enum

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.4", "pynacl~=1.5.0", "requests~=2.32.3"
)

app = modal.App("example-discord-bot", image=image)

# ## Hit the Free Public APIs API

# We start by defining the core service that our bot will provide.

# In a real application, this might be [music generation](https://modal.com/docs/examples/musicgen),
# a [chatbot](https://modal.com/docs/examples/chat_with_pdf_vision),
# or [interacting with a database](https://modal.com/docs/examples/cron_datasette).

# Here, we just hit a simple free public API:
# the [Free Public APIs](https://www.freepublicapis.com) API,
# an "API of APIs" that returns information about free public APIs,
# like the [Global Shark Attack API](https://www.freepublicapis.com/global-shark-attack-api)
# and the [Corporate Bullshit Generator](https://www.freepublicapis.com/corporate-bullshit-generator).
# We convert the response into a Markdown-formatted message.

# We turn our Python function into a Modal Function by attaching the `app.function` decorator.
# We make the function `async` and add `@modal.concurrent()` with a large `max_inputs` value, because
# communicating with an external API is a classic case for better performance from asynchronous execution.
# Modal handles things like the async event loop for us.


@app.function()
@modal.concurrent(max_inputs=1000)
async def fetch_api() -> str:
    import aiohttp

    url = "https://www.freepublicapis.com/api/random"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                message = (
                    f"# {data.get('emoji') or '🤖'} [{data['title']}]({data['source']})"
                )
                message += f"\n _{''.join(data['description'].splitlines())}_"
        except Exception as e:
            message = f"# 🤖: Oops! {e}"

    return message


# This core component has nothing to do with Discord,
# and it's nice to be able to interact with and test it in isolation.

# For that, we add a `local_entrypoint` that calls the Modal Function.
# Notice that we add `.remote` to the function's name.

# Later, when you replace this component of the app with something more interesting,
# test it by triggering this entrypoint with  `modal run discord_bot.py`.


@app.local_entrypoint()
def test_fetch_api():
    result = fetch_api.remote()
    if result.startswith("# 🤖: Oops! "):
        raise Exception(result)
    else:
        print(result)


# ## Integrate our Modal Function with Discord Interactions

# Now we need to map this function onto Discord's interface --
# in particular the [Interactions API](https://discord.com/developers/docs/interactions/overview).

# Reviewing the documentation, we see that we need to send a JSON payload
# to a specific API URL that will include an `app_id` that identifies our bot
# and a `token` that identifies the interaction (loosely, message) that we're participating in.

# So let's write that out. This function doesn't need to live on Modal,
# since it's just encapsulating some logic -- we don't want to turn it into a service or an API on its own.
# That means we don't need any Modal decorators.


async def send_to_discord(payload: dict, app_id: str, interaction_token: str):
    import aiohttp

    interaction_url = f"https://discord.com/api/v10/webhooks/{app_id}/{interaction_token}/messages/@original"

    async with aiohttp.ClientSession() as session:
        async with session.patch(interaction_url, json=payload) as resp:
            print("🤖 Discord response: " + await resp.text())


# Other parts of our application might want to both hit the Free Public APIs API and send the result to Discord,
# so we both write a Python function for this and we promote it to a Modal Function with a decorator.

# Notice that we use the `.local` suffix to call our `fetch_api` Function. That means we run
# the Function the same way we run all the other Python functions, rather than treating it as a special
# Modal Function. This reduces a bit of extra latency, but couples these two Functions more tightly.


@app.function()
@modal.concurrent(max_inputs=1000)
async def reply(app_id: str, interaction_token: str):
    message = await fetch_api.local()
    await send_to_discord({"content": message}, app_id, interaction_token)


# ## Set up a Discord app

# Now, we need to actually connect to Discord.
# We start by creating an application on the Discord Developer Portal.

# 1. Go to the
#    [Discord Developer Portal](https://discord.com/developers/applications) and
#    log in with your Discord account.
# 2. On the portal, go to **Applications** and create a new application by
#    clicking **New Application** in the top right next to your profile picture.
# 3. [Create a custom Modal Secret](https://modal.com/docs/guide/secrets) for your Discord bot.
#    On Modal's Secret creation page, select 'Discord'. Copy your Discord application’s
#    **Public Key** and **Application ID** (from the **General Information** tab in the Discord Developer Portal)
#    and paste them as the value of `DISCORD_PUBLIC_KEY` and `DISCORD_CLIENT_ID`.
#    Additionally, head to the **Bot** tab and use the **Reset Token** button to create a new bot token.
#    Paste this in the value of an additional key in the Secret, `DISCORD_BOT_TOKEN`.
#    Name this Secret `discord-secret`.

# We access that Secret in code like so:

discord_secret = modal.Secret.from_name(
    "discord-secret",
    required_keys=[  # included so we get nice error messages if we forgot a key
        "DISCORD_BOT_TOKEN",
        "DISCORD_CLIENT_ID",
        "DISCORD_PUBLIC_KEY",
    ],
)

# ## Register a Slash Command

# Next, we’re going to register a [Slash Command](https://discord.com/developers/docs/interactions/application-commands#slash-commands)
# for our Discord app. Slash Commands are triggered by users in servers typing `/` and the name of the command.

# The Modal Function below will register a Slash Command for your bot named `bored`.
# More information about Slash Commands can be found in the Discord docs
# [here](https://discord.com/developers/docs/interactions/application-commands).

# You can run this Function with

# ```bash
# modal run discord_bot::create_slash_command
# ```


@app.function(secrets=[discord_secret], image=image)
def create_slash_command(force: bool = False):
    """Registers the slash command with Discord. Pass the force flag to re-register."""
    import os

    import requests

    BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bot {BOT_TOKEN}",
    }
    url = f"https://discord.com/api/v10/applications/{CLIENT_ID}/commands"

    command_description = {
        "name": "api",
        "description": "Information about a random free, public API",
    }

    # first, check if the command already exists
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
    except Exception as e:
        raise Exception("Failed to create slash command") from e

    commands = response.json()
    command_exists = any(
        command.get("name") == command_description["name"] for command in commands
    )

    # and only recreate it if the force flag is set
    if command_exists and not force:
        print(f"🤖: command {command_description['name']} exists")
        return

    response = requests.post(url, headers=headers, json=command_description)
    try:
        response.raise_for_status()
    except Exception as e:
        raise Exception("Failed to create slash command") from e
    print(f"🤖: command {command_description['name']} created")


# ## Host a Discord Interactions endpoint on Modal

# If you look carefully at the definition of the Slash Command above,
# you'll notice that it doesn't know anything about our bot besides an ID.

# To hook the Slash Commands in the Discord UI up to our logic for hitting the Bored API,
# we need to set up a service that listens at some URL and follows a specific protocol,
# described [here](https://discord.com/developers/docs/interactions/overview#configuring-an-interactions-endpoint-url).

# Here are some of the most important facets:

# 1. We'll need to respond within five seconds or Discord will assume we are dead.
# Modal's fast-booting serverless containers usually start faster than that,
# but it's not guaranteed. So we'll add the `min_containers` parameter to our
# Function so that there's at least one live copy ready to respond quickly at any time.
# Modal charges a minimum of about 2¢ an hour for live containers (pricing details [here](https://modal.com/pricing)).
# Note that that still fits within Modal's $30/month of credits on the free tier.

# 2. We have to respond to Discord that quickly, but we don't have to respond to the user that quickly.
# We instead send an acknowledgement so that they know we're alive and they can close their connection to us.
# We also trigger our `reply` Modal Function, which will respond to the user via Discord's Interactions API,
# but we don't wait for the result, we just `spawn` the call.

# 3. The protocol includes some authentication logic that is mandatory
# and checked by Discord. We'll explain in more detail in the next section.

# We can set up our interaction endpoint by deploying a FastAPI app on Modal.
# This is as easy as creating a Python Function that returns a FastAPI app
# and adding the `modal.asgi_app` decorator.
# For more details on serving Python web apps on Modal, see
# [this guide](https://modal.com/docs/guide/webhooks).


@app.function(secrets=[discord_secret], min_containers=1)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()

    # must allow requests from other domains, e.g. from Discord's servers
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/api")
    async def get_api(request: Request):
        body = await request.body()

        # confirm this is a request from Discord
        authenticate(request.headers, body)

        print("🤖: parsing request")
        data = json.loads(body.decode())
        if data.get("type") == DiscordInteractionType.PING.value:
            print("🤖: acking PING from Discord during auth check")
            return {"type": DiscordResponseType.PONG.value}

        if data.get("type") == DiscordInteractionType.APPLICATION_COMMAND.value:
            print("🤖: handling slash command")
            app_id = data["application_id"]
            interaction_token = data["token"]

            # kick off request asynchronously, will respond when ready
            reply.spawn(app_id, interaction_token)

            # respond immediately with defer message
            return {
                "type": DiscordResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE.value
            }

        print(f"🤖: unable to parse request with type {data.get('type')}")
        raise HTTPException(status_code=400, detail="Bad request")

    return web_app


# The authentication for Discord is a bit involved and there aren't,
# to our knowledge, any good Python libraries for it.

# So we have to implement the protocol "by hand".

# Essentially, Discord sends a header in their request
# that we can use to verify the request comes from them.
# For that, we use the `DISCORD_PUBLIC_KEY` from
# our Application Information page.

# The details aren't super important, but they appear in the `authenticate` function below
# (which defers the real cryptography work to [PyNaCl](https://pypi.org/project/PyNaCl/),
# a Python wrapper for [`libsodium`](https://github.com/jedisct1/libsodium)).

# Discord will also check that we reject unauthorized requests,
# so we have to be sure to get this right!


def authenticate(headers, body):
    import os

    from fastapi.exceptions import HTTPException
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    print("🤖: authenticating request")
    # verify the request is from Discord using their public key
    public_key = os.getenv("DISCORD_PUBLIC_KEY")
    verify_key = VerifyKey(bytes.fromhex(public_key))

    signature = headers.get("X-Signature-Ed25519")
    timestamp = headers.get("X-Signature-Timestamp")

    message = timestamp.encode() + body

    try:
        verify_key.verify(message, bytes.fromhex(signature))
    except BadSignatureError:
        # either an unauthorized request or Discord's "negative control" check
        raise HTTPException(status_code=401, detail="Invalid request")


# The code above used a few enums to abstract bits of the Discord protocol.
# Now that we've walked through all of it,
# we're in a position to understand what those are
# and so the code for them appears below.


class DiscordInteractionType(Enum):
    PING = 1  # hello from Discord during auth check
    APPLICATION_COMMAND = 2  # an actual command


class DiscordResponseType(Enum):
    PONG = 1  # hello back during auth check
    DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5  # we'll send a message later


# ## Deploy on Modal

# You can deploy this app on Modal by running the following commands:

# ``` shell
# modal run discord_bot.py  # checks the API wrapper, little test
# modal run discord_bot.py::create_slash_command  # creates the slash command, if missing
# modal deploy discord_bot.py  # deploys the web app and the API wrapper
# ```

# Copy the Modal URL that is printed in the output and go back to the **General Information** section on the
# [Discord Developer Portal](https://discord.com/developers/applications).
# Paste the URL, making sure to append the path of your `POST` route (here, `/api`), in the
# **Interactions Endpoint URL** field, then click **Save Changes**. If your
# endpoint URL is incorrect or if authentication is incorrectly implemented,
# Discord will refuse to save the URL. Once it saves, you can start
# handling interactions!

# ## Finish setting up Discord bot

# To start using the Slash Command you just set up, you need to invite the bot to
# a Discord server. To do so, go to your application's **Installation** section on the
# [Discord Developer Portal](https://discord.com/developers/applications).
# Copy the **Discored Provided Link** and visit it to invite the bot to your bot to the server.

# Now you can open your Discord server and type `/api` in a channel to trigger the bot.
# You can see a working version [in our test Discord server](https://discord.gg/PmG7P47EPQ).
