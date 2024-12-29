# ---
# lambda-test: false
# ---
# # Create a Discord bot on modal

# In this example we will build a discord bot that given a city as input, tells us the weather in the city for that day
# We can do this using slash commands](https://discord.com/developers/docs/interactions/application-commands)
# a feature that lets you register a text command on Discord that triggers a custom webhook when a user interacts with it.
# We handle all our Discord events in a [FastAPI app](https://fastapi.tiangolo.com/). Luckily,
# we can deploy this app easily and serverlessly using Modal’s
# [@asgi_app](/docs/guide/webhooks#serving-asgi-and-wsgi-apps) decorator.

# ## Create a Discord app

# To connect our model to a Discord bot, we’re first going to create an
# application on the Discord Developer Portal.

# 1. Go to the
#    [Discord Developer Portal](https://discord.com/developers/applications) and
#    login with your Discord account.
# 2. On the portal, go to **Applications** and create a new application by
#    clicking **New Application** in the top right next to your profile picture.
# 3. Create a custom[create a custom Modal secret](/docs/guide/secrets) for your Discord bot.
#    On Modal's secret creation page, select 'Discord'. Copy your Discord application’s
#    **Public Key** (in **General Information**) and paste the value of the public key
#    as the value of the `DISCORD_PUBLIC_KEY` environment variable.
#    Name this secret `weather-discord-secret`.

#### Register a Slash Command

# Next, we’re going to register a command for our Discord app via an HTTP
# endpoint.

# Run the following command in your terminal, replacing the appropriate variable
# inputs. `BOT_TOKEN` can be found by resetting your token in the application’s
# **Bot** section, and `CLIENT_ID` is the **Application ID** available in
# **General Information**.


#```shell
# BOT_TOKEN='replace_with_bot_token'
# CLIENT_ID='replace_with_client_token'
# curl -X POST \
# -H 'Content-Type: application/json' \
# -H "Authorization: Bot $BOT_TOKEN" \
# -d '{
#   "name":"get_weather",
#   "description":"get weather",
#   "options":[
#     {
#       "name":"city",
#       "description":"The city for which you want to get the weather",
#       "type":3,
#       "required":true
#     }
#   ]
# }' "https://discord.com/api/v10/applications/$CLIENT_ID/commands"
#```


# This will register a Slash Command for your bot named `get_weather`, and has a
# parameter called `city`. More information about the
# command structure can be found in the Discord docs
# [here](https://discord.com/developers/docs/interactions/application-commands).

#### Defining the asgi app with modal

# We now create a `POST /get_weather` endpoint using [FastAPI](https://fastapi.tiangolo.com/) and Modal's
# [@asgi_app](/docs/guide/webhooks#serving-asgi-and-wsgi-apps) decorator to handle
# interactions with our Discord app (so that every time a user does a slash
# command, we can respond to it).

# Let's get the imports out of the way and define an [`App`](https://modal.com/docs/reference/modal.App)

import json

from modal import App, Image, Secret, asgi_app

app = App("discord-weather-bot")

# We define an [image](https://modal.com/docs/guide/images) that has the [`python-weather`](https://github.com/null8626/python-weather) package, and
# the [FastAPI](https://fastapi.tiangolo.com/) package installed.

image = Image.debian_slim(python_version="3.8").pip_install("python-weather==2.0.7", "fastapi[standard]==0.115.4", "pynacl==1.5.0")

# We define a function that uses the python_weather library to get the weather of a city
# Note that since Discord requires an interaction response within 3 seconds, we
# use [`spawn`](/docs/reference/modal.Function#spawn) to kick off
# `get_weather_for_city`as a background task from the asgi app while returning a `defer` message to
# Discord within the time limit. We then update our response with the results once
# the model has finished running. The

@app.function(image = image)
async def get_weather_forecast_for_city(city: str, interaction_token, application_id):
    import aiohttp
    from python_weather import IMPERIAL, Client, Error, RequestError

    interaction_url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}/messages/@original"
    async with Client(unit=IMPERIAL) as client:
        try:
            weather = await client.get(city)
            daily_forecasts = "\n".join([f"Date: {daily.date}, Highest temperature: {daily.highest_temperature}°F, Lowest Temperature: {daily.lowest_temperature}°F" for daily in weather])
            message = f"The forecast for {weather.location} is as follows:\n{daily_forecasts}"
        except RequestError:
            message = "An error occurred, issue with connecting to weather api"
        except Error:
            message = "An error occurred, please check city name"

    json_payload = {"content": message}
    async with aiohttp.ClientSession() as session:
        async with session.patch(interaction_url, json=json_payload) as resp:
            print(await resp.text())

# We now define an asgi app using the Modal ASGI syntax [@asgi_app](/docs/guide/webhooks#serving-asgi-and-wsgi-apps).

@app.function(
    secrets=[Secret.from_name("advay-discord-secret")],
    keep_warm=1, # eliminates risk of container startup making discord ack time too long
    image = image

)
@asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()

    # allow CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/get_weather_forecast")
    async def get_weather_forecast_api(request: Request):

        import os

        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey


        # Verify the request using the Discord public key
        public_key = os.getenv("DISCORD_PUBLIC_KEY")
        verify_key = VerifyKey(bytes.fromhex(public_key))

        signature = request.headers.get("X-Signature-Ed25519")
        timestamp = request.headers.get("X-Signature-Timestamp")
        body = await request.body()

        message = timestamp.encode() + body

        try:
            verify_key.verify(message, bytes.fromhex(signature))
        except BadSignatureError:
            raise HTTPException(status_code=401, detail="Invalid request")


        # Parse request
        data = json.loads(body.decode())
        if data.get("type") == 1:  # ack ping from Discord
            return {"type": 1}

        if data.get("type") == 2:  # triggered by slash command interaction
            options = data["data"]["options"]
            for option in options:
                name = option["name"]
                if name == "city":
                    city = option["value"]


            app_id = data["application_id"]
            interaction_token = data["token"]

            # Kick off request asynchronously, send value when we have it
            get_weather_forecast_for_city.spawn(city, interaction_token, app_id)

            return {
                "type": 5,  # respond immediately with defer message
            }

        raise HTTPException(status_code=400, detail="Bad request")

    return web_app

#### Deploy the Modal web endpoint
# You can deploy this app by running the following command from your root
# directory:

# ```shell
# modal deploy discord_bot.py
# ```

# Copy the Modal URL that is printed in the output and go back to your
# application's **General Information** section on the
# [Discord Developer Portal](https://discord.com/developers/applications). Paste
# the URL, making sure to append the path of your `POST` endpoint, in the
# **Interactions Endpoint URL** field, then click **Save Changes**. If your
# endpoint is valid, it will properly save and you can start receiving
# interactions via this web endpoint.

#### Finish setting up Discord bot

# To start using the Slash Command you just set up, you need to invite the bot to
# a Discord server. To do so, go to your application's **OAuth2** section on the
# [Discord Developer Portal](https://discord.com/developers/applications). Select
# `applications.commands` as the scope of your bot and copy the invite URL that is
# generated at the bottom of the page.

# Paste this URL in your browser, then select your desired server (create a new
# server if needed) and click **Authorize**. Now you can open your Discord server
# and type `/{name of your slash command}` - your bot should be connected and
# ready for you to use!
