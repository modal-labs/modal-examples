# ---
# cmd: ["modal", "serve", "07_webhooks/streaming.py"]
# deploy: true
# ---
import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import modal

stub = modal.Stub("example-fastapi-streaming")

web_app = FastAPI()

# This asynchronous generator function simulates
# progressively returning data to the client. The `asyncio.sleep`
# is not necessary, but makes it easier to see the iterative behavior
# of the response.


async def fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: hello world!".encode()
        await asyncio.sleep(1.0)


# ASGI app with streaming handler.
#
# This `fastapi_app` also uses the fake video streamer async generator,
# passing it directly into `StreamingResponse`.


@web_app.get("/")
async def main():
    return StreamingResponse(
        fake_video_streamer(), media_type="text/event-stream"
    )


@stub.function()
@stub.asgi_app()
def fastapi_app():
    return web_app


# This `hook` web endpoint Modal function calls *another* Modal function,
# and it just works!


@stub.function()
def sync_fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: some data\n".encode()
        time.sleep(1)


@stub.function()
@stub.web_endpoint()
def hook():
    return StreamingResponse(
        sync_fake_video_streamer.call(), media_type="text/event-stream"
    )


# This `mapped` web endpoint Modal function does a parallel `.map` on a simple
# Modal function. Using `.starmap` also would work in the same fashion.


@stub.function()
def map_me(i):
    time.sleep(i)  # stagger the results for demo purposes
    return f"hello from {i}\n"


@stub.function()
@stub.web_endpoint()
def mapped():
    return StreamingResponse(
        map_me.map(range(10)), media_type="text/event-stream"
    )


# A collection of basic examples of a webhook streaming response.
#
#
# ```
# modal serve streaming.py
# ```
#
# To try out the webhook, ensure that your client is not buffering the server response
# until it gets newline (\n) characters. By default browsers and `curl` are buffering,
# though modern browsers should respect the "text/event-stream" content type header being set.
#
# ```shell
# curl --no-buffer https://modal-labs--example-fastapi-streaming-fastapi-app.modal.run
# curl --no-buffer https://modal-labs--example-fastapi-streaming-hook.modal.run
# curl --no-buffer https://modal-labs--example-fastapi-streaming-mapped.modal.run
# ````
