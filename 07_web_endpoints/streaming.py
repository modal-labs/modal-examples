# ---
# cmd: ["modal", "serve", "07_web_endpoints/streaming.py"]
# ---

# # Deploy a FastAPI app with streaming responses

# This example shows how you can deploy a [FastAPI](https://fastapi.tiangolo.com/) app with Modal that streams results back to the client.

import asyncio
import time

import modal
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("example-fastapi-streaming", image=image)

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

# This `fastapi_app` also uses the fake video streamer async generator,
# passing it directly into `StreamingResponse`.


@web_app.get("/")
async def main():
    return StreamingResponse(
        fake_video_streamer(), media_type="text/event-stream"
    )


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app


# This `hook` web endpoint Modal function calls *another* Modal function,
# and it just works!


@app.function()
def sync_fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: some data\n".encode()
        time.sleep(1)


@app.function()
@modal.web_endpoint()
def hook():
    return StreamingResponse(
        sync_fake_video_streamer.remote_gen(), media_type="text/event-stream"
    )


# This `mapped` web endpoint Modal function does a parallel `.map` on a simple
# Modal function. Using `.starmap` also would work in the same fashion.


@app.function()
def map_me(i):
    time.sleep(i)  # stagger the results for demo purposes
    return f"hello from {i}\n"


@app.function()
@modal.web_endpoint()
def mapped():
    return StreamingResponse(
        map_me.map(range(10)), media_type="text/event-stream"
    )


# To try for yourself, run

# ```shell
# modal serve streaming.py
# ```

# and then send requests to the URLs that appear in the terminal output.

# Make sure that your client is not buffering the server response
# until it gets newline (\n) characters. By default browsers and `curl` are buffering,
# though modern browsers should respect the "text/event-stream" content type header being set.
