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

# This is 'faker' asynchronous generator function simulates
# progressively returning data to the client. The `asyncio.sleep`
# is not necessary, but makes it easier to see the iterative behavior
# of the response.


async def fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: hello world!".encode()
        await asyncio.sleep(1.0)


@web_app.get("/")
async def main():
    return StreamingResponse(
        fake_video_streamer(), media_type="text/event-stream"
    )


@stub.function()
@stub.asgi_app()
def fastapi_app():
    return web_app


@stub.function(is_generator=True)
def sync_fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: some data\n".encode()
        time.sleep(1)


@stub.function()
@stub.web_endpoint()
def hook():
    return StreamingResponse(
        iter(sync_fake_video_streamer.call()), media_type="text/event-stream"
    )


# This is a very basic 'hello world' example of a webhook streaming response.
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
# ````
