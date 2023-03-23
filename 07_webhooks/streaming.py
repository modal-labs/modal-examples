# ---
# cmd: ["modal", "serve", "07_webhooks/streaming.py"]
# deploy: true
# ---
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import modal

stub = modal.Stub("example-fastapi-streaming")

web_app = FastAPI()


async def fake_video_streamer():
    for i in range(10):
        yield f"frame {i}: some fake video bytes\n".encode()
        await asyncio.sleep(1.0)


@web_app.get("/")
async def main():
    return StreamingResponse(fake_video_streamer())


@stub.asgi()
def fastapi_app():
    return web_app
