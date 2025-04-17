from aiortc import RTCPeerConnection
from fastapi import FastAPI
import modal

web_image_server = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
)

web_image_client = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
)

app = modal.App(
    "aoirtc-demo"
)

MINUTES = 60  # seconds
test_timeout = 5 * MINUTES



@app.function(
    image=web_image_client,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="webrtc-server")
def webrtc_server():

    from fastapi import FastAPI

    from aiortc import RTCIceCandidate, RTCSessionDescription
    from aiortc.contrib.signaling import create_signaling, BYE

    web_app = FastAPI()

    @web_app.get("/")
    async def get_server():

        signaling = create_signaling()
        pc = RTCPeerConnection()

        async def consume_signaling(pc, signaling):
            while True:
                obj = await signaling.receive()

                if isinstance(obj, RTCSessionDescription):
                    await pc.setRemoteDescription(obj)

                    if obj.type == "offer":
                        # send answer
                        await pc.setLocalDescription(await pc.createAnswer())
                        await signaling.send(pc.localDescription)
                elif isinstance(obj, RTCIceCandidate):
                    await pc.addIceCandidate(obj)
                elif obj is BYE:
                    print("Exiting")
                    break
                
        await signaling.connect()

        @pc.on("datachannel")
        def on_datachannel(channel):

            @channel.on("message")
            def on_message(message):

                if isinstance(message, str) and message.startswith("ping"):
                    # reply
                    channel.send("pong")

        await consume_signaling(pc, signaling)
    
    return web_app

@app.function(
    image=web_image_client,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="webrtc-client")
def webrtc_client():

    import json
    import time
    import urllib

    from fastapi import FastAPI
    import asyncio
    from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription
    from aiortc.contrib.signaling import create_signaling, BYE

    web_app = FastAPI()

    @web_app.get("/")
    async def get_client():
        print(f"Running health check for server at {webrtc_server.web_url}")
        up, start, delay = False, time.time(), 10
        while not up:
            try:
                with urllib.request.urlopen(webrtc_server.web_url + "/health") as response:
                    if response.getcode() == 200:
                        up = True
            except Exception:
                if time.time() - start > test_timeout:
                    break
                time.sleep(delay)

        assert up, f"Failed health check for server at {webrtc_server.web_url}"

        print(f"Successful health check for server at {webrtc_server.web_url}")

        signaling = create_signaling()
        pc = RTCPeerConnection()

        await signaling.connect()

        channel = pc.createDataChannel("chat")

        async def send_pings():
            while True:
                channel.send("ping %d")
                await asyncio.sleep(1)

        @channel.on("open")
        def on_open():
            asyncio.ensure_future(send_pings())

        @channel.on("message")
        def on_message(message):

            if isinstance(message, str) and message.startswith("pong"):
                print(f"Client received {message}")

        async def consume_signaling(pc, signaling):
            while True:
                obj = await signaling.receive()

                if isinstance(obj, RTCSessionDescription):
                    await pc.setRemoteDescription(obj)

                    if obj.type == "offer":
                        # send answer
                        await pc.setLocalDescription(await pc.createAnswer())
                        await signaling.send(pc.localDescription)
                elif isinstance(obj, RTCIceCandidate):
                    await pc.addIceCandidate(obj)
                elif obj is BYE:
                    print("Exiting")
                    break

        # send offer
        await pc.setLocalDescription(await pc.createOffer())
        await signaling.send(pc.localDescription)

        await consume_signaling(pc, signaling)

    return web_app



@app.local_entrypoint()
def main():
    import json
    import time
    import urllib

    print(f"Running health check of cloud client at {webrtc_client.web_url}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(webrtc_client.web_url + "/health") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for cloud clien at {webrtc_client.web_url}"

    print(f"Successful health check for cloud clien at {webrtc_client.web_url}")

