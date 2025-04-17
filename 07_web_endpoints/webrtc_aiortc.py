from aiortc import RTCPeerConnection
from fastapi import FastAPI
import modal
import websockets

web_image_server = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
    "argparse",
)

web_image_client = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
    "argparse",
)

app = modal.App(
    "aoirtc-demo"
)

MINUTES = 60  # seconds
test_timeout = 5 * MINUTES



@app.function(
    image=web_image_server,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="webrtc-server")
def webrtc_server():

    import json

    from fastapi import FastAPI, WebSocket

    from aioice.candidate import Candidate
    from aiortc import RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.signaling import create_signaling, add_signaling_arguments, BYE

    web_app = FastAPI()

     # create peer connection with STUN server
    config = RTCConfiguration()
    config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
    pc = RTCPeerConnection(configuration = config)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"Server ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            print(f"Received message: {message}")
            channel.send("pong")

    @web_app.get("/")
    async def get_server():
        pass

    @web_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket):
        await websocket.accept()

        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                print(f"Received message: {data}")
                if data.get("type") == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    answer_msg = json.dumps({"sdp": pc.localDescription.sdp, "type": "answer"})
                    print(pc.iceConnectionState)
                    print(pc.iceGatheringState)
                    print(pc.signalingState)
                    print(f"Sending answer: {answer_msg}")
                    

                    await websocket.send_text(answer_msg)
                elif data.get("type") == "bye":
                    print("Exiting")
                    break
                else:
                    print(f"Unknown message type: {data.get('type')}")
            except Exception as e:
                print(f"Error: {e}")
                break



    return web_app

@app.function(
    image=web_image_client,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="webrtc-client")
def webrtc_client():

    import json
    import time
    import urllib
    import argparse
    from fastapi import FastAPI, WebSocket
    import asyncio
    from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.signaling import create_signaling, add_signaling_arguments, BYE

    web_app = FastAPI()

    @web_app.get("/")
    async def start_client():

        # confirm server container is running
        print(f"Attempting to connect to server at {webrtc_server.web_url}")
        up, start, delay = False, time.time(), 10
        while not up:
            try:
                with urllib.request.urlopen(webrtc_server.web_url) as response:
                    if response.getcode() == 200:
                        up = True
            except Exception:
                if time.time() - start > test_timeout:
                    break
                time.sleep(delay)

        assert up, f"Failed health check for server at {webrtc_server.web_url}"

        print(f"Successful health check for server at {webrtc_server.web_url}")

        # create peer connection with STUN server
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        pc = RTCPeerConnection(configuration = config)

        channel = pc.createDataChannel("data")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"Client ICE connection state is {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()


        @pc.on("signalingstatechange")
        async def on_signalingstatechange():
            print(f"Client signaling state is {pc.signalingState}")

        async def send_pings():
            while True:
                channel.send("ping")
                await asyncio.sleep(1)

        @channel.on("open")
        def on_open():
            asyncio.ensure_future(send_pings())

        @channel.on("message")
        def on_message(message):

            if isinstance(message, str) and message.startswith("pong"):
                print(f"Client received {message}")


        # send offer
       
        ws_uri = webrtc_server.web_url.replace("http", "ws") + "/ws"
        print(f"Connecting to {ws_uri}")
        async with websockets.connect(ws_uri) as websocket:

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            while pc.iceGatheringState != "complete":
                print(pc.iceGatheringState)
                await asyncio.sleep(1)

            print(pc.iceConnectionState)
            print(pc.iceGatheringState)
            print(pc.signalingState)
            
            print(f"Sending offer: {offer.sdp}")
            await websocket.send(json.dumps({"sdp": pc.localDescription.sdp, "type": offer.type}))
            answer = json.loads(await websocket.recv())
            print(f"Received answer: {answer}")
            await pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))

            print(pc.iceConnectionState)
            print(pc.iceGatheringState)
            print(pc.signalingState)


            while True:
                print(pc.iceConnectionState)
                print(pc.iceGatheringState)
                print(pc.signalingState)

                await asyncio.sleep(1)
            # answer = await pc.createAnswer()
            # await pc.setLocalDescription(answer)
            # await websocket.send(pc.localDescription.sdp)

            # answer_sdp = await websocket.recv()
            # await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

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
            with urllib.request.urlopen(webrtc_client.web_url) as response:
                if response.getcode() == 200:
                    print(f"Cloud client is up at {webrtc_client.web_url}")
                    up = True
                else:
                    print(f"Cloud client is not up at {webrtc_client.web_url}")
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for cloud client at {webrtc_client.web_url}"

    print(f"Successful health check for cloud client at {webrtc_client.web_url}")

