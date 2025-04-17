from aiortc import RTCPeerConnection
from fastapi import FastAPI
import modal
import websockets

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
    "argparse",
)


app = modal.App(
    "aoirtc-demo"
)

MINUTES = 60  # seconds
test_timeout = 0.5 * MINUTES



@app.cls(
    image=web_image,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebRTCServer:

    @modal.enter()
    def init(self):
        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
        # create peer connection with STUN server
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pc = RTCPeerConnection(configuration = config)

    @modal.exit()
    async def exit(self):
        await self.pc.close()       

    @modal.asgi_app(label="webrtc-server")
    def webapp(self):

        import json

        from fastapi import FastAPI, WebSocket

        from aioice.candidate import Candidate
        from aiortc import RTCSessionDescription, RTCConfiguration, RTCIceServer
        from aiortc.contrib.signaling import create_signaling, add_signaling_arguments, BYE

        web_app = FastAPI()

        # create peer connection with STUN server
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pc = RTCPeerConnection(configuration = config)
                

        @self.pc.on("datachannel")
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
                        await self.pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))
                        answer = await self.pc.createAnswer()
                        await self.pc.setLocalDescription(answer)
                        answer_msg = json.dumps({"sdp": self.pc.localDescription.sdp, "type": "answer"})
                        print(self.pc.iceConnectionState)
                        print(self.pc.iceGatheringState)
                        print(self.pc.signalingState)
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
        

@app.cls(
    image=web_image,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebRTCClient():


    @modal.enter()
    def init(self):

        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        self.connection_successful = False

        # create peer connection with STUN server
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pc = RTCPeerConnection(configuration = config)

    @modal.asgi_app(label="webrtc-client")
    def webapp(self):

        import json
        import time
        import urllib
        from fastapi import FastAPI
        import asyncio
        from aiortc import  RTCSessionDescription

        web_app = FastAPI()

        

        @web_app.get("/")
        async def start_client():

            # confirm server container is running
            print(f"Attempting to connect to server at {WebRTCServer().webapp.web_url}")
            up, start, delay = False, time.time(), 10
            while not up:
                try:
                    with urllib.request.urlopen(WebRTCServer().webapp.web_url) as response:
                        if response.getcode() == 200:
                            up = True
                except Exception:
                    if time.time() - start > test_timeout:
                        break
                    time.sleep(delay)

            assert up, f"Failed health check for server at {WebRTCServer().webapp.web_url}"

            print(f"Successful health check for server at {WebRTCServer().webapp.web_url}")

            

            channel = self.pc.createDataChannel("data")

            @self.pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                print(f"Client ICE connection state is {self.pc.iceConnectionState}")
                if self.pc.iceConnectionState == "failed":
                    await self.pc.close()


            @self.pc.on("signalingstatechange")
            async def on_signalingstatechange():
                print(f"Client signaling state is {self.pc.signalingState}")

            async def send_ping():
                channel.send("ping")

            @channel.on("open")
            def on_open():
                asyncio.ensure_future(send_ping())

            @channel.on("message")
            def on_message(message):

                if isinstance(message, str) and message.startswith("pong"):
                    self.connection_successful = True
                    print(f"Client received {message}")
                            
            ws_uri = WebRTCServer().webapp.web_url.replace("http", "ws") + "/ws"
            print(f"Connecting to {ws_uri}")
            async with websockets.connect(ws_uri) as websocket:

                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)

                while self.pc.iceGatheringState != "complete":
                    print(self.pc.iceGatheringState)
                    await asyncio.sleep(1)
                
                print(f"Sending offer: {self.pc.localDescription.sdp}")
                await websocket.send(json.dumps({"sdp": self.pc.localDescription.sdp, "type": offer.type}))
                answer = json.loads(await websocket.recv())
                print(f"Received answer: {answer}")
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))


        @web_app.get("/success")
        async def success():
            return {"success": self.connection_successful}
        
        return web_app
    
    @modal.exit()
    async def exit(self):
        self.connection_successful = False
        await self.pc.close()



@app.local_entrypoint()
def main():
    import json
    import time
    import urllib

    print(f"Running health check of cloud client at {WebRTCClient().webapp.web_url}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(WebRTCClient().webapp.web_url) as response:
                if response.getcode() == 200:
                    print(f"Cloud client is up at {WebRTCClient().webapp.web_url}")
                    up = True
                else:
                    print(f"Cloud client is not up at {WebRTCClient().webapp.web_url}")
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for cloud client at {WebRTCClient().webapp.web_url}"

    print(f"Successful health check for cloud client at {WebRTCClient().webapp.web_url}")

    print("Testing connection...")
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCClient().webapp.web_url + "/success",
        method="GET",
        headers=headers,
    )
    success = False
    now = time.time()
    while time.time() - now < test_timeout:
        with urllib.request.urlopen(req) as response:
            success = (json.loads(response.read().decode())["success"])
        try:
            assert success, "Connection failed"
            return
        except:
            print("Connection failed")
            time.sleep(1)
            pass
    assert success, "Connection failed"

    