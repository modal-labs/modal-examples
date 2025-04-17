import modal

# pretty minimal image
web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "aiortc",
)

# instantiate our app
app = modal.App(
    "aoirtc-demo"
)

# set timeout for health checks and connection test
MINUTES = 60  # seconds
test_timeout = 0.5 * MINUTES

class WebRTCPeer:

    @modal.enter()
    def init(self):

        from aiortc import RTCPeerConnection
        from fastapi import FastAPI

        # create peer connection with STUN server
        self.pc = RTCPeerConnection()

        # aiortc automatically uses google's STUN server, but we can also specify our own
        # as shown below
        
        # config = RTCConfiguration()
        # config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        # self.pc = RTCPeerConnection(configuration = config)

        self.connection_successful = False
        self.web_app = FastAPI()

        @self.web_app.get("/success")
        async def success():
            return {"success": self.connection_successful}

    @modal.exit()
    async def exit(self):
        await self.pc.close()  
        self.connection_successful = False


# a class for our server. in this case server just means that this process is responding to an offer
# to establish a P2P connection as opposed to initiating the connection
@app.cls(
    image=web_image,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebRTCResponder(WebRTCPeer):   

    async def handle_offer(self, data):

        import json

        from aiortc import RTCSessionDescription
        # set remote description
        await self.pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))

        # create answer
        answer = await self.pc.createAnswer()

        # set local/our description, this also triggers ICE gathering
        await self.pc.setLocalDescription(answer)

        # send local description DSP
        # NOTE: we can't use `answer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return json.dumps({"sdp": self.pc.localDescription.sdp, "type": "answer"})  

    # add responder logic to webapp
    @modal.asgi_app(label="webrtc-server")
    def webapp(self):

        import json
        from fastapi import WebSocket

        # when a data channel is opened
        @self.pc.on("datachannel")
        def on_datachannel(channel):

            def pong():
                channel.send("pong")

            # add a message handler to the data channel
            @channel.on("message")
            def on_message(message):
                print(f"Received message: {message}\n")

                if message == "ping":
                    pong()
                    self.connection_successful = True
                

        # create root endpoint to use for health checks
        @self.web_app.get("/")
        async def get_server():
            pass

        # create websocket endpoint to handle incoming connections
        @self.web_app.websocket("/ws")
        async def on_connection_request(websocket: WebSocket):

            # accept websocket connection
            await websocket.accept()

            # handle websocket messages and loop for lifetime
            while True:
                try:
                    # get websocket message and parse as json
                    msg = json.loads(await websocket.receive_text())

                    # handle offer
                    if msg.get("type") == "offer":
                        print(f"Server received offer...")

                        reply = await self.handle_offer(msg)
                        await websocket.send_text(reply)
                        print(f"Server sent answer...")
                    else:
                        print(f"Unknown message type: {msg.get('type')}")
                except Exception as e:
                    print(f"Error: {e}")
                    break

        return self.web_app
    
    
        

@app.cls(
    image=web_image,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebRTCRequester(WebRTCPeer):

    # add initiator logic to webapp
    @modal.asgi_app(label="webrtc-client")
    def webapp(self):

        import json
        import time
        import urllib

        from aiortc import  RTCSessionDescription
        import websockets

        # create root endpoint to trigger connection request
        @self.web_app.get("/")
        async def start_client():

            # confirm server container is running
            print(f"Attempting to connect to server at {WebRTCResponder().webapp.web_url}")
            up, start, delay = False, time.time(), 10
            while not up:
                try:
                    with urllib.request.urlopen(WebRTCResponder().webapp.web_url) as response:
                        if response.getcode() == 200:
                            up = True
                except Exception:
                    if time.time() - start > test_timeout:
                        break
                    time.sleep(delay)

            assert up, f"Failed health check for server at {WebRTCResponder().webapp.web_url}"

            print(f"Successful health check for server at {WebRTCResponder().webapp.web_url}")

            # create data channel, in more complex use cases you might stream audio and/or video
            channel = self.pc.createDataChannel("data")

            # when the channel is opened, i.e. the P2P connection is established, send a ping to the server
            @channel.on("open")
            def on_open():
                channel.send("ping")

            # when a message is received from the server, check if it is a pong
            # if so, set the connection successful flag to true
            @channel.on("message")
            def on_message(message):

                if isinstance(message, str) and message.startswith("pong"):
                    self.connection_successful = True
                    print(f"Client received {message}")
                            

            # setup WebRTC connection using websockets
            ws_uri = WebRTCResponder().webapp.web_url.replace("http", "ws") + "/ws"
            print(f"Connecting to server websocket at {ws_uri}")
            async with websockets.connect(ws_uri) as websocket:

                # create offer
                offer = await self.pc.createOffer()
                # set local/our description, this also triggers ICE gathering
                await self.pc.setLocalDescription(offer)
                
                print(f"Sending offer to server...")
                # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
                # these are embedded in the SDP after setLocalDescription() is called
                await websocket.send(json.dumps({"sdp": self.pc.localDescription.sdp, "type": offer.type}))

                # receive answer
                answer = json.loads(await websocket.recv())
                print(f"Received answer from server...")
                # set remote peer description
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))


        
        
        return self.web_app
    



@app.local_entrypoint()
def main():
    import json
    import time
    import urllib

    print(f"Running health check of client container at {WebRTCRequester().webapp.web_url}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(WebRTCRequester().webapp.web_url) as response:
                if response.getcode() == 200:
                    print(f"Cloud client is up at {WebRTCRequester().webapp.web_url}")
                    up = True
                else:
                    print(f"Cloud client is not up at {WebRTCRequester().webapp.web_url}")
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for client at {WebRTCRequester().webapp.web_url}"

    print(f"Successful health check for client at {WebRTCRequester().webapp.web_url}")

    # build request to check connection status
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCRequester().webapp.web_url + "/success",
        method="GET",
        headers=headers,
    )

    # test if P2P data channel is established once a second
    print("Testing connection...")
    success = False
    now = time.time()
    while time.time() - now < test_timeout:
        with urllib.request.urlopen(req) as response:
            success = (json.loads(response.read().decode())["success"])
        try:
            assert success
            print("Connection successful!!!!")
            return
        except:
            print("Connection failed")
            time.sleep(1)
            pass

    # assert that the connection was successful
    assert success, "Connection failed"

    