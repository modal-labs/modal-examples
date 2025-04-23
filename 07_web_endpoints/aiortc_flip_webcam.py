import abc
import modal
from pathlib import Path
import os

this_directory = Path(__file__).parent.resolve()

# pretty minimal image
web_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc",
    )
    .add_local_dir(os.path.join(this_directory, "media"), remote_path="/media")
)

# instantiate our app
app = modal.App(
    "aiortc-video-processing"
)

# set timeout for health checks and connection test
MINUTES = 60  # seconds
test_timeout = 0.5 * MINUTES

class WebRTCPeer(abc.ABC):

    @modal.enter()
    def _init(self):

        from aiortc import RTCPeerConnection

        # create peer connection with STUN server

        # aiortc automatically uses google's STUN server, but we can also specify our own
        # config = RTCConfiguration()
        # config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        # self.pc = RTCPeerConnection(configuration = config)

        self.pc = RTCPeerConnection()

        self.init()

    def init(self):
        pass


    @modal.exit()
    async def exit(self):
        await self.pc.close()  

    def setup_streams(self):
        pass

    async def generate_offer(self):

        self.setup_streams()
        
        # create initial offer
        offer = await self.pc.createOffer()
        
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pc.setLocalDescription(offer)

        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pc.localDescription.sdp, "type": offer.type}

    async def handle_offer(self, data):

        import json
        from aiortc import RTCSessionDescription

        self.setup_streams()

        # set remote description
        await self.pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))

        # create answer
        answer = await self.pc.createAnswer()

        # set local/our description, this also triggers ICE gathering
        await self.pc.setLocalDescription(answer)

        # send local description SDP 
        # NOTE: we can't use `answer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return json.dumps({"sdp": self.pc.localDescription.sdp, "type": "answer"}) 
    
    async def handle_answer(self, answer):

        from aiortc import RTCSessionDescription

        # set remote peer description
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))


# a class for our server. in this case server just means that this process is responding to an offer
# to establish a P2P connection as opposed to initiating the connection
@app.cls(
    image=web_image,
)
@modal.concurrent(max_inputs=100)
class WebRTCTestResponder(WebRTCPeer):   

    def setup_streams(self):


        from aiortc import VideoStreamTrack

        class WebcamFlipTrack(VideoStreamTrack):

            def __init__(self, track):
                super().__init__()
                self.track = track

            async def recv(self):

                import numpy as np

                frame = await self.track.recv()
                # frame = np.flip(frame, axis=1)
                return frame


        @self.pc.on("track")
        def on_track(track):
            
            self.pc.addTrack(WebcamFlipTrack(track))


    def init(self):

        from fastapi import FastAPI

        self.connection_successful = False
        self.web_app = FastAPI()

        @self.web_app.get("/success")
        async def success():
            return {"success": self.connection_successful}

    # add responder logic to webapp
    @modal.asgi_app(label="webrtc-server")
    def webapp(self):

        import json
        from fastapi import FastAPI, WebSocket


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
                        print(f"Received offer: {msg}")

                        # handle offer and generate answer
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
    image=web_image
)
@modal.concurrent(max_inputs=100)
class WebRTCWebcamProvider(WebRTCPeer):

    def init(self):

        from fastapi import FastAPI
        
        self.connection_successful = False
        
        self.web_app = FastAPI()

        @self.web_app.get("/success")
        async def success():
            return {"success": self.connection_successful}

    def setup_streams(self):

        import platform
        from aiortc.contrib.media import MediaPlayer, MediaRelay

        @self.pc.on("track")
        def on_track(track):
            print(f"Received track: {track}")
        
        # options = {"framerate": "30", "video_size": "640x480"}
        
        # if platform.system() == "Darwin":
        #     webcam = MediaPlayer(
        #         "default:none", format="avfoundation", options=options
        #     )
        # elif platform.system() == "Windows":
        #     webcam = MediaPlayer(
        #         "video=Integrated Camera", format="dshow", options=options
        #     )
        # else:
        #     webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)

        source_video = "/media/cliff_jumping.mp4"
        player = MediaPlayer(source_video)

        # print(f"Player track kind: {player.kind}")
        print(f"Player track kind: {player.video.kind}")
        # relay = MediaRelay()
        # relay.subscribe(webcam.video)

        self.pc.addTrack(player.video)

        
    
    
    def check_responder_is_up(self):

        import time
        import urllib

        print(f"Attempting to connect to server at {WebRTCTestResponder().webapp.web_url}")
        up, start, delay = False, time.time(), 10
        while not up:
            try:
                with urllib.request.urlopen(WebRTCTestResponder().webapp.web_url) as response:
                    if response.getcode() == 200:
                        up = True
            except Exception:
                if time.time() - start > test_timeout:
                    break
                time.sleep(delay)

        assert up, f"Failed to connect to server at {WebRTCTestResponder().webapp.web_url}"

        print(f"Server is up at {WebRTCTestResponder().webapp.web_url}")

    # def stream_video(self):



    # add initiator logic to webapp
    @modal.asgi_app(label="webrtc-client")
    def webapp(self):

        import json
        import websockets

        # create root endpoint to trigger connection request
        @self.web_app.get("/")
        async def setup_connection():

            # confirm server container is running
            self.check_responder_is_up()
            
            # setup WebRTC connection using websockets
            ws_uri = WebRTCTestResponder().webapp.web_url.replace("http", "ws") + "/ws"
            print(f"Connecting to server websocket at {ws_uri}")
            async with websockets.connect(ws_uri) as websocket:

                offer_msg = await self.generate_offer()

                print(f"Sending offer to server...")
                
                await websocket.send(json.dumps(offer_msg))

                # receive answer
                answer = json.loads(await websocket.recv())
                print(f"Received answer from server...")

                await self.handle_answer(answer)
            
        
        return self.web_app
    

@app.local_entrypoint()
def main():

    import json
    import time
    import urllib

    print(f"Attempting to trigger WebRTC connector at {WebRTCWebcamProvider().webapp.web_url}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(WebRTCWebcamProvider().webapp.web_url) as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed to trigger WebRTC connector at {WebRTCWebcamProvider().webapp.web_url}"

    print(f"Successfully triggered WebRTC connector at {WebRTCWebcamProvider().webapp.web_url}")

    # build request to check connection status
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCWebcamProvider().webapp.web_url + "/success",
        method="GET",
        headers=headers,
    )

    # test if P2P data channel is established once a second
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
            time.sleep(1)
            pass

    # assert that the connection was successful
    assert success, "Connection failed"

    