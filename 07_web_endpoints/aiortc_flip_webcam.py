import abc
from pathlib import Path
import os

import modal

this_directory = Path(__file__).parent.resolve()

# pretty minimal image
web_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .run_commands("pip install --upgrade pip")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc",
        "opencv-python",
    )
    .add_local_file(
        os.path.join(this_directory, "media", "cliff_jumping.mp4"), 
        remote_path="/media/cliff_jumping.mp4"
    )
)

# create an output volume to store the transmitted videos
output_volume = modal.Volume.from_name("aiortc-video-processing", create_if_missing=True)
OUTPUT_VOLUME_PATH = Path("/output")

# instantiate our app
app = modal.App(
    "aiortc-video-processing"
)

# set timeout for health checks and connection test
MINUTES = 60  # seconds
test_timeout = 2.0 * MINUTES

class WebRTCPeer(abc.ABC):

    @modal.enter()
    async def _setup_peer(self):
        """
        Initialize the peer connection
        and call custom init method for
        subclasses
        """
        from fastapi import FastAPI
        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.pc = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pc = RTCPeerConnection(configuration = config)

        await self.init()

        self.web_app = FastAPI()

        @self.web_app.get("/connection_state")
        async def get_connection_state():
            return {"connection_state": self.connection_state}

    async def init(self):
        pass

    async def setup_streams(self):
        pass

    @modal.exit()
    async def _exit(self):
        await self.exit()
        await self.pc.close()

    async def exit(self):
        pass

    @property
    def connection_state(self):
        return self.pc.connectionState

    async def generate_offer(self):

        await self.setup_streams()
        
        # create initial offer
        offer = await self.pc.createOffer()
        
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pc.setLocalDescription(offer)

        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pc.localDescription.sdp, "type": offer.type}

    async def handle_offer(self, data):

        from aiortc import RTCSessionDescription

        await self.setup_streams()

        # set remote description
        await self.pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))
        # create answer
        answer = await self.pc.createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.pc.setLocalDescription(answer)

    
    def generate_answer(self):

        import json

        # send local description SDP 
        # NOTE: we can't use `answer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return json.dumps({"sdp": self.pc.localDescription.sdp, "type": "answer"})  
    
    async def handle_answer(self, answer):

        from aiortc import RTCSessionDescription

        # set remote peer description
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))


# this class responds to an offer
# to establish a P2P connection as opposed to initiating the connection
@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoFlipper(WebRTCPeer):   

    async def init(self):

        from aiortc.contrib.media import MediaRecorder

        if os.path.exists(OUTPUT_VOLUME_PATH / "flipped_vid.mp4"):
            os.remove(OUTPUT_VOLUME_PATH / "flipped_vid.mp4")

        print(f"Initializing recorder at {OUTPUT_VOLUME_PATH / 'flipped_vid.mp4'}")
        self.recorder = MediaRecorder(OUTPUT_VOLUME_PATH / "flipped_vid.mp4")

    async def setup_streams(self):

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Responder side connection state is %s" % self.pc.connectionState)
        
        
        @self.pc.on("track")
        async def on_track(track):
            print(f"Responder received {track.kind} track: {track}")
            # self.pc.addTrack(VideoFlipTrack(self.relay.subscribe(track)))
            self.recorder.addTrack(track)

            @track.on("ended")
            async def on_ended():
                print("Track %s ended" % track.kind)
                await self.recorder.stop()

    # add responder logic to webapp
    @modal.asgi_app(label="webrtc-server")
    def webapp(self):

        import json
        from fastapi import WebSocket, WebSocketDisconnect

        # create root endpoint (useful for testing container is running)
        @self.web_app.get("/")
        async def get_server():
            pass

        @self.web_app.get("/video_size")
        async def get_flipped_video_size():
            try:
                return {
                    "size_flipped": os.path.getsize(OUTPUT_VOLUME_PATH / "flipped_vid.mp4")
                }
            except Exception as e:
                return {
                    "error": str(e)
                }

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
                        
                        print("Server received offer...")

                        # handle offer
                        await self.handle_offer(msg)
                        # start recording stream
                        await self.recorder.start()
                        # send answer
                        await websocket.send_text(self.generate_answer())

                        print("Server sent answer...")

                    else:
                        print(f"Unknown message type: {msg.get('type')}")

                except Exception as e:
                    if isinstance(e, WebSocketDisconnect):
                        await websocket.close()
                        break
                    else:
                        raise e
                
        return self.web_app
    

@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoProvider(WebRTCPeer):

    LOCAL_TEST_VIDEO_SOURCE = "/media/cliff_jumping.mp4"

    async def init(self):

        import shutil
        from aiortc.contrib.media import MediaPlayer

        # copy that video to the output volume
        shutil.copy(self.LOCAL_TEST_VIDEO_SOURCE, OUTPUT_VOLUME_PATH / "cliff_jumping.mp4")
        self.player = MediaPlayer(self.LOCAL_TEST_VIDEO_SOURCE)

        self.test_task = None

    async def setup_streams(self):

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Provder side connection state updated: {self.pc.connectionState}")

        self.pc.addTrack(self.player.video)

    def check_responder_is_up(self):

        import time
        import urllib

        print(f"Attempting to connect to video flipper at {WebRTCVideoFlipper().webapp.web_url}")
        up, start, delay = False, time.time(), 10
        while not up:
            try:
                with urllib.request.urlopen(WebRTCVideoFlipper().webapp.web_url) as response:
                    if response.getcode() == 200:
                        up = True
            except Exception:
                if time.time() - start > test_timeout:
                    break
                time.sleep(delay)

        assert up, f"Failed to connect to video flipper at {WebRTCVideoFlipper().webapp.web_url}"

        print(f"Video flipper is up at {WebRTCVideoFlipper().webapp.web_url}")

    async def start_webrtc_connection(self):

        import json
        import websockets
        import asyncio

        # setup WebRTC connection using websockets
        ws_uri = WebRTCVideoFlipper().webapp.web_url.replace("http", "ws") + "/ws"
        print(f"Connecting to server websocket at {ws_uri}")
        async with websockets.connect(ws_uri) as websocket:

            print(f"Generating provider offer...")
            offer_msg = await self.generate_offer()

            print(f"Sending offer to responder...")
            await websocket.send(json.dumps(offer_msg))

            try: 
                # receive answer
                answer = json.loads(await websocket.recv())

                if answer.get("type") == "answer":
                    print(f"Received answer from responder...")
                    await self.handle_answer(answer)

                # loop until video player is finished
                while True:
                    
                    
                    if self.player.video.readyState == "ended":
                        print(f"Provider finished playing video, ending task...")
                        break

                    await asyncio.sleep(1.0)

            except Exception as e:
                if isinstance(e, websockets.exceptions.ConnectionClosed):
                    print("Connection closed")
                    await websocket.close()
                else:
                    raise e


    @modal.asgi_app(label="webrtc-client")
    def web_endpoints(self):
        
        import asyncio

        
        @self.web_app.get("/")
        async def root():
            # create root endpoint (useful for testing container is running)
            pass

        
        @self.web_app.get("/run_test")
        async def run_test():
            # create root endpoint to trigger connection request

            # confirm server container is running first
            self.check_responder_is_up()
            # start WebRTC connection test
            self.test_task = asyncio.create_task(self.start_webrtc_connection())

        @self.web_app.get("/test_complete")
        async def test_complete():

            if self.test_task is None:
                test_complete = False
            else:
                test_complete = self.test_task.done()

            return {
                "test_complete": test_complete
            }

        return self.web_app
    
        

@app.local_entrypoint()
def main():

    import json
    import time
    import urllib

    print(f"Attempting to trigger WebRTC connector at {WebRTCVideoProvider().web_endpoints.web_url + '/run_test'}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(WebRTCVideoProvider().web_endpoints.web_url + "/run_test") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed to trigger WebRTC connector at {WebRTCVideoProvider().web_endpoints.web_url + '/run_test'}"

    print(f"Successfully triggered WebRTC connector at {WebRTCVideoProvider().web_endpoints.web_url + '/run_test'}")

    # build request to check connection status
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCVideoFlipper().webapp.web_url + "/video_size",
        method="GET",
        headers=headers,
    )

    # test if P2P data channel is established once a second
    success = False
    now = time.time()
    while time.time() - now < test_timeout:
        with urllib.request.urlopen(req) as response:
            return_data = json.loads(response.read().decode())
            print(f"Response: {return_data}")
        time.sleep(1)
        try:
            success = return_data["size_flipped"] > 100
            assert success
            print("Connection successful!!!!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            pass

    # assert that the connection was successful
    assert success, "Connection failed"

    # wait for the test to complete
    test_complete = False
    # build request to check connection status
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCVideoProvider().web_endpoints.web_url + "/test_complete",
        method="GET",
        headers=headers,
    )
    while not test_complete:
        with urllib.request.urlopen(req) as response:
            return_data = json.loads(response.read().decode())
            print(f"Response: {return_data}")
            test_complete = return_data["test_complete"]
            time.sleep(1)

    