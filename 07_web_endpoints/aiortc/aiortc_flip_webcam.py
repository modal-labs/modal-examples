import abc
from pathlib import Path
import os
from dataclasses import dataclass
import modal

this_directory = Path(__file__).parent.resolve()

# image
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
    .add_local_dir(
        os.path.join(this_directory, "media"), 
        remote_path="/media"
    )
    .add_local_dir(
        os.path.join(this_directory, "frontend"), 
        remote_path="/frontend"
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

TEST_VIDEO_SOURCE_FILE = "/media/cliff_jumping.mp4"
TEST_VIDEO_RECORD_FILE = OUTPUT_VOLUME_PATH / "flipped_test_video.mp4"

class WebRTCPeer(abc.ABC):

    @modal.enter()
    async def _setup_peer(self):
        """
        Initialize the peer connection
        and call custom init method for
        subclasses
        """
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware

        self.web_app = FastAPI()
        self.pcs = set()
        self.active_pc = None

        # Add CORS middleware
        self.web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # call custom init logic
        await self.setup_peer()

    async def setup_peer(self):
        """
        Any custom logic to setup the peer connection
        """
        pass

    async def _setup_streams(self):

        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.active_pc = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.active_pc = RTCPeerConnection(configuration = config)
        self.pcs.add(self.active_pc)

        await self.setup_streams()

    async def setup_streams(self):
        pass

    @modal.exit()
    async def _exit(self):
        await self.exit()
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()

    async def exit(self):
        pass

    async def generate_offer(self):

        await self._setup_streams()
        
        # create initial offer
        offer = await self.active_pc.createOffer()
        
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.active_pc.setLocalDescription(offer)

        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.active_pc.localDescription.sdp, "type": offer.type}

    async def handle_offer(self, offer):

        from aiortc import RTCSessionDescription

        await self._setup_streams()

        # set remote description
        await self.active_pc.setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
        # create answer
        answer = await self.active_pc.createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.active_pc.setLocalDescription(answer)

    
    def generate_answer(self):

        # send local description SDP    
        # NOTE: we can't use `answer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.active_pc.localDescription.sdp, "type": "answer"}
    
    async def handle_answer(self, answer):

        from aiortc import RTCSessionDescription

        # set remote peer description
        await self.active_pc.setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))

    async def handle_ice_candidate(self, candidate):

        await self.active_pc.addIceCandidate(candidate)


# this class responds to an offer
# to establish a P2P connection
# and flips the video stream
@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoProcessor(WebRTCPeer):   

    async def setup_peer(self):

        from fastapi.staticfiles import StaticFiles

        self.output_filepath = TEST_VIDEO_RECORD_FILE
        self.record_stream = False
        self.recorder = None
        self.relay = None

        self.web_app.mount(
            "/static",
            StaticFiles(directory="/frontend"),
            name="static",
        )

    async def setup_streams(self):

        import cv2

        from aiortc import MediaStreamTrack
        from aiortc.contrib.media import VideoFrame, MediaRelay, MediaRecorder

        class VideoFlipTrack(MediaStreamTrack):

            kind = "video"

            def __init__(self, track):
                super().__init__()
                self.track = track
                self.frame_count = 0

            async def recv(self):

                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 0)

                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                if self.frame_count == 0:
                    self.frame_count = frame.pts
                new_frame.pts = self.frame_count
                new_frame.time_base = frame.time_base

                self.frame_count += 3330

                return new_frame
            
        if self.record_stream:
            print(f"Initializing recorder at {self.output_filepath}")
            if os.path.exists(self.output_filepath):
                os.remove(self.output_filepath)
            self.recorder = MediaRecorder(self.output_filepath)
            self.relay = MediaRelay()

        @self.active_pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if self.active_pc:
                print(f"Responder side connection state is {self.active_pc.connectionState}")
                # if self.active_pc.connectionState == "closed":
                #     if self.recorder:
                #         await self.recorder.stop()
                #         self.recorder = None
                #     self.active_pc = None
                #     self.relay = None
        
        
        @self.active_pc.on("track")
        def on_track(track):
            
            print(f"Responder received {track.kind} track: {track}")

            flipped_track = VideoFlipTrack(track)
            
            if self.record_stream:
                # relay the flipped track to the recorder and back to the provider
                self.relay = MediaRelay()
                self.recorder.addTrack(self.relay.subscribe(flipped_track))
                self.active_pc.addTrack(self.relay.subscribe(flipped_track))
            else:
                self.relay = None
                self.active_pc.addTrack(flipped_track)

            @track.on("ended")
            async def on_ended():
                print("VideoFlipper:Incoming Track ended")
                if self.record_stream and self.recorder:
                    await self.recorder.stop()
                    self.recorder = None

    # add responder logic to webapp
    @modal.asgi_app(label="webrtc-video-flipper")
    def web_endpoints(self):

        import json
        from fastapi import WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from aiortc import RTCIceCandidate
        from aiortc.sdp import candidate_from_sdp

        @dataclass
        class IceCandidate:
            candidate: str
            sdpMid: str
            sdpMLineIndex: int
            usernameFragment: str

        # create root endpoint (useful for testing container is running)
        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)
        
        @self.web_app.get("/config")
        async def get_config():
            return {
                "videoProcessorUrl": str(self.web_endpoints.web_url)
            }
        
        @self.web_app.post("/ice_candidate")
        async def process_ice_candidate(candidate: IceCandidate):
            ice_candidate = candidate_from_sdp(candidate.candidate)
            ice_candidate.sdpMid = candidate.sdpMid
            ice_candidate.sdpMLineIndex = candidate.sdpMLineIndex
            
            await self.handle_ice_candidate(ice_candidate)

        @self.web_app.get("/offer")
        async def process_webcam_peer_offer(sdp: str, type: str):
            
            self.record_stream = False

            if type != "offer":
                return {"error": "Invalid offer type"}
            await self.handle_offer({"sdp": sdp, "type": type})
            return self.generate_answer()

        # create websocket endpoint to handle incoming connections
        @self.web_app.websocket("/ws")
        async def test_video_streaming(websocket: WebSocket):

            self.record_stream = True

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


                        await self.recorder.start()
                        # send answer
                        await websocket.send_text(json.dumps(self.generate_answer()))

                        print("Server sent answer...")

                    else:
                        print(f"Unknown message type: {msg.get('type')}")

                except WebSocketDisconnect as e:
                    await websocket.close()
                    break

                
        return self.web_app
    

@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoFlipTester(WebRTCPeer):


    async def setup_peer(self):
        

        self.input_filepath = TEST_VIDEO_SOURCE_FILE
        self.video_src = None
        self.test_task = None


    async def setup_streams(self):

        if self.video_src:
            @self.active_pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"Provder side connection state updated: {self.active_pc.connectionState}")

            self.active_pc.addTrack(self.video_src.video)

    async def start_webrtc_connection(self):

        import json
        import websockets

        # setup WebRTC connection using websockets
        ws_uri = self.video_processor_url.replace("http", "ws") + f"/ws"
        print(f"Connecting to video flipper websocket at {ws_uri}")
        async with websockets.connect(ws_uri) as websocket:

            print(f"Generating provider offer...")
            offer_msg = await self.generate_offer()

            print(f"Sending offer...")
            await websocket.send(json.dumps(offer_msg))

            try: 
                # receive answer
                answer = json.loads(await websocket.recv())

                if answer.get("type") == "answer":
                    print(f"Received answer from responder...")
                    await self.handle_answer(answer)

            except websockets.exceptions.ConnectionClosed as e:
                print("Connection closed")
                await websocket.close()



    @modal.asgi_app(label="webrtc-video-provider")
    def web_endpoints(self):
        
        import asyncio
        from aiortc.contrib.media import MediaPlayer
        from fastapi.responses import HTMLResponse

        @self.web_app.get("/config")
        async def get_config():
            return {
                "videoProcessorUrl": str(self.video_processor_url)
            }

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)

        
        @self.web_app.get("/run_test")
        async def run_test():

            self.video_src = MediaPlayer(self.input_filepath)
            # start WebRTC connection test
            self.test_task = asyncio.create_task(self.start_webrtc_connection())

            # loop until video player is finished
            while self.video_src.video.readyState != "ended":

                await asyncio.sleep(1.0)

            self.video_src = None
            

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

    print(f"Attempting to trigger WebRTC connector at {WebRTCVideoFlipTester().web_endpoints.web_url + '/run_test'}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(WebRTCVideoFlipTester().web_endpoints.web_url + "/run_test") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    # build request to check connection status
    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCVideoFlipTester().web_endpoints.web_url + "/test_complete",
        method="GET",
        headers=headers,
    )
    # test if P2P test is complete once a second
    now = time.time()
    while time.time() - now < test_timeout:
        with urllib.request.urlopen(req) as response:
            return_data = json.loads(response.read().decode())
            print(f"Response: {return_data}")
        time.sleep(1)
        try:
            test_complete = return_data["test_complete"]
            assert test_complete
            print("Test complete!!!!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            pass

    # assert that the connection was successful
    assert test_complete, "Test failed to complete"


    