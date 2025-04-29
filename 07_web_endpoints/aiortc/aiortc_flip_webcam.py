from pathlib import Path
import os
import time
import urllib
import json

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

class WebRTCPeer:

    @modal.enter()
    async def _initialize(self):
        """
        Initialize the peer connection
        and call custom init method for
        subclasses
        """

        from dataclasses import dataclass
        import asyncio
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from aiortc.sdp import candidate_from_sdp
        import uuid

        self.id = str(uuid.uuid4())
        self.web_app = FastAPI()
        self.pcs = {}

        # Add CORS middleware
        self.web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # HTTP NEGOTIATIONENDPOINTS
        @dataclass
        class IceCandidate:
            peer_id: str
            candidate_sdp: str
            sdpMid: str
            sdpMLineIndex: int
            usernameFragment: str
        
        @self.web_app.post("/ice_candidate")
        async def ice_candidate(candidate: IceCandidate):

            if not candidate:
                return 
            
            peer_id = candidate.peer_id
            if not self.pcs.get(peer_id):
                await asyncio.sleep(0.5)
            
            # if self.pcs[peer_id].connectionState == "connected":
            #     return
            
            ice_candidate = candidate_from_sdp(candidate.candidate_sdp)
            ice_candidate.sdpMid = candidate.sdpMid
            ice_candidate.sdpMLineIndex = candidate.sdpMLineIndex
            
            await self.handle_ice_candidate(peer_id, ice_candidate)

        @self.web_app.get("/offer")
        async def offer(peer_id: str, sdp: str, type: str):
            
            self.record_stream = False

            if type != "offer":
                return {"error": "Invalid offer type"}
            await self.handle_offer(peer_id, {"sdp": sdp, "type": type})
            return self.generate_answer(peer_id)

        @self.web_app.post("/run_stream")
        async def run_stream(peer_id: str):
            await self.run_streams(peer_id)
        
        # handling signaling through websocket
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws_negotiation(websocket: WebSocket, peer_id: str):

            # accept websocket connection
            await websocket.accept()

            # handle websocket messages and loop for lifetime
            while True:
                try:

                    if self.pcs.get(peer_id):

                        await asyncio.sleep(0.1)
                        
                        if self.pcs[peer_id].connectionState == "connected":
                            await websocket.close()
                            print("Websocket connection closed")
                            break

                    # get websocket message and parse as json
                    msg = json.loads(await websocket.receive_text())

                    # handle offer
                    if msg.get("type") == "offer":
                        
                        print("Server received offer...")

                        # handle offer
                        await self.handle_offer(peer_id, msg)


                        # await self.recorder.start()
                        # send answer
                        await websocket.send_text(json.dumps(self.generate_answer(peer_id)))

                        print("Server sent answer...")

                    elif msg.get("type") == "ice_candidate":
                        candidate = msg.get("candidate")
                        if not candidate or not self.pcs.get(peer_id):
                            return 
                        if self.pcs[peer_id].connectionState == "connected":
                            return
                        
                        ice_candidate = candidate_from_sdp(candidate["candidate_sdp"])
                        ice_candidate.sdpMid = candidate["sdpMid"]
                        ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]
                        
                        await self.handle_ice_candidate(peer_id, ice_candidate)

                    elif msg.get("type") == "identify":
                        await websocket.send_text(json.dumps({"type": "identify", "peer_id": self.id}))
                    else:
                        print(f"Unknown message type: {msg.get('type')}")

                except Exception as e:
                    if isinstance(e, WebSocketDisconnect):
                        print("Websocket connection closed")
                        break
                    else:
                        print(f"Error: {e}")
                        await websocket.close()
                        break

            await self._run_streams(peer_id)

        # call custom init logic
        await self.initialize()

    async def initialize(self):
        """
        Any custom logic when instantiating the peer
        """
        pass

    async def _setup_peer(self, peer_id):

        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.pcs[peer_id] = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        pc = RTCPeerConnection(configuration = config)
        print(f"Created peer connection for {peer_id}")

        self.pcs[peer_id] = pc

        await self.setup_streams(peer_id)

    async def setup_streams(self, peer_id):
        """
        Any custom logic when setting up the connection and streams
        """
        pass

    @modal.exit()
    async def _exit(self):

        import asyncio

        await self.exit()

        if self.pcs:
            print("Closing peer connections...")
            await asyncio.gather(*[pc.close() for pc in self.pcs.values()])
            self.pcs = {}

    async def exit(self):
        """
        Any custom logic when shutting down container
        """
        pass

    async def generate_offer(self, peer_id):

        await self._setup_peer(peer_id)
        
        # create initial offer
        offer = await self.pcs[peer_id].createOffer()
        
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pcs[peer_id].setLocalDescription(offer)

        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pcs[peer_id].localDescription.sdp, "type": offer.type, "peer_id": self.id}

    async def handle_offer(self, peer_id, offer):

        from aiortc import RTCSessionDescription

        await self._setup_peer(peer_id)

        # set remote description
        await self.pcs[peer_id].setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
            
        # create answer
        answer = await self.pcs[peer_id].createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.pcs[peer_id].setLocalDescription(answer)


    def generate_answer(self, peer_id):

        # send local description SDP    
        # NOTE: we can't use `answer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pcs[peer_id].localDescription.sdp, "type": "answer", "peer_id": self.id}
    
    async def handle_answer(self, peer_id, answer):

        from aiortc import RTCSessionDescription

        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(RTCSessionDescription(sdp = answer["sdp"], type = answer["type"]))

    async def handle_ice_candidate(self, peer_id, candidate):

        await self.pcs[peer_id].addIceCandidate(candidate)

    async def _run_streams(self, peer_id):

        import asyncio 

        await self.run_streams(peer_id)

        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(1.0)

    async def run_streams(self, peer_id):
        pass


# this class responds to an offer
# to establish a P2P connection,
# flips the video stream, and then
# streams the flipped video back to the provider
# and/or records the flipped video to a file
@app.cls(
    image=web_image,
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoProcessor(WebRTCPeer):   

    async def initialize(self):

        from fastapi.staticfiles import StaticFiles

        self.web_app.mount(
            "/static",
            StaticFiles(directory="/frontend"),
            name="static",
        )

    async def setup_streams(self, peer_id):

        import cv2

        from aiortc import MediaStreamTrack
        from aiortc.contrib.media import VideoFrame, MediaRelay, MediaRecorder

        class VideoFlipTrack(MediaStreamTrack):

            kind = "video"

            def __init__(self, track):
                super().__init__()
                self.track = track

            # this is the essential method we need to implement a custom stream
            async def recv(self):

                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 0)

                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                return new_frame
            

        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pcs[peer_id]:
                print(f"Video Processor connection state is {self.pcs[peer_id].connectionState}")
        
        
        @self.pcs[peer_id].on("track")
        def on_track(track):

            from aiortc import RTCPeerConnection
            
            print(f"Video Processor received {track.kind} track: {track}")
            
            # create processed track
            flipped_track = VideoFlipTrack(track)
            self.pcs[peer_id].addTrack(flipped_track)

            @track.on("ended")
            async def on_ended():
                print("Incoming video track ended")

    

    # add frontend and websocket endpoint for testing
    @modal.asgi_app(label="webrtc-video-flipper")
    def web_endpoints(self):

        import asyncio
        import json
        from fastapi import WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from aiortc.sdp import candidate_from_sdp

        # serve the frontend
        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)
        
        return self.web_app  

@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoFlipTester(WebRTCPeer):

    TEST_VIDEO_SOURCE_FILE = "/media/cliff_jumping.mp4"
    TEST_VIDEO_RECORD_FILE = OUTPUT_VOLUME_PATH / "flipped_test_video.mp4"
    FRAME_DIFFERENCE_THRESHOLD = 5
    VIDEO_DURATION_BUFFER_SECS = 5.0

    async def initialize(self):

        import cv2

        self.input_filepath = self.TEST_VIDEO_SOURCE_FILE
        # get input video duration in seconds
        self.input_video = cv2.VideoCapture(self.input_filepath)
        self.input_video_duration = self.input_video.get(cv2.CAP_PROP_FRAME_COUNT) / self.input_video.get(cv2.CAP_PROP_FPS)
        self.input_video.release()
        self.stream_duration = self.input_video_duration + self.VIDEO_DURATION_BUFFER_SECS
        print(f"Stream duration: {self.stream_duration} seconds")

        self.player = None

        self.output_filepath = self.TEST_VIDEO_RECORD_FILE
        self.recorder = None

    async def setup_streams(self, peer_id):

        from aiortc.contrib.media import MediaPlayer, MediaRecorder
        
        # setup video source
        self.video_src = MediaPlayer(self.input_filepath)
        self.pcs[peer_id].addTrack(self.video_src.video)

        # setup video recorder
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)
        self.recorder = MediaRecorder(self.output_filepath)

        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Stream Tester side connection state updated: {self.pcs[peer_id].connectionState}")

        @self.pcs[peer_id].on("track")
        def on_track(track):
            
            print(f"Test peer received {track.kind} track: {track}")
            # record track to file
            self.recorder.addTrack(track)
            
            @track.on("ended")
            async def on_ended():
                print("Returned processed video track ended")
                await self.recorder.stop()
                self.recorder = None
                self.video_src = None
                
    async def run_streams(self, peer_id):

        import asyncio

        await self.recorder.start()
        await asyncio.sleep(self.stream_duration)
        await self.pcs[peer_id].close()

    def confirm_recording(self):

        import cv2

        # compare output video length to input video length
        input_video = cv2.VideoCapture(self.input_filepath)
        input_video_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video = cv2.VideoCapture(self.output_filepath)
        output_video_length = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video.release()
        output_video.release()

        if (input_video_length - output_video_length) < self.FRAME_DIFFERENCE_THRESHOLD:
            return True
        else:
            return False
               

    async def start_webrtc_connection(self):

        import asyncio
        import json
        import websockets

        peer_id = None
        # setup WebRTC connection using websockets
        ws_uri = WebRTCVideoProcessor().web_endpoints.web_url.replace("http", "ws") + f"/ws/{self.id}"

        print(f"Connecting to video flipper websocket at {ws_uri}")
        async with websockets.connect(ws_uri) as websocket:

            await websocket.send(json.dumps({"type": "identify"}))
            peer_id = json.loads(await websocket.recv())["peer_id"]

            offer_msg = await self.generate_offer(peer_id)

            print(f"Sending offer...")
            await websocket.send(json.dumps(offer_msg))

            try: 
                # receive answer
                answer = json.loads(await websocket.recv())

                if answer.get("type") == "answer":
                    print(f"Received answer from responder...")
                    await self.handle_answer(peer_id, answer)

            except websockets.exceptions.ConnectionClosed as e:
                print("Connection closed")
                await websocket.close()

        # loop until video player is finished
        if peer_id:
            await self.run_streams(peer_id)




    @modal.asgi_app(label="webrtc-video-provider")
    def web_endpoints(self):
        
        import asyncio
        
        @self.web_app.get("/run_test")
        async def run_test():

            # self.video_src = MediaPlayer(self.input_filepath)
            # start WebRTC connection test
            await self.start_webrtc_connection()
            return True
            

        @self.web_app.get("/check_test")
        async def check_test():
            return self.confirm_recording()
            

        return self.web_app
    
# set timeout for health checks and connection test
MINUTES = 60  # seconds
TEST_TIMEOUT = 2.0 * MINUTES  

def trigger_webrtc_test():

    print(f"Attempting to trigger WebRTC connector at {WebRTCVideoFlipTester().web_endpoints.web_url + '/run_test'}")
    test_triggered, start, delay = False, time.time(), 10
    while not test_triggered:
        try:
            with urllib.request.urlopen(WebRTCVideoFlipTester().web_endpoints.web_url + "/run_test") as response:
                if response.getcode() == 200:
                    test_triggered = response.read().decode()
        except Exception as e:
            print(f"Error: {e}")
            if time.time() - start > TEST_TIMEOUT:
                break
            time.sleep(delay)
    
    return test_triggered
    

def check_test_successful():

    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCVideoFlipTester().web_endpoints.web_url + "/check_test",
        method="GET",
        headers=headers,
    )
    test_successful, start, delay = False, time.time(), 10
    while not test_successful:
        try:
            with urllib.request.urlopen(req) as response:
                test_successful = json.loads(response.read().decode())
                return test_successful
        except Exception as e:
            print(f"Error: {e}")
            if time.time() - start > TEST_TIMEOUT:
                break
            time.sleep(delay)

    return test_successful


@app.local_entrypoint()
def main():

    assert trigger_webrtc_test(), "Test failed to trigger"
    assert check_test_successful(), "Test faileda to complete"



    