# standard python imports...
import asyncio
from pathlib import Path
import os
import time
import urllib
import json

# ...and modal
import modal

this_directory = Path(__file__).parent.resolve()

# image
web_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc",
        "opencv-python",
    )
    # video file for testing
    .add_local_dir(
        os.path.join(this_directory, "media"), 
        remote_path="/media"
    )
    # frontend files
    .add_local_dir(
        os.path.join(this_directory, "frontend"), 
        remote_path="/frontend"
    )
)

# instantiate our app
app = modal.App(
    "aiortc-video-processing-example"
)

class WebRTCPeer:
    """
    Base class for WebRTC peer connections using aiortc 
    that handles connection setup, negotiation, and stream management.

    This class provides the core WebRTC functionality including:
    - Peer connection initialization and cleanup
    - Signaling endpoints via HTTP and WebSocket
      - SDP offer/answer exchange
      - Trickle ICE candidate handling
    - Stream setup and management
    
    Subclasses can implement the following methods:
    - initialize(): Any custom initialization logic
    - setup_streams(): Logic for setting up media tracks and streams (this is where the main business logic goes)
    - run_streams(): Logic for starting streams (not always necessary)
    - exit(): Any custom cleanup logic
    """

    @modal.enter()
    async def _initialize(self):

        import asyncio
        import uuid
        from dataclasses import dataclass
        
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from aiortc.sdp import candidate_from_sdp

        self.id = str(uuid.uuid4())
        self.web_app = FastAPI()
        self.pcs = {}

        # HTTP NEGOTIATION ENDPOINTS
        @dataclass
        class IceCandidate:
            peer_id: str
            candidate_sdp: str
            sdpMid: str
            sdpMLineIndex: int
            usernameFragment: str
        
        # handle ice candidate (trickle ice)
        @self.web_app.post("/ice_candidate")
        async def ice_candidate(candidate: IceCandidate):

            if not candidate:
                return 
            
            print(f"Peer {self.id} received ice candidate from {candidate.peer_id}")
            
            peer_id = candidate.peer_id
            
            ice_candidate = candidate_from_sdp(candidate.candidate_sdp)
            ice_candidate.sdpMid = candidate.sdpMid
            ice_candidate.sdpMLineIndex = candidate.sdpMLineIndex
            
            await self.handle_ice_candidate(peer_id, ice_candidate)

        @self.web_app.get("/offer")
        async def offer(peer_id: str, sdp: str, type: str):
            
            if type != "offer":
                return {"error": "Invalid offer type"}
            await self.handle_offer(peer_id, {"sdp": sdp, "type": type})
            return self.generate_answer(peer_id)

        # run until finished
        @self.web_app.post("/run_stream")
        async def run_stream(peer_id: str):
            await self._run_streams(peer_id)
        
        # handling signaling through websocket
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws_negotiation(websocket: WebSocket, peer_id: str):

            # accept websocket connection
            await websocket.accept()

            # handle websocket messages and loop for lifetime
            while True:
                
                try:
                    # get websocket message and parse as json
                    msg = json.loads(await websocket.receive_text())

                    # handle offer
                    if msg.get("type") == "offer":
                        
                        print(f"Peer {self.id} received offer from {peer_id}...")

                        await self.handle_offer(peer_id, msg)
                        # generate and send answer
                        await websocket.send_text(
                            json.dumps(self.generate_answer(peer_id))
                        )

                    # handle ice candidate (trickle ice)
                    elif msg.get("type") == "ice_candidate":

                        candidate = msg.get("candidate")
                        
                        if not candidate or not self.pcs.get(peer_id):
                            return 
                        
                        print(f"Peer {self.id} received ice candidate from {peer_id}...")
                        
                        # parse ice candidate
                        ice_candidate = candidate_from_sdp(candidate["candidate_sdp"])
                        ice_candidate.sdpMid = candidate["sdpMid"]
                        ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]
                        
                        await self.handle_ice_candidate(peer_id, ice_candidate)

                        # wait and break if connected
                        # this ensures that we close websocket asap (could remove)
                        await asyncio.sleep(0.2) 
                        if self.pcs[peer_id].connectionState == "connected":
                            break
                    
                    # get peer's id
                    elif msg.get("type") == "identify":

                        await websocket.send_text(json.dumps({"type": "identify", "peer_id": self.id}))
                    
                    else:
                        print(f"Unknown message type: {msg.get('type')}")

                except Exception as e:
                    if isinstance(e, WebSocketDisconnect):
                        break
                    else:
                        print(f"Error: {e}")
            
            await websocket.close()
            print("Websocket connection closed")

            # run until complete
            await self._run_streams(peer_id)

        # call custom init logic
        await self.initialize()

    async def initialize(self):
        """
        Any custom logic when instantiating the peer
        """
        pass

    async def _setup_peer_connection(self, peer_id):

        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.pcs[peer_id] = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pcs[peer_id] = RTCPeerConnection(configuration = config)

        await self.setup_streams(peer_id)

        print(f"Created peer connection and setup streams from {self.id} to {peer_id}")

    async def setup_streams(self, peer_id):
        """
        Any custom logic when setting up the connection and streams
        """
        pass

    async def generate_offer(self, peer_id):

        print(f"Peer {self.id} generating offer for {peer_id}...")

        # initalize peer connection
        await self._setup_peer_connection(peer_id)
        # create initial offer
        offer = await self.pcs[peer_id].createOffer()
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pcs[peer_id].setLocalDescription(offer)
        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pcs[peer_id].localDescription.sdp, "type": offer.type, "peer_id": self.id}

    async def handle_offer(self, peer_id, offer):

        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling offer from {peer_id}...")

        # initalize peer connection and streams
        await self._setup_peer_connection(peer_id)
        # set remote description
        await self.pcs[peer_id].setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
        # create answer
        answer = await self.pcs[peer_id].createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.pcs[peer_id].setLocalDescription(answer)

    def generate_answer(self, peer_id):

        print(f"Peer {self.id} generating answer for {peer_id}...")

        return {
            "sdp": self.pcs[peer_id].localDescription.sdp, 
            "type": "answer", 
            "peer_id": self.id
        }
    
    async def handle_answer(self, peer_id, answer):

        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling answer from {peer_id}...")
        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(
                sdp = answer["sdp"], 
                type = answer["type"]
            )
        )

    async def handle_ice_candidate(self, peer_id, candidate):

        import asyncio

        print(f"Peer {self.id} handling ice candidate from {peer_id}...")

        # sometimes this event is called before 
        # the peer connection is created on this end
        retries = 5
        while not self.pcs.get(peer_id) and retries > 0:
            await asyncio.sleep(0.1)
            retries -= 1

        if not retries:
            print(f"Peer {self.id} failed to create peer connection for {peer_id} before ICE candidate event")
            return

        await self.pcs[peer_id].addIceCandidate(candidate)

    async def _run_streams(self, peer_id):

        import asyncio 

        print(f"Peer {self.id} running streams for {peer_id}...")

        # trigger custom streams if necessary
        await self.run_streams(peer_id)

        # run until connection is closed
        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(1.0)

    # custom logic for running streams
    async def run_streams(self, peer_id):
        pass

    @modal.exit()
    async def _exit(self):

        import asyncio

        print(f"Shutting down peer: {self.id}...")
        # call custom exit logic
        await self.exit()

        # close peer connections
        if self.pcs:
            print(f"Closing peer connections for peer {self.id}...")
            await asyncio.gather(*[pc.close() for pc in self.pcs.values()])
            self.pcs = {}

    async def exit(self):
        """
        Any custom logic when shutting down container
        """
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

        # frontend files
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
            """
            Custom media stream track that flips the video stream
            and passes it back to the source peer
            """

            kind = "video"

            def __init__(self, track):
                super().__init__()
                self.track = track

            # this is the essential method we need to implement 
            # to create a custom stream
            async def recv(self):

                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 0)

                # VideoFrames are from a really nice package called av
                # which is a pythonic wrapper around ffmpeg
                # and a dep of aiortc
                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                return new_frame
            

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pcs[peer_id]:
                print(f"Video Processor, {self.id}, connection state to {peer_id}: {self.pcs[peer_id].connectionState}")
        
        # when we receive a track from the source peer
        # we create a processed track and add it to the peer connection
        @self.pcs[peer_id].on("track")
        def on_track(track):

            from aiortc import RTCPeerConnection
            
            print(f"Video Processor, {self.id}, received {track.kind} track from {peer_id}")
            
            # create processed track
            flipped_track = VideoFlipTrack(track)
            self.pcs[peer_id].addTrack(flipped_track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended():
                print(f"Video Processor, {self.id}, incoming video track from {peer_id} ended")

    # add frontend end point
    @modal.asgi_app(label="webrtc-video-processor")
    def web_endpoints(self):

        from fastapi.responses import HTMLResponse

        # serve the frontend
        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)
        
        return self.web_app  

# create an output volume to store the transmitted videos
output_volume = modal.Volume.from_name("aiortc-video-processing", create_if_missing=True)
OUTPUT_VOLUME_PATH = Path("/output")

@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
@modal.concurrent(max_inputs=100)
class WebRTCVideoProcessorTester(WebRTCPeer):

    TEST_VIDEO_SOURCE_FILE = "/media/cliff_jumping.mp4"
    TEST_VIDEO_RECORD_FILE = OUTPUT_VOLUME_PATH / "flipped_test_video.mp4"
    DURATION_DIFFERENCE_THRESHOLD_FRAMES = 5
    VIDEO_DURATION_BUFFER_SECS = 5.0

    async def initialize(self):

        import cv2

        self.input_filepath = self.TEST_VIDEO_SOURCE_FILE
        self.output_filepath = self.TEST_VIDEO_RECORD_FILE

        # get input video duration in seconds
        self.input_video = cv2.VideoCapture(self.input_filepath)
        self.input_video_duration = self.input_video.get(cv2.CAP_PROP_FRAME_COUNT) / self.input_video.get(cv2.CAP_PROP_FPS)
        self.input_video.release()

        # set streaming duration to input video duration plus a buffer
        self.stream_duration = self.input_video_duration + self.VIDEO_DURATION_BUFFER_SECS

        self.player = None # video stream source
        self.recorder = None # processed video stream sink

    async def setup_streams(self, peer_id):

        from aiortc.contrib.media import MediaPlayer, MediaRecorder
        
        # setup video player and to peer connection
        self.video_src = MediaPlayer(self.input_filepath)
        self.pcs[peer_id].addTrack(self.video_src.video)

        # setup video recorder
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)
        self.recorder = MediaRecorder(self.output_filepath)

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Video Tester connection state updated: {self.pcs[peer_id].connectionState}")

        # when we receive a track back from 
        # the video processing peer we record it 
        # to the output file
        @self.pcs[peer_id].on("track")
        def on_track(track):
            
            print(f"Video Tester received {track.kind} track from {peer_id}")
            # record track to file
            self.recorder.addTrack(track)
            
            @track.on("ended")
            async def on_ended():
                print(f"Video Tester's processed video stream ended")
                # stop recording when incoming track ends to finish writing video
                await self.recorder.stop()
                # reset recorder and player
                self.recorder = None
                self.video_src = None
                
    async def run_streams(self, peer_id):

        import asyncio

        print(f"Video Tester running streams for {peer_id}...")

        # MediaRecorders need to be started manually
        # but in most cases the stream is already streaming
        await self.recorder.start()

        # run until sufficient time has passed
        await asyncio.sleep(self.stream_duration)

        # close peer connection manually
        await self.pcs[peer_id].close()

    # confirm that the output video is (nearly) the same length as the input video
    # we lose a few frames at the beginning
    def confirm_recording(self):

        import cv2

        # compare output video length to input video length
        input_video = cv2.VideoCapture(self.input_filepath)
        input_video_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video.release()

        output_video = cv2.VideoCapture(self.output_filepath)
        output_video_length = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video.release()

        if (input_video_length - output_video_length) < self.DURATION_DIFFERENCE_THRESHOLD_FRAMES:
            return True
        else:
            return False
               
    async def run_video_processing_test(self):

        import json
        import websockets

        peer_id = None
        # setup WebRTC connection using websockets
        ws_uri = WebRTCVideoProcessor().web_endpoints.web_url.replace("http", "ws") + f"/ws/{self.id}"
        async with websockets.connect(ws_uri) as websocket:

            await websocket.send(json.dumps({"type": "identify"}))
            peer_id = json.loads(await websocket.recv())["peer_id"]

            offer_msg = await self.generate_offer(peer_id)
            await websocket.send(json.dumps(offer_msg))

            try: 
                # receive answer
                answer = json.loads(await websocket.recv())

                if answer.get("type") == "answer":
                    await self.handle_answer(peer_id, answer)

            except websockets.exceptions.ConnectionClosed as e:
                await websocket.close()

        # loop until video player is finished
        if peer_id:
            await self.run_streams(peer_id)

    @modal.asgi_app(label="webrtc-video-processor-tester")
    def web_endpoints(self):
                
        @self.web_app.get("/run_test")
        async def run_test():
            await self.run_video_processing_test()
            return True

        @self.web_app.get("/check_test")
        async def check_test():
            return self.confirm_recording()

        return self.web_app
    
# set timeout for health checks and connection test
MINUTES = 60  # seconds
TEST_TIMEOUT = 2.0 * MINUTES  

# trigger the test locally
def trigger_webrtc_test():

    print(f"Triggering WebRTC connector at {WebRTCVideoProcessorTester().web_endpoints.web_url + '/run_test'}")
    test_triggered, start, delay = False, time.time(), 10
    while not test_triggered:
        try:
            with urllib.request.urlopen(WebRTCVideoProcessorTester().web_endpoints.web_url + "/run_test") as response:
                if response.getcode() == 200:
                    test_triggered = response.read().decode()
        except Exception as e:
            print(f"Error: {e}")
            if time.time() - start > TEST_TIMEOUT:
                break
            time.sleep(delay)
    
    return test_triggered
    
# check that the test was successful locally
def check_successful_test():

    headers = {
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        WebRTCVideoProcessorTester().web_endpoints.web_url + "/check_test",
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

# run tests
@app.local_entrypoint()
def main():

    assert trigger_webrtc_test(), "Test failed to trigger"
    assert check_successful_test(), "Test faileda to complete"



    