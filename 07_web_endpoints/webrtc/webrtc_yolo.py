# ---
# deploy: true
# ---

# # Real-time Webcam Object Detection with WebRTC

# This example combines WebRTC's peer-to-peer video streaming capabilities with
# Modal's efficient GPU scaling to deploy a real-time, browser-based object detection app.
#
# ## What is WebRTC?
#
# WebRTC (Web Real-Time Communication) is a framework that allows real-time media streaming between browsers (and other services).
# It powers Zoom, Twitch, Peloton, and a host of other apps that got us through the pandemic.
#
# What makes WebRTC so effective and different from
# other low latency web-based communications (e.g. WebSockets) is that it's purpose built for media streaming
# by enabling two devices on the web to
# - establish a direct, bidirectional, and managed UDP (or TCP) connection via NAT hole-punching and
# - coordinate their capabilities, like media codecs and connection .
#
# The term WebRTC refers to both the protocol and API implementations - the primary implementation being
# the JavaScript API; however, there are other implementations such as `aiortc` in Python. We'll use both
# in this example.
#
# ### How does it work?
#
# The simple WebRTC app that you'll find in most explainers consists of three players:
# 1. A peer that initiates the connection
# 2. A peer that responds to the connection
# 3. A signaling server that passes messages between the two peer.
#
# #### DIAGRAM
#
# The connection is established using a quick back and forth. The initating peer offers up a description of itself -
# its media sources, codec capabilities, IP information, etc - to the other peer through the server. The other peer considers that info,
# and answers with a description of itself. The info itself is generated and saved using the getter/setters for
# the local (this peer) and remote (the other peer) descriptions provided by the WebRTC API implementation.
#
# #### DIAGRAM
#
# Once these messages have been relayed and saved, there's a brief pause, and then... you're live. The streams are flowing. It just works.
#
# Obviously there's more going on under the hood, and there are the RFCs and excellent, in-depth explainers out there if you want to deep dive.
# Here, we'll give you enough info to make sure you're aware of all the essential parts, some of the important nuances that can trip you up,
# and be prepared to start buildling your own real-time app with Modal.
#
# #### Messages, the Session Description Protocol (SDP), and ICE Candidates
#
# Messages are implemented as dictionaries with a `type` key that holds a string describing the message type (e.g. `offer`, `answer`, `candidate`)
# and a `spd` key that contains the SDP encoded string. Most apps will add their own custom types to handle app-specific logic.
#
# SDP defines the format of the messages exchanged between peers. It's a text-based format that allows peers to describe
# their media capabilities, negotation roles, and ICE candidates. You can probably survive without groking this spec fully,
# but it doesn't hurt when you're in the throes of debugging a connection issue.
#
# ICE (Internet Connectivity Establishment) is the protocol that drives hole-punching in WebRTC. It solves the problem of allowing two devices to route to each others ports
# without needing to do any special configuration beforehand. Each peer makes a request to either a STUN or TURN server which then sends it back a list of candidates.
# These candidates are basically all of the possible IPs and ports that make up the layers of firewalls and NATs between the peer and its public IP. This process is called ICE gathering.
# The peers exchange these candidates and then coordinate testing candidate pairs until they successfully connect.
#
# ##### STUN and TURN servers
#
# STUN servers allow peers to discover their public IP addresses and ports and establish a direct connection over TCP or UDP. They don't require any authentication, and usually using
# Google's public STUN server is sufficient.
#
# TURN servers are used when one or both peers are behind restrictive NATs that don't allow direct connections using the information in the ICE candidates provided by a STUN server. A TURN
# server acts as a relay between the peers. These servers enable WebRTC connections under more conditions (e.g. cellular networks), but require authentication and add a small latency due to the extra hop.
#
# #### Navigating an asynchronous negotation
#
# The sequence of events involved in connecting two peers is called **the negotation**, and like any negotiation, you can easily ---- it up.
# What makes it particularly tricky is that the negotiation is asynchronous and only observable to us mere application developers through events and browser dashboards (which are honestly pretty sick).
# Here's a detailed breakdown of (one form of) the negotiation process from the RFC:
#
# ![WebRTC Negotiation](https://www.w3.org/TR/2013/WD-webrtc-20130910/images/ladder-2party-simple.svg)
#
# In practice, there are a few things to watch out for:
#
# 1. There are two ways to send ICE candidates to the other peer:
#    - The first is to send them embedded in the offer/answer SDP fields. To do this, the intiating peer generates an offer and sets it to its local description. Setting the local description triggers ICE gathering and waits for it to complete.
# At that point, you can throw out the original offer and send the SDP string assigned to your local description - which will now contain the ICE candidates! This is the simplest way to send candidates.
#    - The second approach is callled Trickle ICE. Here, you
#
# 2. The first peer to send an offer/answer message will generally wait for the other peer to respond before sending its candidates.
#


import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcServer

# set up video processing image

py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

video_processing_image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",  # uncomment w sed
        "locale-gen en_US.UTF-8",  # set locale
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LD_LIBRARY_PATH": tensorrt_ld_path, "LANG": "en_US.UTF-8"})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg")
)

# now we can install Python packages

video_processing_image = video_processing_image.pip_install(
    "aiortc==1.11.0",
    "fastapi==0.115.12",
    "huggingface-hub==0.30.2",
    "onnxruntime-gpu==1.21.0",
    "opencv-python==4.11.0.86",
    "tensorrt==10.9.0.34",
    "torch==2.7.0",
    "shortuuid==1.0.13",
)

# instantiate our app
app = modal.App("example-yolo-webrtc")

# create an output volume to store the transmitted videos and model weights
# we cache the model weights from hf hub
# as well as the onnx inference graph
# the graph can take a few minutes to build
# the very first time you run the app
# recommend using `modal run`.

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")

cache = {CACHE_PATH: CACHE_VOLUME}

# add TURN server credentials
turn_secret = modal.Secret.from_dotenv()  # TODO: Modal Secret


@app.cls(
    image=video_processing_image,
    gpu="A100-40GB",
    volumes=cache,
    secrets=[turn_secret],
)
@modal.concurrent(
    # input concurrency helps with faster restarts of stream
    # by avoiding initliazing a new container for every streaming
    # call. it takes ~15 sec to load the onnx model/session for each
    # container
    target_inputs=4,
    max_inputs=6,
)
class ObjDet(ModalWebRtcPeer):
    yolo_model = None

    async def initialize(self):
        import numpy as np
        import onnxruntime
        from aiortc import MediaStreamTrack
        from aiortc.contrib.media import VideoFrame

        from .yolo import YOLOv10

        onnxruntime.preload_dlls()
        self.yolo_model = YOLOv10(CACHE_PATH)

        class YOLOTrack(MediaStreamTrack):
            """
            Custom media stream track that flips the video stream
            and passes it back to the source peer
            """

            kind: str = "video"
            conf_threshold: float = 0.15

            def __init__(self, track: MediaStreamTrack, model) -> None:
                super().__init__()
                self.track = track
                self.yolo_model = model

            def detection(self, image: np.ndarray) -> np.ndarray:
                import cv2

                orig_shape = image.shape[:-1]

                image = cv2.resize(
                    image,
                    (self.yolo_model.input_width, self.yolo_model.input_height),
                )

                image = self.yolo_model.detect_objects(image, self.conf_threshold)

                image = cv2.resize(image, (orig_shape[1], orig_shape[0]))

                return image

            # this is the essential method we need to implement
            # to create a custom MediaStreamTrack
            async def recv(self) -> VideoFrame:
                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")

                processed_img = self.detection(img)

                # VideoFrames are from a really nice package called av
                # which is a pythonic wrapper around ffmpeg
                # and a dep of aiortc
                new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                return new_frame

        self.processing_track_cls = YOLOTrack

    async def setup_streams(self, peer_id: str):
        from aiortc import MediaStreamTrack

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.pcs[peer_id]:
                print(
                    f"Video Processor, {self.id}, connection state to {peer_id}: {self.pcs[peer_id].connectionState}"
                )

        # when we receive a track from the source peer
        # we create a processed track and add it to the peer connection
        @self.pcs[peer_id].on("track")
        def on_track(track: MediaStreamTrack) -> None:
            print(
                f"Video Processor, {self.id}, received {track.kind} track from {peer_id}"
            )

            output_track = self.processing_track_cls(track, self.yolo_model)
            self.pcs[peer_id].addTrack(output_track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended() -> None:
                print(
                    f"Video Processor, {self.id}, incoming video track from {peer_id} ended"
                )

    # some free turn servers that can handle up to 5 GB of traffic
    async def get_turn_servers(self, peer_id=None, msg=None) -> dict:
        import os

        creds = {
            "username": os.environ["TURN_USERNAME"],
            "credential": os.environ["TURN_CREDENTIAL"],
        }

        turn_servers = [
            {"urls": "stun:stun.relay.metered.ca:80"},
            {"urls": "turn:standard.relay.metered.ca:80"} | creds,
            {"urls": "turn:standard.relay.metered.ca:80?transport=tcp"} | creds,
            {"urls": "turn:standard.relay.metered.ca:443"} | creds,
            {"urls": "turns:standard.relay.metered.ca:443?transport=tcp"} | creds,
        ]

        return {"type": "turn_servers", "ice_servers": turn_servers}


webrtc_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "shortuuid==1.0.13",
    )
)


assets_parent_directory = Path(__file__).parent.resolve()

server_image = webrtc_base_image.add_local_dir(
    os.path.join(assets_parent_directory, "frontend"), remote_path="/frontend"
)


# for the server, all we have to do is
# let it know which ModalWebRTCPeer subclass to spawn
# attach our front end
@app.cls(image=server_image)
class WebcamObjDet(ModalWebRtcServer):
    modal_peer_cls = ObjDet

    def initialize(self):
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        # frontend files
        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)


# run test
@app.local_entrypoint()
def test():
    input_frames, output_frames = TestPeer().run_video_processing_test.remote()
    # allow a few dropped frames from the connection starting up
    assert input_frames - output_frames < 5, "Streaming failed"


# extra code just for testing


@app.cls(image=webrtc_base_image, volumes=cache)
class TestPeer(ModalWebRtcPeer):
    TEST_VIDEO_SOURCE_URL = "https://modal-cdn.com/cliff_jumping.mp4"
    TEST_VIDEO_RECORD_FILE = CACHE_PATH / "flipped_test_video.mp4"
    # extra time to run streams beyond input video duration
    VIDEO_DURATION_BUFFER_SECS = 5.0
    # allow time for container to spin up (can timeout with default 10)
    WS_OPEN_TIMEOUT = 30

    async def initialize(self) -> None:
        import cv2

        # get input video duration in seconds
        self.input_video = cv2.VideoCapture(self.TEST_VIDEO_SOURCE_URL)
        self.input_video_duration_frames = self.input_video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        self.input_video_duration_seconds = (
            self.input_video_duration_frames / self.input_video.get(cv2.CAP_PROP_FPS)
        )
        self.input_video.release()

        # set streaming duration to input video duration plus a buffer
        self.stream_duration = (
            self.input_video_duration_seconds + self.VIDEO_DURATION_BUFFER_SECS
        )

        self.player = None  # video stream source
        self.recorder = None  # processed video stream sink

    async def setup_streams(self, peer_id: str) -> None:
        from aiortc import MediaStreamTrack
        from aiortc.contrib.media import MediaPlayer, MediaRecorder

        # setup video player and to peer connection
        self.video_src = MediaPlayer(self.TEST_VIDEO_SOURCE_URL)
        self.pcs[peer_id].addTrack(self.video_src.video)

        # setup video recorder
        if os.path.exists(self.TEST_VIDEO_RECORD_FILE):
            os.remove(self.TEST_VIDEO_RECORD_FILE)
        self.recorder = MediaRecorder(self.TEST_VIDEO_RECORD_FILE)

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            print(
                f"Video Tester connection state updated: {self.pcs[peer_id].connectionState}"
            )

        # when we receive a track back from
        # the video processing peer we record it
        # to the output file
        @self.pcs[peer_id].on("track")
        def on_track(track: MediaStreamTrack) -> None:
            print(f"Video Tester received {track.kind} track from {peer_id}")
            # record track to file
            self.recorder.addTrack(track)

            @track.on("ended")
            async def on_ended() -> None:
                print("Video Tester's processed video stream ended")
                # stop recording when incoming track ends to finish writing video
                await self.recorder.stop()
                # reset recorder and player
                self.recorder = None
                self.video_src = None

    async def run_streams(self, peer_id: str) -> None:
        import asyncio

        print(f"Video Tester running streams for {peer_id}...")

        # MediaRecorders need to be started manually
        # but in most cases the stream is already streaming
        await self.recorder.start()

        # run until sufficient time has passed
        await asyncio.sleep(self.stream_duration)

        # close peer connection manually
        await self.pcs[peer_id].close()

    def count_frames(self):
        import cv2

        # compare output video length to input video length
        output_video = cv2.VideoCapture(self.TEST_VIDEO_RECORD_FILE)
        output_video_duration_frames = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video.release()

        return self.input_video_duration_frames, output_video_duration_frames

    @modal.method()
    async def run_video_processing_test(self) -> bool:
        import json

        import websockets

        peer_id = None
        # connect to server via websocket
        ws_uri = WebcamObjDet().web.web_url.replace("http", "ws") + f"/ws/{self.id}"
        print(f"ws_uri: {ws_uri}")
        async with websockets.connect(
            ws_uri, open_timeout=self.WS_OPEN_TIMEOUT
        ) as websocket:
            await websocket.send(json.dumps({"type": "identify", "peer_id": self.id}))
            peer_id = json.loads(await websocket.recv())["peer_id"]

            offer_msg = await self.generate_offer(peer_id)
            await websocket.send(json.dumps(offer_msg))

            try:
                # receive answer
                answer = json.loads(await websocket.recv())

                if answer.get("type") == "answer":
                    await self.handle_answer(peer_id, answer)

            except websockets.exceptions.ConnectionClosed:
                await websocket.close()

        # loop until video player is finished
        if peer_id:
            await self.run_streams(peer_id)

        return self.count_frames()
