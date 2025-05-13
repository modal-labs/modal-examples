# ---
# deploy: true
# ---

# # Real-time Webcam Object Detection with WebRTC

# This example combines WebRTC's peer-to-peer media streaming capabilities with Modal's efficient GPU scaling to deploy a real-time, browser-based object detection app.

# ## What is WebRTC?

# WebRTC (Web Real-Time Communication) is a framework that allows real-time media streaming between browsers (and other services). It powers Zoom, Twitch, and a host of other apps that got us through the pandemic. What makes WebRTC so effective and different from other low latency web-based communications (e.g. WebSockets) is that its stack and API is purpose built for media streaming.
# <figure align="middle">
#   <img src="https://hpbn.co/assets/diagrams/f91164cbbb944d8986c90a1e93afcd82.svg" width="60%" />
#   <figcaption>HTTP-based web stacks (left) and the WebRTC stack (right).</figcaption>
# </figure>

# In particular, it defines how two peers can establish a direct, bidirectional, and managed UDP (or TCP) connection using the ICE protocol. It then layers on security and real-time streaming protocols. WebRTC specifies both the protocol for establishing the connection and the API - the primary implementation being the JavaScript API; however, there are other implementations such as `aiortc` in Python. We'll use `aiortc` in this example, but the Github repo also includes a frontend for a web-based peer using the Javascript API.

# ### How does it work?

# A simple WebRTC app generally consists of three players:
# 1. a peer that initiates the connection (the offer)
# 2. a peer that responds to the connection (the answer), and
# 3. a server that passes messages between the two peers (signaling).

# First, the initating peer offers up a description of itself - its media sources, codec capabilities, IP information, etc - to the other peer through the server. Then, the other peer either accepts the offer by providing a compatible description of its own capabilities or rejects it if no compatible configuration is possible.

# Once the peers have exchanged the necessary information, there's a brief pause... and then you're live. It just works.

# <figure align="middle">
#   <img src="https://www.mdpi.com/futureinternet/futureinternet-12-00092/article_deploy/html/images/futureinternet-12-00092-g001.png" width="50%" />
#   <figcaption>Your basic WebRTC app.</figcaption>
# </figure>

# ### But how does it _really_ work?

# Obviously there's more going on under the hood. If you want to deep dive into WebRTC, we recommend checking out the RFCs or a more-thorough explainer. Here, we'll give you enough info to make sure you're aware of the essential parts, some of the nuances that can trip you up, and be prepared to start buildling your own real-time app with Modal.

# #### Messages, SDP, and ICE Candidates

# **Messages** are implemented as dictionaries (JSON, Python `dict`s) with a `type` key whose value is a string describing... the message type (e.g. `offer`, `answer`). For WebRTC-specific messages, there will also be an `spd` key that contains the SDP encoded string. Most apps will add their own custom types and keys (e.g. sender ids) to handle app-specific logic.

# **SDP (Session Description Protocol)** defines the format of the messages exchanged between peers. It's a text-based format that allows peers to describe their media capabilities, negotation roles, and ICE candidates. You can probably survive without groking this spec fully, but it's probably inevitable once you're in the throes of debugging a connection issue.

# **ICE (Internet Connectivity Establishment)** is the protocol that drives NAT hole-punching in WebRTC. It solves the problem of allowing two devices to set up a direct UDP/TCP connection without needing to re-configure routers or firewalls. Each peer requests a list of "candidates" from either a STUN or TURN server. These candidates contain the IP and port addresses between that peer's local IP address and public IP address, i.e. the addresses of all the routers and the peer itself. The peers exchange these candidates and then coordinate testing candidate pairs until they successfully connect.

# <figure align="middle">
#   <img src="https://miro.medium.com/v2/resize:fit:1302/1*HmMdrpVBTP2vYMhrVOdNOw.jpeg" width="50%" />
#   <figcaption>A higher resolution view of a WebRTC app.</figcaption>
# </figure>

# #### STUN and TURN servers

# **STUN** servers allow peers to discover their public IP addresses and ports and establish a direct connection over TCP or UDP. They don't require any authentication. Using Google's public STUN server is usually sufficient.

# **TURN** servers are used when a peer is behind a restrictive NAT or firewall configuration that blocks direct connections. By requiring authentication, a TURN server is able to establish trust and directly connect to a peer. It then relays packets to the other peer - either directly if that peer can use STUN or indirectly through another TURN server (potentially itself) if both peers are behind strict networking rules. The tradeoff is that the extra hops add latency and developers must either run their own TURN server or pay to use one operated by a third-party.

# #### Behind the scenes

# WebRTC implementations spawn several workers who asychronously take care of things like handling ICE candidates and media streaming. While we gladly accept these workers' help, it can also make building a WebRTC app a little tricky since you, as the application developer, only directly control and observe part of the flow.

# What's more, each WebRTC API may implement some of this work differently *or not all*. If you're going to build an app with WebRTC, make sure to familiarize yourself with the idiosynchrocies of your chosen API - and probably the common browsers as well.directly

# <figure align="middle">
#   <img src="https://hpbn.co/assets/diagrams/f38aae954de1cde63e2dffddc23a13f3.svg" width="50%" />
#   <figcaption>WebRTC BTS</figcaption>
# </figure>

# When debugging, it's helpful to provide listeners to WebRTC's events and, if a peer is web-based, using the browser's WebRTC debugging tools.

# ## Mapping WebRTC connection lifetimes to Modal function call lifetimes

# With Modal, the `FunctionCall` is the essential unit of execution, and it doesn't make any promises that two calls made in quick succession will be run on the same container. When functions are called, more compute is provided, and when they return, that compute is released. This is necessary for Modal to provide dev-friendly auto-scaling that make it so useful.

# The WebRTC protocol, on the other hand, involves passing messages back and forth, coordinating with asynchronous agents, and running a P2P connection in the background below the application layer. Standard WebRTC apps require coordinating many function calls and may even be idle once the connection has been established.

# If we don't carefully reconcile this disparity, we won't be able to properly leverage Modal's auto-scaling or concurrency features and could end up with bugs like prematurely cancelled streams. For example, we should't use HTTP for signaling because each message requires a new call. Modal doesn't know these calls are related and therefore can't promise to send them to the same server instance. Likewise, if we return from the call to the Modal GPU peer while the WebRTC connection is still active, Modal will scaledown as if the instance was idle.

# To align our WebRTC app with Modal's assumptions, it needs to meet the following requirements:

# - **The client peer only makes one call to the signaling server which returns after the connection is established.**

#     We can use a WebSocket for persistant, bidirectional communication between the client peer and the signaling server running on Modal. When the client detects that the P2P connection has been established, it can close the WebSocket which will in turn end the call.

# - **The server only makes one call to the Modal GPU peer which only returns once the connection has been closed, i.e. the user has finished processing the media stream.**

#     To meet this requirement we have the call the GPU peer via `.spawn` which doesn't block the server process and therefore decouples the server and GPU peer function calls. We also pass a `modal.Queue` to the GPU peer in the spawned function call which we use to pass messages between it and the server. When signaling finishes, the GPU peer goes into a loop until it detects that the P2P connection has been closed.

# #### DIAGRAM

# ## Building the object detection app with WebRTC, YOLO, and Modal

# Now let's see how we can combine WebRTC with Modal's on-demand GPUs to build a scalable, real-time object dection app.

# We'll use `aiortc`, Python's lowest-level WebRTC API and run the signaling server and both peers on Modal as `Cls`es. One peer will stream a video file to the other, who will then run inference and return the annotated video as a stream.

# > **_NB_**: We also wave a web frontend that streams a device's webcam using the local browser and the Javascript WebRTC API.

# ### `ModalWebRtc`

# To help you out, we've implemented two classes that abstract away most of the WebRTC and design details. One for a peer and one for the server. We've also ensured that these classes are Modal-ready and even pre-applied some decorators.

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
