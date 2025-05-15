# ---
# deploy: true
# ---

# # Real-time Webcam Object Detection with WebRTC

# This example combines WebRTC's peer-to-peer media streaming capabilities with Modal's efficient GPU scaling to deploy a real-time object detection app.

# ## What is WebRTC?

# WebRTC (Web Real-Time Communication) is a framework that allows real-time media streaming between browsers (and other services).
# It powers Zoom, Twitch, and a host of other apps that got us through the pandemic.
# What makes WebRTC so effective and different from other low latency web-based communications (e.g. WebSockets) is that its stack and API are purpose built for media streaming.

# A simple WebRTC app generally consists of three players:
# 1. a peer that initiates the connection
# 2. a peer that responds to the connection, and
# 3. a server that passes messages between the two peers.

# First, the initating peer offers up a description of itself - its media sources, codec capabilities, IP information, etc - to the other peer through the server.
# Then, the other peer either accepts the offer by providing a compatible description of its own capabilities or rejects it if no compatible configuration is possible.

# Once the peers have exchanged the necessary information, there's a brief pause... and then you're live. It just works.

# TODO: serve all images from modal-cdn
# <figure align="middle">
#   <img src="https://i.imgur.com/o4vgWsR.png" width="80%" />
#   <figcaption>Your basic WebRTC app.</figcaption>
# </figure>

# Obviously there’s more going on under the hood. If you want to deep dive into WebRTC, we recommend checking out the RFCs or a more-thorough explainer.
# For this demo, we're going to focus on how to use WebRTC with Modal's design patterns.

# ## Coupling WebRTC connection lifetimes to Modal function call lifetimes

# Modal let's you turn your functions into GPU-powered cloud services.
# When a function is called, Modal provisions the compute, and when it returns, that compute is released.
# A core feature of this design is that function calls are assumed to be indepenedent and self-contained, i.e. calls should be to run in any order and they shouldn't launch other processes or tasks which continue working after the function call returns.
# Modal's ability to dynamically scale compute resources as demand fluctuates is a direct consequence of this assumption.

# WebRTC apps, on the other hand, typically require coordinating many function calls to establish the connection between two peers.
# Addtionally, API implementations runs several asynchronous tasks below the application layer - including the P2P connection itself.
# This means that P2P streaming may only just have just begun when the application logic has returned.

# TODO: resize/scale these diagrams
# <figure align="middle" style="display: flex; justify-content: space-between;">
#   <div style="width: 45%;">
#     <img src="https://i.imgur.com/uQWgtLs.png" width="100%" />
#     <figcaption>Stateless is part of the design.</figcaption>
#   </div>
#   <div style="width: 45%;">
#     <img src="https://i.imgur.com/ZF4iKdQ.png" width="100%" />
#     <figcaption>A simplified view of a WebRTC negotiation.</figcaption>
#   </div>
# </figure>

# If we don't carefully reconcile this disparity, we won't be able to properly leverage Modal's auto-scaling or concurrency features and could end up with bugs like prematurely cancelled streams.
# For example, we should't use HTTP for signaling because each message requires a new call. Modal doesn't know these calls are related and therefore can't promise to send them to the same server instance. Likewise, if we return from the call to the Modal GPU peer while the WebRTC connection is still active, Modal will scaledown as if the instance was idle.

# To align our WebRTC app with Modal's assumptions, it needs to meet the following requirements:

# - **The client peer only makes one call to the signaling server which returns after the connection is established.**

#     We can use a WebSocket for persistant, bidirectional communication between the client peer and the signaling server running on Modal. When the client detects that the P2P connection has been established, it can close the WebSocket which will in turn end the call.

# - **The server only makes one call to the Modal GPU peer which only returns once the connection has been closed, i.e. the user has finished processing the media stream.**

#     To meet this requirement we have the call the GPU peer via `.spawn` which doesn't block the server process and therefore decouples the server and GPU peer function calls. We also pass a `modal.Queue` to the GPU peer in the spawned function call which we use to pass messages between it and the server. When signaling finishes, the GPU peer goes into a loop until it detects that the P2P connection has been closed.

# <figure align="middle">
#   <img src="https://i.imgur.com/FfslIg8.png" width="80%" />
#   <figcaption>Connecting with Modal using WebRTC.</figcaption>
# </figure>

# We wrote two classes, `ModalWebRtcPeer` and `ModalWebRtcServer`, to abstract away most of the boilerplate and ensure everything happens in the right order.
# The server class handles signaling and the peer class handles the WebRTC/`aiortc` stuff.
# They are also partially decorated as [`Cls`es](https://modal.com/docs/reference/modal.Cls).
# Add the `app.cls` decorator and some custom logic, and you're ready to deploy on Modal.

# ## Building the app

# You can find the `ModalWebRtcPeer` and `ModalWebRtcServer` classes in the `modal_webrtc.py` file provided alongside this example in the Github repo.

import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcServer

# ### Implementing a YOLO `ModalWebRtcPeer`

# We're going to run YOLO in the cloud on an A100 GPU with the ONNX Runtime and TensorRT. With this setup, we can achieve inference times between 2-4 milliseconds per frame and RTTs below video frame rates (30 fps -> 33.3 milliseconds).

# #### Set up the containers and runtime environments

# We'll start with a simple `Image` and then
# - set it up to properly use TensorRT and the ONNX Runtime,
# - install the necessary libs for processing video, `opencv` and `ffmpeg`, and
# - install the necessary Python packages.

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
    .pip_install(
        "aiortc==1.11.0",
        "fastapi==0.115.12",
        "huggingface-hub==0.30.2",
        "onnxruntime-gpu==1.21.0",
        "opencv-python==4.11.0.86",
        "tensorrt==10.9.0.34",
        "torch==2.7.0",
        "shortuuid==1.0.13",
    )
)

# We also need to create an output volume to store the model weights,
# ONNX inference graph, and other artifacts like a video file where
# we'll write out the processed video stream for testing.

# The very first time we run the app, downloading the model and building the ONNX inference graph
# will take a few minutes. After that, the we can load the cached
# weights and graph which reduces the startup time to about 15 seconds per container.

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("example-yolo-webrtc")

# Let's implement our `ModalWebRtcPeer` class to process an incoming video track with YOLO and return an annotated video track to the source peer.

# To implement a `ModalWebRtcPeer`, we need to:
# - Decorate our subclass with `@app.cls`. We'll use an A100 GPU and grab some secrets from Modal (you'll see what they're for in a moment).
# - Implement the method `setup_streams`. This is where we'll use `aiortc` to add the logic for processing the incoming video track with YOLO and returning an annotated video track to the source peer.

# `ModalWebRtcPeer` has a few other methods that users can optionally implement:
# - `initialize()`: Any custom initialization logic, called when `@modal.enter()` is called
# - `run_streams()`: Logic for starting streams. This is necessary when the peer is the source of the stream. This is where you'd ensure a webcam was running or start playing a video file (see the TestPeer class)
# - `get_turn_servers()`: We haven't talked about TURN servers yet, so for now just know that it's necessary if you want to use WebRTC behind strict NAT or firewall configurations.
# - `exit()`: Any custom cleanup logic, called when `@modal.exit()` is called.

# In our case, we'll load the YOLO model in `initialize` and also demonstrate how to provide TURN server information with `get_turn_servers`. We're also going to use the `@modal.concurrent` decorator to allow multiple instances of our peer to run on one GPU.


@app.cls(
    image=video_processing_image,
    gpu="A100-40GB",
    volumes=cache,
    secrets=[modal.Secret.from_dotenv()],  # TODO: Modal Secret
)
@modal.concurrent(
    target_inputs=3,
    max_inputs=4,
)
class ObjDet(ModalWebRtcPeer):
    async def initialize(self):
        self.yolo_model = get_yolo_model(CACHE_PATH)

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

            output_track = get_yolo_track(track, self.yolo_model)
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


# #### Implementing a `ModalWebRtcServer`

# The `ModalWebRtcServer` class is much simpler to implement. The only thing you need to do is provide the `ModalWebRtcPeer` subclass you want to use as the cloud peer. It also has an `initialize()` you can optionally override which is called when `@modal.enter()` is called - like in `ModalWebRtcPeer`.

# We're also going to add a frontend to the server which uses the JavaScript API to send a peer's webcam using a web browser. The `ModalWebRtcServer` class has a `web_app` property which is a `fastapi.FastAPI` instance that will be handled by Modal. We'll add the endpoints in the `initialize` method.
#
# The JavaScript and HTML files are alongside this example in the Github repo.

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "shortuuid==1.0.13",
    )
)  # we'll resuse this base image for the testing peer

assets_parent_directory = Path(__file__).parent.resolve()

server_image = base_image.add_local_dir(
    os.path.join(assets_parent_directory, "frontend"), remote_path="/frontend"
)


@app.cls(image=server_image)
class WebcamObjDet(ModalWebRtcServer):
    modal_peer_cls = ObjDet  # <---- setting the cloud peer

    def initialize(self):
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)


# #### YOLO helper functions
def get_yolo_model(cache_path):
    import onnxruntime

    from .yolo import YOLOv10

    onnxruntime.preload_dlls()
    return YOLOv10(cache_path)


def get_yolo_track(track, yolo_model=None):
    import numpy as np
    import onnxruntime
    from aiortc import MediaStreamTrack
    from aiortc.contrib.media import VideoFrame

    from .yolo import YOLOv10

    class YOLOTrack(MediaStreamTrack):
        """
        Custom media stream track performs object detection
        on the video stream and passes it back to the source peer
        """

        kind: str = "video"
        conf_threshold: float = 0.15

        def __init__(self, track: MediaStreamTrack, yolo_model=None) -> None:
            super().__init__()

            self.track = track
            if yolo_model is None:
                onnxruntime.preload_dlls()
                self.yolo_model = YOLOv10(CACHE_PATH)
            else:
                self.yolo_model = yolo_model

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

    return YOLOTrack(track)


# #### Testing with a local entrypoint and two `ModalWebRtcPeer`s


@app.local_entrypoint()
def test():
    input_frames, output_frames = TestPeer().run_video_processing_test.remote()
    # allow a few dropped frames from the connection starting up
    assert input_frames - output_frames < 5, "Streaming failed"


@app.cls(image=base_image, volumes=cache)
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
