# ---
# deploy: true
# cmd: ["cd", "07_web_endpoints", "&&", "modal", "run", "-m", "webrtc.webrtc_yolo"]
# ---

# # Real-time Webcam Object Detection with WebRTC

# This example combines WebRTC's peer-to-peer media streaming capabilities with Modal's efficient GPU scaling to deploy a real-time object detection app.

# ## What is WebRTC?

# WebRTC (Web Real-Time Communication) is a protocol and API specification for real-time media streaming between peers.
# What makes it so effective and different from other low latency web-based communications (e.g. WebSockets) is that its stack and API are purpose built for media streaming.
# It's primarily designed for browser applications using the Javascript API, but [APIs exist for other languages](https://www.webrtc-developers.com/did-i-choose-the-right-webrtc-stack/).
# We'll build our app using the [Python `aiortc` library](https://aiortc.readthedocs.io/en/latest/) to run a peer in the cloud.

# A simple WebRTC app generally consists of three players:
# 1. a peer that initiates the connection
# 2. a peer that responds to the connection, and
# 3. a server that passes messages between the two peers.

# First, the initating peer offers up a description of itself - its media sources, codec capabilities, IP information, etc - which is relayed to another peer through the server.
# The other peer then either accepts the offer by providing a compatible description of its own capabilities or rejects it if no compatible configuration is possible.

# Once the peers have agreed on a configuration there's a brief pause... and then you're live.

# <figure align="middle">
#   <img src="https://modal-cdn.com/cdnbot/just_webrtcnhhr0n2h_412df868.webp" width="95%" />
#   <figcaption>Your basic WebRTC app.</figcaption>
# </figure>

# Obviously there’s more going on under the hood.
# If you want to deep dive, we recommend checking out the [RFCs](https://www.rfc-editor.org/rfc/rfc8825) or a [more-thorough explainer](https://webrtcforthecurious.com/).
# In this document, we'll focus on the Modal-specific details.

# ## A stateless negotiation

# Modal let's you turn your functions into GPU-powered cloud services.
# When you call a Modal function, you get a GPU.
# When you call 1000 Modal functions, you get 1000 GPUs.
# When your functions return, you have 0 GPUs.

# A core feature of Modal's design that makes this possible is that function calls are assumed to be independent and self-contained.
# In other words, Modal functions are _stateless_ and they shouldn't launch other processes or tasks which continue working after the function call returns.

# WebRTC apps, on the other hand, require passing messages back and forth and APIs spawn several "agents" which do work behind the scenes - including managing the P2P connection itself.
# This means that streaming may only just have just begun when our application logic has finished.

# <figure align="middle">
#     <img src="https://modal-cdn.com/cdnbot/sequence_diagramsyt1upmqk_bdb00440.webp" width="95%" />
#     <figcaption>Modal's stateless autoscaling (left) and WebRTC's stateful P2P negotiation (right).</figcaption>
# </figure>

# If we don't carefully reconcile this disparity, we won't be able to properly leverage Modal's auto-scaling or concurrency features, and could end up with bugs like prematurely cancelled streams.
# For example, we can't use HTTP for signaling because each message requires a new call.
# We also need to ensure that the cloud peer doesn't return when the negotiation finishes.

# To align our WebRTC app with Modal's assumptions, it needs to meet the following requirements:

# - **The client peer only makes one call to the signaling server which returns after the connection is established.**

#     We'll use a WebSocket for persistant, bidirectional communication between the client peer and the signaling server running on Modal.
# When the client detects that the P2P connection has been established, it closes the WebSocket.

# - **The server only makes one call to the cloud peer which returns once the P2Pconnection has been closed.**

#     To meet this requirement, the server will the call the cloud peer using Modal's [`.spawn` method](https://modal.com/docs/reference/modal.Function#spawn).
# `spawn` doesn't block which decouples the server and cloud peer function calls.
# We also pass a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue) to the cloud peer in the spawned function call which we use to pass messages between it and the server.
# When signaling finishes, the function we spawned goes into a loop until it detects that the P2P connection has been closed.

# <figure align="middle">
#   <img src="https://modal-cdn.com/cdnbot/modal_webrtcjngux8vw_02988d57.webp" width="95%" />
#   <figcaption>Connecting with Modal using WebRTC.</figcaption>
# </figure>

# We wrote two classes, `ModalWebRtcPeer` and `ModalWebRtcServer`, to abstract away most of the boilerplate and ensure things happens in the correct order.
# The server class handles signaling and the peer class handles the WebRTC/`aiortc` stuff.
# They are also decorated with Modal [lifetime hooks](https://modal.com/docs/guide/lifecycle-functions).
# Add the [`app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator and some custom logic, and you're ready to deploy on Modal.

# ## Building the app

# You can find the `ModalWebRtcPeer` and `ModalWebRtcServer` classes in the `modal_webrtc.py` file provided alongside this example in the Github repo.

import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcServer

# ### Implementing the YOLO `ModalWebRtcPeer`

# We're going to run YOLO in the cloud on an A100 GPU with the ONNX Runtime and TensorRT.
# With this setup, we can achieve inference times between 2-4 milliseconds per frame and RTTs below video frame rates (usually around 30 milliseconds per frame).

# #### Setting up the containers and runtime environments

# We'll start with a simple [`modal.Image`](https://modal.com/docs/reference/modal.Image) and then
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

# The very first time we run the app, downloading the model and building the ONNX inference graph will take a few minutes.
# After that, the we can load the cached
# weights and graph which reduces the startup time to about 15 seconds per container.

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("example-yolo-webrtc")

# Let's implement our `ModalWebRtcPeer` class to process an incoming video track with YOLO and return an annotated video track to the source peer.

# To implement a `ModalWebRtcPeer`, we need to:
# - Decorate our subclass with `@app.cls`.
# We'll use an A100 GPU and grab some secrets from Modal (you'll see what they're for in a moment).
# - Implement the method `setup_streams`.
# This is where we'll use `aiortc` to add the logic for processing the incoming video track with YOLO and returning an annotated video track to the source peer.

# `ModalWebRtcPeer` has a few other methods that users can optionally implement:
# - `initialize()`: Any custom initialization logic, called when `@modal.enter()` is called
# - `run_streams()`: Logic for starting streams. This is necessary when the peer is the source of the stream.
# This is where you'd ensure a webcam was running or start playing a video file (see the TestPeer class)
# - `get_turn_servers()`: We haven't talked about TURN servers yet, so for now just know that it's necessary if you want to use WebRTC behind strict NAT or firewall configurations.
# - `exit()`: Any custom cleanup logic, called when `@modal.exit()` is called.

# In our case, we'll load the YOLO model in `initialize` and also demonstrate how to provide TURN server information with `get_turn_servers`.
# We're also going to use the `@modal.concurrent` decorator to allow multiple instances of our peer to run on one GPU.


@app.cls(
    image=video_processing_image,
    gpu="A100-40GB",
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials", environment_name="examples")],
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

            output_track = get_yolo_track(track, self.yolo_model)  # see Addenda
            self.pcs[peer_id].addTrack(output_track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended() -> None:
                print(
                    f"Video Processor, {self.id}, incoming video track from {peer_id} ended"
                )

    # some free turn servers we signed up forthat can handle up to 5 GB of traffic
    # when they hit the limit they'll stop working
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


# ### Implementing the `ModalWebRtcServer`

# The `ModalWebRtcServer` class is much simpler to implement.
# The only thing you need to do is provide the `ModalWebRtcPeer` subclass you want to use as the cloud peer.
# It also has an `initialize()` you can optionally override which is called when `@modal.enter()` is called - like in `ModalWebRtcPeer`.

# We're also going to add a frontend to the server which uses the JavaScript API to send a peer's webcam using a web browser.
# The `ModalWebRtcServer` class has a `web_app` property which is a `fastapi.FastAPI` instance that will be handled by Modal.
# We'll add the endpoints in the `initialize` method.
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


# ### YOLO helper functions

# You'll need these two functions to get the app to run.

# The first, `get_yolo_model` sets up the ONNXRuntime and loads the model weights.
# We call this in the `initialize` method of the `ModalWebRtcPeer` class
# so it only happens once per container (i.e. when `@modal.enter()` methods are called).


def get_yolo_model(cache_path):
    import onnxruntime

    from .yolo import YOLOv10

    onnxruntime.preload_dlls()
    return YOLOv10(cache_path)


# The second, `get_yolo_track` creates a custom `MediaStreamTrack` that performs object detection on the video stream.
#
# We call this in the `setup_streams` method of the `ModalWebRtcPeer` class
# so it happens once per peer connection.


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
            # and a dependency of aiortc
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

    return YOLOTrack(track)


# ## Testing

# First we define a `local_entrypoint` to run and evaluate the test.
# Our test will stream an .mp4 file to the cloud peer and record the annoated video to a new file.
# The test itself ensurse that the new video is no more than five frames shorter than the source file.
# The difference is due to dropped frames while the connection is starting up.


@app.local_entrypoint()
def test():
    input_frames, output_frames = TestPeer().run_video_processing_test.remote()
    # allow a few dropped frames from the connection starting up
    assert input_frames - output_frames < 5, "Streaming failed"


# Because our test will require Python dependencies outside the standard library, we'll run the test itself in a container on Modal.
# In fact, this will be another `ModalWebRtcPeer` class. So the test will also demonstrate how to setup WebRTC between Modal containers.
# There are some details in here regarding the use of `aiortc`'s `MediaPlayer` and `MediaRecorder` classes that won't cover here.
# Just know that these are `aiortc` specific classes - not a WebRTC thing.

# That said, using these classes does require us to manually `start` and `stop` streams.
# For example, we'll need to override the `run_streams` method to start the source stream, and we'll make use of the `on_ended` callback to stop the recording.


@app.cls(image=base_image, volumes=cache)
class TestPeer(ModalWebRtcPeer):
    TEST_VIDEO_SOURCE_URL = "https://modal-cdn.com/cliff_jumping.mp4"
    TEST_VIDEO_RECORD_FILE = CACHE_PATH / "test_video.mp4"
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
        # but in most cases the track is already streaming
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
