# ---
# lambda-test: false
# ---

# # Real-time Webcam Object Detection with WebRTC

# This example demonstrates how to use WebRTC with Modal for real-time, low latency inference on media streams that's serverless and scales efficiently.

# ## What is WebRTC?

# WebRTC (Web Real-Time Communication) is a protocol and API specification for real-time media streaming between peers.
# What makes it so effective and different from other low latency web-based communications (e.g. WebSockets) is that its stack and API are purpose built for media streaming.
# It's primarily designed for browser applications using the JavaScript API, but [APIs exist for other languages](https://www.webrtc-developers.com/did-i-choose-the-right-webrtc-stack/).
# We'll build our app using Python's [`aiortc`](https://aiortc.readthedocs.io/en/latest/) package.

# A simple WebRTC app generally consists of three players:
# 1. a peer that initiates the connection
# 2. a peer that responds to the connection, and
# 3. a server that passes some initial messages between the two peers.

# First, one peer initiates the connection by offering up a description of itself - its media sources, codec capabilities, IP information, etc - which is relayed to another peer through the server.
# The other peer then either accepts the offer by providing a compatible description of its own capabilities or rejects it if no compatible configuration is possible.
# This process is called "signaling" or sometimes the "negotation" in the WebRTC world, and the server that mediates it is usually called the "signaling server".

# Once the peers have agreed on a configuration there's a brief pause... and then you're live.

# <figure align="middle">
#   <img src="https://modal-cdn.com/cdnbot/just_webrtcnhhr0n2h_412df868.webp" width="95%" />
#   <figcaption>Your basic WebRTC app.</figcaption>
# </figure>

# Obviously there’s more going on under the hood.
# If you want to deep dive, we recommend checking out the [RFCs](https://www.rfc-editor.org/rfc/rfc8825) or a [more-thorough explainer](https://webrtcforthecurious.com/).
# In this document, we'll focus on the Modal-specific details.

# ## Stateless signaling

# Modal lets you turn your functions into scalable, GPU-powered cloud services.
# When you call a Modal function, you get a GPU.
# When you call 1000 Modal functions, you get 1000 GPUs.
# When your functions return, you have 0 GPUs.

# A core assumption of Modal that makes this possible is that function calls are independent and self-contained.
# In other words, Modal functions are _stateless_ and they shouldn't launch other processes or tasks which continue working after the function call returns.

# WebRTC apps, on the other hand, require passing messages back and forth, and APIs spawn several "agents" which do work behind the scenes - including managing the P2P connection itself.
# This means that streaming may only just have just begun when our application logic has finished.

# <figure align="middle">
#     <img src="https://modal-cdn.com/cdnbot/sequence_diagramsyt1upmqk_bdb00440.webp" width="95%" />
#     <figcaption>Modal's stateless autoscaling (left) and WebRTC's stateful signaling (right).</figcaption>
# </figure>

# To ensure we properly leverage Modal's autoscaling and concurrency features, we need to align the signaling and streaming liftetimes with our Modal function call lifetimes.

# We'll handle passing messages between the client peer and the signaling server using a
# [WebSocket](https://modal.com/docs/guide/webhooks#websockets) for persistant, bidirectional communication within a single function call (in this case the Modal call is a web endpoint).
# We'll also [`.spawn`](https://modal.com/docs/reference/modal.Function#spawn) the cloud peer inside the WebSocket endpoint
# and pass messages with it using a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue).
#
# We can then use the state of the P2P connection to determine when to return from the calls to both the signaling server and the cloud peer.
# When the P2P connection has been _established_, we'll close the WebSocket which in turn ends the call to the signaling server.
# And when the P2P connection has been _closed_, we'll return from the call to the cloud peer.


# <figure align="middle">
#   <img src="https://modal-cdn.com/cdnbot/modal_webrtcwx3nrjhp_0f47e9ff.webp" width="95%" />
#   <figcaption>Connecting with Modal using WebRTC.</figcaption>
# </figure>

# We wrote two classes, `ModalWebRtcPeer` and `ModalWebRtcSignalingServer`, to abstract away all of that stuff as well as a lot of the `aiortc` implementation details.
# They're also decorated with Modal [lifetime hooks](https://modal.com/docs/guide/lifecycle-functions).
# Add the [`app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator and some custom logic, and you're ready to deploy on Modal.

# You can find them in the `modal_webrtc.py` file provided alongside this example in the [Github repo](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/modal_webrtc.py).

# ## Building the app

# For our WebRTC app, we'll take a client's video stream, YOLO it with an A100 GPU on Modal, and then stream the annotated video back to the client. Let's get started!

import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

# ### Implementing the YOLO `ModalWebRtcPeer`

# We're going to run YOLO on an A100 GPU with the ONNX Runtime and TensorRT.
# With this setup, we can achieve inference times between 2-4 milliseconds per frame and RTTs below video frame rates (usually around 30 milliseconds per frame).

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
        "huggingface-hub[hf_xet]==0.30.2",
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
# We'll use an A100 GPU and grab some secrets from Modal (creds for some free TURN servers - see below).
# - Implement the method `setup_streams`.
# This is where we'll use `aiortc` to add the logic for processing the incoming video track with YOLO and returning an annotated video track to the source peer.

# `ModalWebRtcPeer` has a few other methods that users can optionally implement:
# - `initialize()`: Any custom initialization logic, called when `@modal.enter()` is called
# - `run_streams()`: Logic for starting streams. This is necessary when the peer is the source of the stream.
# This is where you'd ensure a webcam was running or start playing a video file (see the TestPeer class)
# - `get_turn_servers()`: We haven't talked about TURN servers, but just know that they're necessary if you want to use WebRTC behind strict NAT or firewall configurations.
# If you don't provide TURN servers you can still use your app using the default STUN servers on many networks.
# - `exit()`: Any custom cleanup logic, called when `@modal.exit()` is called.

# In our case, we'll load the YOLO model in `initialize` and provide TURN server information.
# We're also going to use the `@modal.concurrent` decorator to allow multiple instances of our peer to run on one GPU.

# **Setting the Region**
# To optimize latency, we want to make the physical distance of the P2P connection
# between your local machine and the GPU container as short as possible.
# We'll use the `region` parameter of the `cls` decorator to set the region of the GPU container.
# You should set this to the closest region to your local machine or users.
# See the [region selection](https://modal.com/docs/guide/region-selection) guide for more information.


@app.cls(
    image=video_processing_image,
    gpu="A100-40GB",
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials", environment_name="examples")],
    region="us-west",  # set to your region
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
        # we create a processed track and add it to our stream
        # back to the source peer
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


# ### Implementing the `ModalWebRtcSignalingServer`

# The `ModalWebRtcSignalingServer` class is much simpler to implement.
# The only thing we need to do is implement the `get_modal_peer_class` method which will return our implementation of the `ModalWebRtcPeer` class, `ObjDet`.
#
# It also has an `initialize()` method we can optionally override (which is called when `@modal.enter()` is called)
# as well as a `web_app` property which will be [served by Modal](https://modal.com/docs/guide/webhooks#asgi-apps---fastapi-fasthtml-starlette).
# We'll use these to add a frontend which uses the WebRTC JavaScript API to stream a peer's webcam from the browser.
#
# The JavaScript and HTML files are alongside this example in the [Github repo](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/yolo).

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
class WebcamObjDet(ModalWebRtcSignalingServer):
    def get_modal_peer_class(self):
        return ObjDet

    def initialize(self):
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)


# ### YOLO helper functions

# The two functions below are used to set up the YOLO model and create our custom [`MediaStreamTrack`](https://aiortc.readthedocs.io/en/latest/api.html#aiortc.MediaStreamTrack).

# The first, `get_yolo_model` sets up the ONNXRuntime and loads the model weights.
# We call this in the `initialize` method of the `ModalWebRtcPeer` class
# so it only happens once per container.


def get_yolo_model(cache_path):
    import onnxruntime

    from .yolo import YOLOv10

    onnxruntime.preload_dlls()
    return YOLOv10(cache_path)


# The second, `get_yolo_track` creates a custom `MediaStreamTrack` that performs object detection on the video stream.
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
