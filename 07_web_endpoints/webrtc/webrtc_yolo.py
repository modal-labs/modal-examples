# ---
# cmd: ["modal", "serve", "-m", "07_web_endpoints.webrtc.webrtc_yolo"]
# deploy: true
# ---

# # Real-time object detection with WebRTC and YOLO

# This example demonstrates how to architect a serverless real-time streaming application with Modal and WebRTC.
# The sample application detects objects in webcam video with YOLO.

# See the clip below from a live demo of this example in a course by [Kwindla Kramer](https://machine-theory.com/), WebRTC OG and co-founder of [Daily](https://www.daily.co/).

# <center>
# <video controls autoplay muted>
# <source src="https://modal-cdn.com/example-webrtc_yolo.mp4" type="video/mp4">
# </video>
# </center>

# You can also try our deployment [here](https://modal-labs-examples--example-webrtc-yolo-webcamobjdet-web.modal.run).

# ## What is WebRTC?

# WebRTC (Web Real-Time Communication) is an [IETF Internet protocol](https://www.rfc-editor.org/rfc/rfc8825) and a [W3C API specification](https://www.w3.org/TR/webrtc/) for real-time media streaming between peers
# over internets or the World Wide Web.
# What makes it so effective and different from other bidirectional web-based communication protocols (e.g. WebSockets) is that it's purpose-built for media streaming in real time.
# It's primarily designed for browser applications using the JavaScript API, but [APIs exist for other languages](https://www.webrtc-developers.com/did-i-choose-the-right-webrtc-stack/).
# We'll build our app using Python's [`aiortc`](https://aiortc.readthedocs.io/en/latest/) package.

# ### What makes up a WebRTC application?

# A simple WebRTC app generally consists of three players:
# 1. a peer that initiates the connection,
# 2. a peer that responds to the connection, and
# 3. a server that passes some initial messages between the two peers.

# First, one peer initiates the connection by offering up a description of itself - its media sources, codec capabilities, Internet Protocol (IP) addressing info, etc - which is relayed to another peer through the server.
# The other peer then either accepts the offer by providing a compatible description of its own capabilities or rejects it if no compatible configuration is possible.
# This process is called "signaling" or sometimes the "negotiation" in the WebRTC world, and the server that mediates it is usually called the "signaling server".

# Once the peers have agreed on a configuration there's a brief pause to establish communication... and then you're live.

# ![Basic WebRTC architecture](https://modal-cdn.com/cdnbot/just_webrtc-1oic3iems_a4a8e77c.webp)
# <small>A basic WebRTC app architecture</small>

# Obviously there’s more going on under the hood.
# If you want to get into the details, we recommend checking out the [RFCs](https://www.rfc-editor.org/rfc/rfc8825) or a [more-thorough explainer](https://webrtcforthecurious.com/).
# In this document, we'll focus on how to architect a WebRTC application where one or more peer is running on Modal's serverless cloud infrastructure.

# If you just want to quickly get started with WebRTC for a small internal service or a hack project, check out
# [our FastRTC example](https://modal.com/docs/examples/fastrtc_flip_webcam) instead.

# ## How do I run a WebRTC app on Modal?

# Modal turns Python code into scalable cloud services.
# When you call a Modal Function, you get one replica.
# If you call it 999 more times before it returns, you have 1000 replicas.
# When your Functions all return, you spin down to 0 replicas.

# The core constraints of the Modal programming model that make this possible are that Function Calls are stateless and self-contained.
# In other words, correctly-written Modal Functions don't store information in memory between runs (though they might cache data to the ephemeral local disk for efficiency) and they don't create processes or tasks which must continue to run after the Function Call returns in order for the application to be correct.

# WebRTC apps, on the other hand, require passing messages back and forth in a multi-step protocol, and APIs spawn several "agents" (no, AI is not involved, just processes) which do work behind the scenes - including managing the peer-to-peer (P2P) connection itself.
# This means that streaming may have only just begun when the application logic in our Function has finished.

# ![Modal programming model and WebRTC signaling](https://modal-cdn.com/cdnbot/flow_comparisong6iibzq3_638bdd84.webp)
# <small>Modal's stateless programming model (left) and WebRTC's stateful signaling (right)</small>

# To ensure we properly leverage Modal's autoscaling and concurrency features, we need to align the signaling and streaming lifetimes with Modal Function Call lifetimes.

# The architecture we recommend for this appears below.

# ![WebRTC on Modal](https://modal-cdn.com/cdnbot/webrtc_with_modal-2horb680q_eab69b28.webp)
# <small>A clean architecture for WebRTC on Modal</small>

# It handles passing messages between the client peer and the signaling server using a
# [WebSocket](https://modal.com/docs/guide/webhooks#websockets) for persistent, bidirectional communication over the Web within a single Function Call.
# (Modal's Web layer maps HTTP and WS onto Function Calls, details [here](https://modal.com/blog/serverless-http)).
# We [`.spawn`](https://modal.com/docs/reference/modal.Function#spawn) the cloud peer inside the WebSocket endpoint
# and communicate it using a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue).

# We can then use the state of the P2P connection to determine when to return from the calls to both the signaling server and the cloud peer.
# When the P2P connection has been _established_, we'll close the WebSocket which in turn ends the call to the signaling server.
# And when the P2P connection has been _closed_, we'll return from the call to the cloud peer.
# That way, our WebRTC application benefits from all the autoscaling and concurrency logic built into Modal
# that enables users to deliver efficient cloud applications.

# We wrote two classes, `ModalWebRtcPeer` and `ModalWebRtcSignalingServer`, to abstract away that boilerplate as well as a lot of the `aiortc` implementation details.
# They're also decorated with Modal [lifetime hooks](https://modal.com/docs/guide/lifecycle-functions).
# Add the [`app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator and some custom logic, and you're ready to deploy on Modal.

# You can find them in the [`modal_webrtc.py` file](https://github.com/modal-labs/modal-examples/blob/main/07_web_endpoints/webrtc/modal_webrtc.py) provided alongside this example in the [GitHub repo](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/modal_webrtc.py).

# ## Using `modal_webrtc` to detect objects in webcam footage

# For our WebRTC app, we'll take a client's video stream, run a [YOLO](https://docs.ultralytics.com/tasks/detect/) object detector on it with an A100 GPU on Modal, and then stream the annotated video back to the client.
# With this setup, we can achieve inference times between 2-4 milliseconds per frame and RTTs below video frame rates (usually around 30 milliseconds per frame).

# Let's get started!

# ### Setup

# We'll start with a simple container [Image](https://modal.com/docs/guide/images) and then

# - set it up to properly use TensorRT and the ONNX Runtime, which keep latency minimal,
# - install the necessary libs for processing video, `opencv` and `ffmpeg`, and
# - install the necessary Python packages.

import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

video_processing_image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",  # use sed to uncomment
        "locale-gen en_US.UTF-8",  # set locale
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LD_LIBRARY_PATH": tensorrt_ld_path, "LANG": "en_US.UTF-8"})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg")
    # install Python dependencies
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

# ### Cache weights and compute graphs on a Volume

# We also need to create a Modal [Volume](https://modal.com/docs/guide/volumes) to store things we need across replicas --
# primarily the model weights and ONNX inference graph, but also a few other artifacts like a video file where
# we'll write out the processed video stream for testing. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

# The very first time we run the app, downloading the model and building the ONNX inference graph will take a few minutes.
# After that, we can load the cached weights and graph from the Volume, which reduces the startup time to about 15 seconds per container.

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("example-webrtc-yolo")

# ### Implement YOLO object detection as a `ModalWebRtcPeer`

# Our application needs to process an incoming video track with YOLO and return an annotated video track to the source peer.

# To implement a `ModalWebRtcPeer`, we need to:

# - Decorate our subclass with `@app.cls`. We provision it with an A100 GPU and a [Secret](https://modal.com/docs/guide/secrets) credential, described below.
# - Implement the method `setup_streams`. This is where we'll use `aiortc` to add the logic for processing the incoming video track with YOLO
# and returning an annotated video track to the source peer.

# `ModalWebRtcPeer` has a few other methods that users can optionally implement:

# - `initialize()`: This contains any custom initialization logic, called when `@modal.enter()` is called.
# - `run_streams()`: Logic for starting streams. This is necessary when the peer is the source of the stream.
# This is where you'd ensure a webcam was running, start playing a video file, or spin up a [video generative model](https://modal.com/docs/examples/image_to_video).
# - `get_turn_servers()`: We haven't talked about [TURN servers](https://datatracker.ietf.org/doc/html/rfc5766),
# but just know that they're necessary if you want to use WebRTC across complex (e.g. carrier-grade) NAT or firewall configurations.
# Free services have tight limits because TURN servers are expensive to run (lots of bandwidth and state management required).
# [STUN](https://datatracker.ietf.org/doc/html/rfc5389) servers, on the other hand, are essentially just echo servers, and so there are many free services available.
# If you don't provide TURN servers you can still serve your app on many networks using any of a number of free STUN servers for NAT traversal.
# - `exit()`: This contains any custom cleanup logic, called when `@modal.exit()` is called.

# In our case, we load the YOLO model in `initialize` and provide server information for the free [Open Relay TURN server](https://www.metered.ca/tools/openrelay/).
# If you want to use it, you'll need to create an account [here](https://dashboard.metered.ca/login?tool=turnserver)
# and then create a Modal [Secret](https://modal.com/docs/guide/secrets) called `turn-credentials` [here](https://modal.com/secrets).
# We also use the `@modal.concurrent` decorator to allow multiple instances of our peer to run on one GPU.

# **Setting the Region**

# Much of the latency in Internet applications comes from distance between communicating parties --
# the Internet operates within a factor of two of the speed of light, but that's just not that fast.
# To minimize latency under this constraint, the physical distance of the P2P connection
# between the webcam-using peer and the GPU container needs to be kept as short as possible.
# We'll use the `region` parameter of the `cls` decorator to set the region of the GPU container.
# You should set this to the closest region to your users.
# See the [region selection](https://modal.com/docs/guide/region-selection) guide for more information.


@app.cls(
    image=video_processing_image,
    gpu="A100-40GB",
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials")],
    region="us-east",  # set to your region
)
@modal.concurrent(
    target_inputs=2,  # try to stick to just two peers per GPU container
    max_inputs=3,  # but allow up to three
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

    async def get_turn_servers(self, peer_id=None, msg=None) -> dict:
        creds = {
            "username": os.environ["TURN_USERNAME"],
            "credential": os.environ["TURN_CREDENTIAL"],
        }

        turn_servers = [
            {"urls": "stun:stun.relay.metered.ca:80"},  # STUN is free, no creds neeeded
            # for TURN, sign up for the free service here: https://www.metered.ca/tools/openrelay/
            {"urls": "turn:standard.relay.metered.ca:80"} | creds,
            {"urls": "turn:standard.relay.metered.ca:80?transport=tcp"} | creds,
            {"urls": "turn:standard.relay.metered.ca:443"} | creds,
            {"urls": "turns:standard.relay.metered.ca:443?transport=tcp"} | creds,
        ]

        return {"type": "turn_servers", "ice_servers": turn_servers}


# ### Implement a `SignalingServer`

# The `ModalWebRtcSignalingServer` class is much simpler to implement.
# The main thing we need to do is implement the `get_modal_peer_class` method which will return our implementation of the `ModalWebRtcPeer` class, `ObjDet`.
#
# It also has an `initialize()` method we can optionally override (called at the beginning of the [container lifecycle](https://modal.com/docs/guide/lifecycle-functions))
# as well as a `web_app` property which will be [served by Modal](https://modal.com/docs/guide/webhooks#asgi-apps---fastapi-fasthtml-starlette).
# We'll use these to add a frontend which uses the WebRTC JavaScript API to stream a peer's webcam from the browser.
#
# The JavaScript and HTML files are alongside this example in the [Github repo](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/frontend).

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "shortuuid==1.0.13",
    )
)

this_directory = Path(__file__).parent.resolve()

server_image = base_image.add_local_dir(
    this_directory / "frontend", remote_path="/frontend"
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


# ## Addenda

# The remainder of this page is not central to running a WebRTC application on Modal,
# but is included for completeness.

# ### YOLO helper functions

# The two functions below are used to set up the YOLO model and create our custom [`MediaStreamTrack`](https://aiortc.readthedocs.io/en/latest/api.html#aiortc.MediaStreamTrack).

# The first, `get_yolo_model`, sets up the ONNXRuntime and loads the model weights.
# We call this in the `initialize` method of the `ModalWebRtcPeer` class
# so that it only happens once per container.


def get_yolo_model(cache_path):
    import onnxruntime

    from .yolo import YOLOv10

    onnxruntime.preload_dlls()
    return YOLOv10(cache_path)


# The second, `get_yolo_track`, creates a custom `MediaStreamTrack` that performs object detection on the video stream.
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

    return YOLOTrack(track, yolo_model)


# ### Testing a WebRTC application on Modal

# As any seasoned developer of real-time applications on the Web will tell you,
# testing and ensuring correctness is quite difficult. We spent nearly as much time
# designing and troubleshooting an appropriate testing process for this application as we did writing
# the application itself!

# You can find the testing code in the GitHub repository [here](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/webrtc_yolo_test.py).
