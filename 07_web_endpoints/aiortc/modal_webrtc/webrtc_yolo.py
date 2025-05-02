# standard python imports...
import os
from pathlib import Path

# ...and modal
import modal

from .modal_webrtc import ModalWebRTCPeer, ModalWebRTCServer

APP_NAME = "aiortc-server-video-processing-example"

# create an output volume to store the transmitted videos and model weights
CACHE_VOLUME = modal.Volume.from_name(
    "webrtc-yolo-cache", create_if_missing=True
)
CACHE_PATH = Path("/cache")

assets_parent_directory = Path(__file__).parent.parent.resolve()

# image
webrtc_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
    )
)

server_image = webrtc_base_image.add_local_dir(
    # frontend files
    os.path.join(assets_parent_directory, "frontend"),
    remote_path="/frontend",
)

tester_image = webrtc_base_image.add_local_dir(
    # video file for testing
    os.path.join(assets_parent_directory, "media"),
    remote_path="/media",
)

video_processing_gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",
        "locale-gen en_US.UTF-8",
        "update-locale LANG=en_US.UTF-8",
    )
    .apt_install("python3-opencv", "ffmpeg")
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/tensorrt_libs",
            "LANG": "en_US.UTF-8",
        }
    )
    .pip_install(
        "fastapi==0.115.12",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "tensorrt==10.9.0.34",
        "torch==2.7.0",
        "onnxruntime-gpu==1.21.0",
        "huggingface-hub==0.30.2",
    )
)

# instantiate our app
app = modal.App(APP_NAME)


# our modal peer, a subclass of ModalWebRTCPeer
# this class flips and incoming video stream
# and then streams the flipped video back to the provider
@app.cls(
    image=video_processing_gpu_image,
    secrets=[modal.Secret.from_dotenv()],
    gpu="A100-40GB",
    volumes={CACHE_PATH: CACHE_VOLUME},
)
class WebRTCVideoProcessor(ModalWebRTCPeer):
    yolo_model = None

    async def initialize(self) -> None:
        import onnxruntime

        from .yolo import YOLOv10

        onnxruntime.preload_dlls()
        self.yolo_model = YOLOv10(CACHE_PATH)

    async def setup_streams(self, peer_id: str):
        import numpy as np
        from aiortc import MediaStreamTrack
        from aiortc.contrib.media import VideoFrame

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
                print(f"YOLO Track initialized: {self.yolo_model}")

            def detection(self, image: np.ndarray) -> np.ndarray:
                import cv2

                orig_shape = image.shape[:-1]
                print(f"Image shape: {image.shape}")
                image = cv2.resize(
                    image,
                    (self.yolo_model.input_width, self.yolo_model.input_height),
                )
                print("Resized image shape: ", image.shape)
                image_w_detections = self.yolo_model.detect_objects(
                    image, self.conf_threshold
                )
                image_w_detections = cv2.resize(
                    image_w_detections, (orig_shape[1], orig_shape[0])
                )
                print(
                    "Resized image with detections shape: ",
                    image_w_detections.shape,
                )
                return image_w_detections

            # this is the essential method we need to implement
            # to create a custom MediaStreamTrack
            async def recv(self) -> VideoFrame:
                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")

                processed_img = self.detection(img)

                # VideoFrames are from a really nice package called av
                # which is a pythonic wrapper around ffmpeg
                # and a dep of aiortc
                new_frame = VideoFrame.from_ndarray(
                    processed_img, format="bgr24"
                )
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                return new_frame

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

            # create processed track
            flipped_track = YOLOTrack(track, self.yolo_model)
            self.pcs[peer_id].addTrack(flipped_track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended() -> None:
                print(
                    f"Video Processor, {self.id}, incoming video track from {peer_id} ended"
                )

    # some free turn servers we can up to 5 GB
    def get_turn_servers(self) -> dict:
        import os

        turn_servers = [
            {
                "urls": "stun:stun.relay.metered.ca:80",
            },
            {
                "urls": "turn:standard.relay.metered.ca:80",
                "username": os.environ["TURN_USERNAME"],
                "credential": os.environ["TURN_CREDENTIAL"],
            },
            {
                "urls": "turn:standard.relay.metered.ca:80?transport=tcp",
                "username": os.environ["TURN_USERNAME"],
                "credential": os.environ["TURN_CREDENTIAL"],
            },
            {
                "urls": "turn:standard.relay.metered.ca:443",
                "username": os.environ["TURN_USERNAME"],
                "credential": os.environ["TURN_CREDENTIAL"],
            },
            {
                "urls": "turns:standard.relay.metered.ca:443?transport=tcp",
                "username": os.environ["TURN_USERNAME"],
                "credential": os.environ["TURN_CREDENTIAL"],
            },
        ]

        return {
            "type": "turn_servers",
            "ice_servers": turn_servers,
        }


# for the server, all we have to do is
# let it know which ModalWebRTCPeer subclass to spawn
# attach our front end
@app.cls(
    image=server_image,
)
class WebRTCVideoProcessorServer(ModalWebRTCServer):
    from fastapi import FastAPI

    # lookup info for the WebRTCPeer to run on modal
    # modal_peer_app_name = APP_NAME
    # modal_peer_cls_name = "WebRTCVideoProcessor"
    modal_peer_cls = WebRTCVideoProcessor

    @modal.asgi_app(label="webrtc-video-processor-server")
    def web_endpoints(self) -> FastAPI:
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        # frontend files
        self.web_app.mount(
            "/static",
            StaticFiles(directory="/frontend"),
            name="static",
        )

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)

        return self.web_app


# create an output volume to store the transmitted videos


@app.cls(image=tester_image, volumes={CACHE_PATH: CACHE_VOLUME})
class WebRTCVideoProcessorTester(ModalWebRTCPeer):
    TEST_VIDEO_SOURCE_FILE = "/media/cliff_jumping.mp4"
    TEST_VIDEO_RECORD_FILE = CACHE_PATH / "flipped_test_video.mp4"
    # allowed difference between source and recorded video files
    DURATION_DIFFERENCE_THRESHOLD_FRAMES = 5
    # extra time to run streams beyond input video duration
    VIDEO_DURATION_BUFFER_SECS = 5.0

    async def initialize(self) -> None:
        import cv2

        self.input_filepath = self.TEST_VIDEO_SOURCE_FILE
        self.output_filepath = self.TEST_VIDEO_RECORD_FILE

        # get input video duration in seconds
        self.input_video = cv2.VideoCapture(self.input_filepath)
        self.input_video_duration = self.input_video.get(
            cv2.CAP_PROP_FRAME_COUNT
        ) / self.input_video.get(cv2.CAP_PROP_FPS)
        self.input_video.release()

        # set streaming duration to input video duration plus a buffer
        self.stream_duration = (
            self.input_video_duration + self.VIDEO_DURATION_BUFFER_SECS
        )

        self.player = None  # video stream source
        self.recorder = None  # processed video stream sink

    async def setup_streams(self, peer_id: str) -> None:
        from aiortc import MediaStreamTrack
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

    # confirm that the output video is (nearly) the same length as the input video
    # we lose a few frames at the beginning
    def confirm_recording(self) -> bool:
        import cv2

        # compare output video length to input video length
        input_video = cv2.VideoCapture(self.input_filepath)
        input_video_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video.release()

        output_video = cv2.VideoCapture(self.output_filepath)
        output_video_length = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video.release()

        if (
            input_video_length - output_video_length
        ) < self.DURATION_DIFFERENCE_THRESHOLD_FRAMES:
            return True
        else:
            return False

    @modal.method()
    async def run_video_processing_test(self) -> bool:
        import json

        import websockets

        peer_id = None
        # setup WebRTC connection using websockets
        ws_uri = (
            WebRTCVideoProcessorServer().web_endpoints.web_url.replace(
                "http", "ws"
            )
            + f"/ws/{self.id}"
        )
        async with websockets.connect(ws_uri) as websocket:
            await websocket.send(
                json.dumps({"type": "identify", "peer_id": self.id})
            )
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

        return self.confirm_recording()


# set timeout for health checks and connection test
MINUTES = 60  # seconds
TEST_TIMEOUT = 2.0 * MINUTES


# run tests
@app.local_entrypoint()
def main():
    assert WebRTCVideoProcessorTester().run_video_processing_test.remote(), (
        "Test failed to complete"
    )
