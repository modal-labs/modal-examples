# ---
# cmd: ["modal", "run", "-m", "07_web_endpoints.webrtc.webrtc_yolo_test"]
# ---


import modal

from .modal_webrtc import ModalWebRtcPeer
from .webrtc_yolo import (
    CACHE_PATH,
    WebcamObjDet,
    app,
    base_image,
    cache,
)

# ## Testing WebRTC and Modal

# First we define a `local_entrypoint` to run and evaluate the test.
# Our test will stream an .mp4 file to the cloud peer and record the annoated video to a new file.
# The test itself ensurse that the new video is no more than five frames shorter than the source file.
# The difference is due to dropped frames while the connection is starting up.


@app.local_entrypoint()
def test():
    input_frames, output_frames = TestPeer().run_video_processing_test.remote()
    # allow a few dropped frames from the connection starting up
    assert input_frames - output_frames < 5, (
        f"Streaming failed. Frame difference: {input_frames} - {output_frames} = {input_frames - output_frames}"
    )


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
    WS_OPEN_TIMEOUT = 300  # seconds

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
        import os

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
        import asyncio
        import json

        import websockets

        peer_id = None
        # connect to server via websocket
        ws_uri = (
            WebcamObjDet().web.get_web_url().replace("http", "ws") + f"/ws/{self.id}"
        )
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
        if self.pcs.get(peer_id) and self.pcs[peer_id].connectionState == "connected":
            await self.run_streams(peer_id)

            # wait for peer to finish processing video
            await asyncio.sleep(5.0)

        return self.count_frames()
