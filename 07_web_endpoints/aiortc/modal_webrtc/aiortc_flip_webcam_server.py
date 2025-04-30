# standard python imports...
from pathlib import Path
import os

# ...and modal
import modal

from .webrtc import ModalWebRTCPeer, ModalWebRTCServer

assets_parent_directory = Path(__file__).parent.parent.resolve()

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
        os.path.join(assets_parent_directory, "media"), 
        remote_path="/media"
    )
    # frontend files
    .add_local_dir(
        os.path.join(assets_parent_directory, "frontend"), 
        remote_path="/frontend"
    )
)

# instantiate our app
app = modal.App(
    "aiortc-server-video-processing-example"
)

@app.cls(
    image=web_image,
)
class WebRTCVideoProcessorServer(ModalWebRTCServer):

    modal_peer_app_name = "aiortc-server-video-processing-example"
    modal_peer_cls_name = "WebRTCVideoProcessor"

    @modal.asgi_app(label="webrtc-video-processor-server")
    def web_endpoints(self):

        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import HTMLResponse
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

# this class responds to an offer
# to establish a P2P connection,
# flips the video stream, and then
# streams the flipped video back to the provider
# and/or records the flipped video to a file
@app.cls(
    image=web_image,
)
class WebRTCVideoProcessor(ModalWebRTCPeer):   

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
            
            print(f"Video Processor, {self.id}, received {track.kind} track from {peer_id}")
            
            # create processed track
            flipped_track = VideoFlipTrack(track)
            self.pcs[peer_id].addTrack(flipped_track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended():
                print(f"Video Processor, {self.id}, incoming video track from {peer_id} ended")

# create an output volume to store the transmitted videos
output_volume = modal.Volume.from_name("aiortc-video-processing", create_if_missing=True)
OUTPUT_VOLUME_PATH = Path("/output")

@app.cls(
    image=web_image,
    volumes={
        OUTPUT_VOLUME_PATH: output_volume
    }
)
class WebRTCVideoProcessorTester(ModalWebRTCPeer):
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
        ws_uri = WebRTCVideoProcessorServer().web_endpoints.web_url.replace("http", "ws") + f"/ws/{self.id}"
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

    import urllib
    import time

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

    import urllib
    import json
    import time

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

    