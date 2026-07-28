# ---
# cmd: ["modal", "run", "-m", "07_web.webrtc.webrtc_yolo_test"]
# ---

import asyncio
import os
import time

import modal

from .webrtc_yolo import (
    CACHE_PATH,
    WebcamObjDet,
    app,
    cache,
    lookup_turn_ice_servers,
)

# ## Testing WebRTC and Modal

# First we define a `local_entrypoint` to run and evaluate the test.
# Our test will stream an .mp4 file to the cloud peer and record the annotated video to a new file.
# The test itself ensures that the new video is no more than five frames shorter than the source file.
# The difference is due to dropped frames while the connection is starting up.


@app.local_entrypoint()
def test():
    input_frames, output_frames = run_video_processing_test.remote()
    # allow a few dropped frames from the connection starting up
    assert input_frames - output_frames < 5, (
        f"Streaming failed. Frame difference: {input_frames} - {output_frames} = {input_frames - output_frames}"
    )


# Because our test will require Python dependencies outside the standard library,
# we'll run the test itself in a container on Modal.
# There are some details in here regarding the use of `aiortc`'s `MediaPlayer` and `MediaRecorder` classes that we won't cover here.
# Just know that these are `aiortc` specific classes - not a WebRTC thing.

# That said, using these classes does require us to manually `start` and `stop` streams.
# We wait until the peer is `connected`, then start the recorder. We stop via the
# track `on_ended` callback and again explicitly after `pc.close()` so the mp4 is finalized.

test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .uv_pip_install(
        "aiortc==1.14.0",
        "aiohttp==3.11.18",
        "opencv-python==4.11.0.86",
    )
)

TEST_VIDEO_SOURCE_URL = "https://modal-cdn.com/cliff_jumping.mp4"
TEST_VIDEO_RECORD_FILE = CACHE_PATH / "test_video.mp4"
# extra time to run streams beyond input video duration
VIDEO_DURATION_BUFFER_SECS = 5.0
# allow time for container / YOLO cold start
TEST_TIMEOUT = 300
ICE_GATHERING_TIMEOUT_SECS = 10.0
# ICE/DTLS after setRemoteDescription
CONNECTION_TIMEOUT_SECS = 30.0


@app.function(
    image=test_image,
    volumes=cache,
    timeout=TEST_TIMEOUT,
)
async def run_video_processing_test() -> tuple[float, int]:
    import urllib.request

    import cv2
    from aiohttp import ClientSession
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

    # cache the source locally so MediaPlayer isn't held open on HTTP across the
    # GPU cold-start wait before ICE connects
    local_source = CACHE_PATH / "cliff_jumping_src.mp4"
    if not local_source.exists():
        urllib.request.urlretrieve(TEST_VIDEO_SOURCE_URL, local_source)

    # get input video duration in frames / seconds
    input_video = cv2.VideoCapture(str(local_source))
    input_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    input_fps = input_video.get(cv2.CAP_PROP_FPS) or 30.0
    input_duration = input_frames / input_fps
    input_video.release()

    if TEST_VIDEO_RECORD_FILE.exists():
        os.remove(TEST_VIDEO_RECORD_FILE)

    try:
        turn_servers = await lookup_turn_ice_servers.remote.aio()
    except Exception as e:
        print(f"Skipping TURN credential check (unavailable): {e}")
    else:
        turn_urls = [entry["urls"] for entry in turn_servers]
        if not any(str(url).startswith("turn") for url in turn_urls):
            raise RuntimeError(f"TURN ICE list missing turn: URLs: {turn_urls}")
        if not any(
            entry.get("username") and entry.get("credential") for entry in turn_servers
        ):
            raise RuntimeError("TURN ICE list missing username/credential")

    base_url = await WebcamObjDet().web.get_web_url.aio()
    offer_url = base_url.rstrip("/") + "/offer"
    stun_ice_url = base_url.rstrip("/") + "/ice-servers?mode=stun"

    async def _json_or_raise(resp, what: str):
        # aiohttp ClientResponseError is not cloudpickle-safe; raise a plain error.
        if resp.status >= 400:
            body = (await resp.text())[:500]
            raise RuntimeError(f"{what} failed: HTTP {resp.status}: {body}")
        return await resp.json()

    def _retriable_http(err: RuntimeError) -> bool:
        # CI can steal the shared ephemeral webhook label mid-request (5xx), or
        # leave the URL pointing at a stopped app (404).
        msg = str(err)
        return "HTTP 5" in msg or "HTTP 404" in msg

    async def _get_json(session, url, what, attempts=3):
        last_err = None
        for attempt in range(attempts):
            try:
                async with session.get(url) as resp:
                    return await _json_or_raise(resp, what)
            except RuntimeError as e:
                last_err = e
                if attempt + 1 == attempts or not _retriable_http(e):
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        raise last_err

    async with ClientSession() as session:
        # fetch STUN ICE servers from the signaling server (same list the GPU peer uses)
        ice_payload = await _get_json(
            session, stun_ice_url, "GET /ice-servers?mode=stun"
        )
        ice_servers = [
            RTCIceServer(
                urls=entry["urls"],
                username=entry.get("username"),
                credential=entry.get("credential"),
            )
            for entry in ice_payload["ice_servers"]
        ]

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        player = MediaPlayer(str(local_source))
        recorder = MediaRecorder(str(TEST_VIDEO_RECORD_FILE))
        blackhole = MediaBlackhole()
        try:
            # src file has audio; MediaPlayer demux stalls if that track isn't read.
            # drain it without sending; browser clients use video-only getUserMedia.
            if player.audio:
                blackhole.addTrack(player.audio)

            # audio before video keeps media m-lines in the order Pipecat expects
            pc.addTransceiver("audio")
            # client-created datachannel required by Pipecat's SmallWebRTCConnection
            pc.createDataChannel("modal-webrtc")
            # setup video player and add track to peer connection
            if player.video:
                pc.addTrack(player.video)

            # when we receive a track back from the video processing peer we record it
            @pc.on("track")
            def on_track(track):
                if track.kind != "video":
                    return
                # record track to file
                recorder.addTrack(track)

                @track.on("ended")
                async def on_ended():
                    # stop recording when incoming track ends to finish writing video
                    await recorder.stop()

            # set local description and send as offer to peer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # wait for ICE gathering; proceed with partial SDP if it times out
            deadline = time.monotonic() + ICE_GATHERING_TIMEOUT_SECS
            while pc.iceGatheringState != "complete":
                if time.monotonic() >= deadline:
                    print(
                        f"ICE gathering timed out after {ICE_GATHERING_TIMEOUT_SECS}s; "
                        "continuing with available candidates"
                    )
                    break
                await asyncio.sleep(0.05)

            last_err = None
            answer = None
            for attempt in range(3):
                try:
                    async with session.post(
                        offer_url,
                        json={
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                            "ice_server_type": "stun",
                        },
                    ) as resp:
                        answer = await _json_or_raise(resp, "POST /offer")
                    break
                except RuntimeError as e:
                    last_err = e
                    if attempt + 1 == 3 or not _retriable_http(e):
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
            else:
                raise last_err

            await blackhole.start()
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            deadline = time.monotonic() + CONNECTION_TIMEOUT_SECS
            while pc.connectionState not in ("connected", "failed", "closed"):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"timed out waiting for WebRTC connected; state={pc.connectionState}"
                    )
                await asyncio.sleep(0.05)
            if pc.connectionState != "connected":
                raise RuntimeError(f"peer connection {pc.connectionState}")

            # mediaRecorders need to be started manually
            await recorder.start()

            # run until sufficient time has passed
            await asyncio.sleep(input_duration + VIDEO_DURATION_BUFFER_SECS)
        finally:
            await pc.close()
            await blackhole.stop()
            # finalize the mp4 even if track "ended" never fires after close.
            await recorder.stop()
            if player.audio:
                player.audio.stop()
            if player.video:
                player.video.stop()

        # wait for peer to finish processing video
        await asyncio.sleep(5.0)

    # compare output video length to input video length
    output_video = cv2.VideoCapture(str(TEST_VIDEO_RECORD_FILE))
    output_frames = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video.release()
    return input_frames, output_frames
