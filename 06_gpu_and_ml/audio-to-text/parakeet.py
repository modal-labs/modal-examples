# # Real time audio transcription using Parakeet 🦜

# [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet) is the name of a family of ASR models built using [NVIDIA's NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).
# We'll show you how to use Parakeet for real-time audio transcription,
# with a simple python client and a websocket server you can spin up easily in Modal.

# This example uses the `nvidia/parakeet-tdt-0.6b-v2` model, which, as of May 13, 2025, sits at the
# top of Hugging Face's [ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# To run this example either:

# - run the browser/microphone frontend, or
# ```bash
# modal serve 06_gpu_and_ml/audio-to-text/parakeet.py
# ```
# - stream a .wav file from a URL (optional, default is "Dream Within a Dream" by Edgar Allan Poe).
# ```bash
# modal run 06_gpu_and_ml/audio-to-text/parakeet.py --audio-url="https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
# ```

# See [Troubleshooting](https://modal.com/docs/examples/parakeet#client) at the bottom if you run into issues.

# Here's what your final output might look like:

# ```bash
# 🌐 Downloading audio file...
# 🎧 Downloaded 6331478 bytes
# ☀️ Waking up model, this may take a few seconds on cold start...
# 📝 Transcription: A Dream Within A Dream Edgar Allan Poe
# 📝 Transcription:
# 📝 Transcription: take this kiss upon the brow, And in parting from you now, Thus much let me avow You are not wrong who deem That my days have been a dream.
# ...
# ```

# ## Setup
import asyncio
import os
from pathlib import Path

import modal

os.environ["MODAL_LOGLEVEL"] = "INFO"
app_name = "parakeet-websocket"

app = modal.App(app_name)
SILENCE_THRESHOLD = -45
SILENCE_MIN_LENGTH_MSEC = 1000
END_OF_STREAM = b"END_OF_STREAM"
# ## Volume for caching model weights
# We use a [Modal Volume](https://modal.com/docs/guide/volumes) to cache the model weights.
# This allows us to avoid downloading the model weights every time we start a new instance.


model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)
# ## Configuring dependencies
# The model runs remotely inside a [custom container](https://modal.com/docs/guide/custom-container). We can define the environment
# and install our Python dependencies in that container's `Image`.

# For inference, we recommend using the official NVIDIA CUDA Docker images from Docker Hub.
# You'll need to add Python 3 and pip with the `add_python` option because the image
# doesn't have these by default.

# Additionally, we install `ffmpeg` for handling audio data and `fastapi` to create a web
# server for our websocket.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy==1.26.4",  # downgrading numpy to avoid issues with CUDA
        "pydub==0.25.1",
    )
    .entrypoint([])
    .add_local_dir(
        os.path.join(Path(__file__).parent.resolve(), "frontend"),
        remote_path="/frontend",
    )
)


# ## Implementing real-time audio transcription on Modal

# Now, we're ready to implement the transcription model. We wrap inference in a [Modal Cls](https://modal.com/docs/guide/lifecycle-functions) that
# ensures models are loaded and then moved to the GPU once when a new container starts. Couple of notes:

# - The `load` method loads the model at start, instead of during inference, using [`modal.enter()`](https://modal.com/docs/reference/modal.enter#modalenter).
# - The `transcribe` method takes bytes of audio data, and returns the transcribed text.
# - The `web` method creates a FastAPI app using [`modal.asgi_app`](https://modal.com/docs/reference/modal.asgi_app#modalasgi_app) that serves a
# [websocket](https://modal.com/docs/guide/webhooks#websockets) endpoint for real-time audio transcription and a browser frontend for transcribing audio from your microphone.

# Parakeet tries really hard to transcribe everything to English!
# Hence it tends to output utterances like "Yeah" or "Mm-hmm" when it runs on silent audio.
# We can pre-process the incoming audio in the server by using `pydub`'s silence detection,
# ensuring that we only pass audio with text to our model.


@app.cls(volumes={"/cache": model_cache}, gpu="a10g", image=image)
@modal.concurrent(max_inputs=14, target_inputs=10)
class Parakeet:
    @modal.enter()
    def load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        import numpy as np

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        output = self.model.transcribe([audio_data])
        return output[0].text

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Response, WebSocket
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        web_app = FastAPI()
        web_app.mount("/static", StaticFiles(directory="/frontend"))

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        # server frontend
        @web_app.get("/")
        async def index():
            return HTMLResponse(content=open("/frontend/index.html").read())

        @web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            from fastapi import WebSocketDisconnect
            from pydub import AudioSegment

            await ws.accept()

            # initialize an empty audio segment
            audio_segment = AudioSegment.empty()

            try:
                while True:
                    # receive a chunk of audio data and convert it to an audio segment
                    chunk = await ws.receive_bytes()
                    if chunk == END_OF_STREAM:
                        await ws.send_bytes(END_OF_STREAM)
                        break
                    audio_segment, text = await self.handle_audio_chunk(
                        chunk, audio_segment
                    )
                    if text:
                        await ws.send_text(text)
            except Exception as e:
                if not isinstance(e, WebSocketDisconnect):
                    print(f"Error handling websocket: {type(e)}: {e}")
                try:
                    await ws.close(code=1011, reason="Internal server error")
                except Exception as e:
                    print(f"Error closing websocket: {type(e)}: {e}")

        return web_app

    @modal.method()
    async def run_with_queue(self, q: modal.Queue):
        from pydub import AudioSegment

        # initialize an empty audio segment
        audio_segment = AudioSegment.empty()

        try:
            while True:
                # receive a chunk of audio data and convert it to an audio segment
                chunk = await q.get.aio(partition="audio")

                if chunk == END_OF_STREAM:
                    await q.put.aio(END_OF_STREAM, partition="transcription")
                    break

                audio_segment, text = await self.handle_audio_chunk(
                    chunk, audio_segment
                )
                if text:
                    await q.put.aio(text, partition="transcription")
        except Exception as e:
            print(f"Error handling queue: {type(e)}: {e}")
            return

    async def handle_audio_chunk(self, chunk: bytes, audio_segment):
        from pydub import AudioSegment, silence

        new_audio_segment = AudioSegment(
            data=chunk,
            channels=1,
            sample_width=2,
            frame_rate=TARGET_SAMPLE_RATE,
        )
        # append the new audio segment to the existing audio segment
        audio_segment += new_audio_segment

        silent_windows = silence.detect_silence(
            audio_segment,
            min_silence_len=SILENCE_MIN_LENGTH_MSEC,
            silence_thresh=SILENCE_THRESHOLD,
        )

        # if there are no silent windows, continue
        if len(silent_windows) == 0:
            return audio_segment, None
        # get the last silent window because
        # we want to transcribe until the final pause
        last_window = silent_windows[-1]
        # if the entire audio segment is silent, reset the audio segment
        if last_window[0] == 0 and last_window[1] == len(audio_segment):
            audio_segment = AudioSegment.empty()
            return audio_segment, None
        # get the segment to transcribe: beginning until last pause
        segment_to_transcribe = audio_segment[: last_window[1]]
        # remove the segment to transcribe from the audio segment
        audio_segment = audio_segment[last_window[1] :]
        try:
            text = await self.transcribe(segment_to_transcribe.raw_data)
            return audio_segment, text
        except Exception as e:
            print("❌ Transcription error:", e)
            raise e


# ## Client
# Next, let's test the model with a `local_entrypoint` that streams audio data to the server and prints
# out the transcriptions in real-time to our terminal. We can also run this using Modal!

# Instead of using the WebSocket endpoint like the frontend,
# we'll use a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue)
# to pass audio data and transcriptions between our local machine and the GPU container.

AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 16000  # send one second of audio at a time


@app.local_entrypoint()
def main(audio_url: str = AUDIO_URL):
    from urllib.request import urlopen

    print("🌐 Downloading audio file...")
    audio_bytes = urlopen(audio_url).read()
    print(f"🎧 Downloaded {len(audio_bytes)} bytes")

    audio_data = preprocess_audio(audio_bytes)

    print("☀️ Waking up model, this may take a few seconds on cold start...")
    try:
        asyncio.run(run(audio_data))
        print("✅ Transcription complete!")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


# Below are the three main functions that coordinate streaming audio and receiving transcriptions.
#
# `send_audio` transmits chunks of audio data and then pauses to approximate streaming
# speech at a natural rate. That said, we set it to faster
# than real-time to compensate for network latency. Plus, we're not
# trying to wait forever for this to finish.


async def send_audio(q, audio_bytes):
    for chunk in chunk_audio(audio_bytes, CHUNK_SIZE):
        await q.put.aio(chunk, partition="audio")
        await asyncio.sleep(
            CHUNK_SIZE / TARGET_SAMPLE_RATE / 8
        )  # simulate real-time pacing
    await q.put.aio(END_OF_STREAM, partition="audio")
    await asyncio.sleep(5.00)


# `receive_transcriptions` is straightforward.
# It just waits for a transcription and prints it after a small delay to avoid colliding with the print statements
# from the GPU container.


async def receive_transcriptions(q):
    while True:
        message = await q.get.aio(partition="transcription")
        if message == END_OF_STREAM:
            break
        await asyncio.sleep(1.00)  # add a delay to avoid stdout collision
        print(f"📝 Transcription: {message}")


# We take full advantage of Modal's asynchronous capabilities here. In `run`, we spawn our function call
# so it doesn't block, and then we create and wait on the send and receive tasks.


async def run(audio_bytes):
    with modal.Queue.ephemeral() as q:
        Parakeet().run_with_queue.spawn(q)
        send_task = asyncio.create_task(send_audio(q, audio_bytes))
        receive_task = asyncio.create_task(receive_transcriptions(q))
        await asyncio.gather(send_task, receive_task)


# ## Troubleshooting
# - Make sure you have the latest version of the Modal CLI installed.
# - The server takes a few seconds to start up on cold start. If your local client times out, try
#   restarting the client.

# ## Addenda
# Helper functions for converting audio to Parakeet's input format and iterating over audio chunks.


def preprocess_audio(audio_bytes: bytes) -> bytes:
    import array
    import io
    import wave

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    # Convert frames to array based on sample width
    if sample_width == 1:
        audio_data = array.array("B", frames)  # unsigned char
    elif sample_width == 2:
        audio_data = array.array("h", frames)  # signed short
    elif sample_width == 4:
        audio_data = array.array("i", frames)  # signed int
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Downmix to mono if needed
    if n_channels > 1:
        mono_data = array.array(audio_data.typecode)
        for i in range(0, len(audio_data), n_channels):
            chunk = audio_data[i : i + n_channels]
            mono_data.append(sum(chunk) // n_channels)
        audio_data = mono_data

    # Resample to 16kHz if needed
    if frame_rate != TARGET_SAMPLE_RATE:
        ratio = TARGET_SAMPLE_RATE / frame_rate
        new_length = int(len(audio_data) * ratio)
        resampled_data = array.array(audio_data.typecode)

        for i in range(new_length):
            # Linear interpolation
            pos = i / ratio
            pos_int = int(pos)
            pos_frac = pos - pos_int

            if pos_int >= len(audio_data) - 1:
                sample = audio_data[-1]
            else:
                sample1 = audio_data[pos_int]
                sample2 = audio_data[pos_int + 1]
                sample = int(sample1 + (sample2 - sample1) * pos_frac)

            resampled_data.append(sample)

        audio_data = resampled_data

    return audio_data.tobytes()


def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]
