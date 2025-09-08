# # Streaming audio transcription using Parakeet

# This examples demonstrates the use of Parakeet ASR models for streaming speech-to-text on Modal.

# [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet)
# is the name of a family of ASR models built using [NVIDIA's NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).
# We'll show you how to use Parakeet for streaming audio transcription on Modal GPUs,
# with simple Python and browser clients.

# This example uses the `nvidia/parakeet-tdt-0.6b-v2` model which, as of June 2025, sits at the
# top of Hugging Face's [Open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# To try out transcription from your terminal,
# provide a URL for a `.wav` file to `modal run`:

# ```bash
# modal run 06_gpu_and_ml/speech-to-text/streaming_parakeet.py --audio-url="https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
# ```

# You should see output like the following:

# ```bash
# 🎤 Starting Transcription
# A Dream Within A Dream Edgar Allan Poe
# take this kiss upon the brow, And in parting from you now, Thus much let me avow You are not wrong who deem That my days have been a dream.
# ...
# ```

# Running a web service you can hit from any browser isn't any harder -- Modal handles the deployment of both the frontend and backend in a single App!
# Just run

# ```bash
# modal serve 06_gpu_and_ml/speech-to-text/streaming_parakeet.py
# ```

# and go to the link printed in your terminal.

# The full frontend code can be found [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/speech-to-text/streaming-parakeet-frontend/).

# ## Setup

import asyncio
import os
import sys
from pathlib import Path

import modal

app = modal.App("example-streaming-parakeet")

# ## Volume for caching model weights

# We use a [Modal Volume](https://modal.com/docs/guide/volumes) to cache the model weights.
# This allows us to avoid downloading the model weights every time we start a new instance.

# For more on storing models on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)

# ## Configuring dependencies

# The model runs remotely inside a container on Modal. We can define the environment
# and install our Python dependencies in that container's [`Image`](https://modal.com/docs/guide/images).

# For finicky setups like NeMO's, we recommend using the official NVIDIA CUDA Docker images from Docker Hub.
# You'll need to install Python and pip with the `add_python` option because the image
# doesn't have these by default.

# Additionally, we install `ffmpeg` for handling audio data and `fastapi` to create a web
# server for our WebSocket.

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
        "numpy<2",
        "pydub==0.25.1",
    )
    .entrypoint([])  # silence chatty logs by container on start
    .add_local_dir(  # changes fastest, so make this the last layer
        Path(__file__).parent / "streaming-parakeet-frontend",
        remote_path="/frontend",
    )
)

# ## Implementing streaming audio transcription on Modal

# Now we're ready to implement transcription. We wrap inference in a [`modal.Cls`](https://modal.com/docs/guide/lifecycle-functions) that
# ensures models are loaded and then moved to the GPU once when a new container starts.

# A couple of notes about this code:
# - The `transcribe` method takes bytes of audio data and returns the transcribed text.
# - The `web` method creates a FastAPI app using [`modal.asgi_app`](https://modal.com/docs/reference/modal.asgi_app#modalasgi_app) that serves a
# [WebSocket](https://modal.com/docs/guide/webhooks#websockets) endpoint for streaming audio transcription and a browser frontend for transcribing audio from your microphone.
# - The `run_with_queue` method takes a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue) and passes audio data and transcriptions between our local machine and the GPU container.

# Parakeet tries really hard to transcribe everything to English!
# Hence it tends to output utterances like "Yeah" or "Mm-hmm" when it runs on silent audio.
# We pre-process the incoming audio in the server using `pydub`'s silence detection,
# ensuring that we don't pass silence into our model.

END_OF_STREAM = (
    b"END_OF_STREAM_8f13d09"  # byte sequence indicating a stream is finished
)


@app.cls(volumes={"/cache": model_cache}, gpu="a10g", image=image)
@modal.concurrent(max_inputs=14, target_inputs=10)
class ParakeetModel:
    @modal.enter()
    def load(self):
        import logging

        import nemo.collections.asr as nemo_asr

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        import numpy as np

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        with NoStdStreams():  # hide output, see https://github.com/NVIDIA/NeMo/discussions/3281#discussioncomment-2251217
            output = self.model.transcribe([audio_data])

        return output[0].text

    @modal.method()
    async def handle_audio_chunk(
        self,
        chunk: bytes,
        audio_segment,
        silence_thresh=-45,  # dB
        min_silence_len=1000,  # ms
    ):
        from pydub import AudioSegment, silence

        new_audio_segment = AudioSegment(
            data=chunk,
            channels=1,
            sample_width=2,
            frame_rate=TARGET_SAMPLE_RATE,
        )

        # append the new audio segment to the existing audio segment
        audio_segment += new_audio_segment

        # detect windows of silence
        silent_windows = silence.detect_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
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
            text = self.transcribe(segment_to_transcribe.raw_data)
            return audio_segment, text
        except Exception as e:
            print("❌ Transcription error:", e)
            raise e

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

                audio_segment, text = await self.handle_audio_chunk.remote.aio(
                    chunk, audio_segment
                )
                if text:
                    await q.put.aio(text, partition="transcription")
        except Exception as e:
            print(f"Error handling queue: {type(e)}: {e}")
            return


@app.cls(image=image)
@modal.concurrent(max_inputs=1000)
class WebServer:
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

        # serve frontend
        @web_app.get("/")
        async def index():
            return HTMLResponse(content=open("/frontend/index.html").read())

        @web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            from fastapi import WebSocketDisconnect

            await ws.accept()

            from pydub import AudioSegment

            model = ParakeetModel()

            # initialize an empty audio segment
            audio_segment = AudioSegment.empty()

            try:
                while True:
                    # receive a chunk of audio data and convert it to an audio segment
                    chunk = await ws.receive_bytes()
                    if chunk == END_OF_STREAM:
                        await ws.send_bytes(END_OF_STREAM)
                        break
                    (
                        audio_segment,
                        text,
                    ) = await model.handle_audio_chunk.remote.aio(chunk, audio_segment)
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


# ## Running transcription from a local Python client

# Next, let's test the model with a [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint) that streams audio data to the server and prints
# out the transcriptions to our terminal as they arrive.

# Instead of using the WebSocket endpoint like the browser frontend,
# we'll use a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue)
# to pass audio data and transcriptions between our local machine and the GPU container.

AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
TARGET_SAMPLE_RATE = 16_000
CHUNK_SIZE = 16_000  # send one second of audio at a time


@app.local_entrypoint()
async def main(audio_url: str = AUDIO_URL):
    from urllib.request import urlopen

    print(f"🌐 Downloading audio file from {audio_url}")
    audio_bytes = urlopen(audio_url).read()
    print(f"🎧 Downloaded {len(audio_bytes)} bytes")

    audio_data = preprocess_audio(audio_bytes)

    print("🎤 Starting Transcription")
    with modal.Queue.ephemeral() as q:
        ParakeetModel().run_with_queue.spawn(q)
        send = asyncio.create_task(send_audio(q, audio_data))
        recv = asyncio.create_task(receive_text(q))
        await asyncio.gather(send, recv)
    print("✅ Transcription complete!")


# Below are the two functions that coordinate streaming audio and receiving transcriptions.

# `send_audio` transmits chunks of audio data with a slight delay,
# as though it was being streamed from a live source, like a microphone.
# `receive_text` waits for transcribed text to arrive and prints it.


async def send_audio(q, audio_bytes):
    for chunk in chunk_audio(audio_bytes, CHUNK_SIZE):
        await q.put.aio(chunk, partition="audio")
        await asyncio.sleep(CHUNK_SIZE / TARGET_SAMPLE_RATE / 8)
    await q.put.aio(END_OF_STREAM, partition="audio")


async def receive_text(q):
    while True:
        message = await q.get.aio(partition="transcription")
        if message == END_OF_STREAM:
            break

        print(message)


# ## Addenda

# The remainder of the code in this example is boilerplate,
# mostly for handling Parakeet's input format.


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


class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()
