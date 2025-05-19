# ---
# lambda-test: false
# ---
# # Real time audio transcription using Parakeet 🦜

# [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet) is the name of a family of ASR models built using [NVIDIA's NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).
# We'll show you how to use Parakeet for real-time audio transcription,
# with a simple python client and a websocket server you can spin up easily in Modal.

# This example uses the `nvidia/parakeet-tdt-0.6b-v2` model, which, as of May 13, 2025, sits at the
# top of Hugging Face's [ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# To run this example:

# 1. Start the server locally:
# ```bash
# modal serve 06_gpu_and_ml/audio-to-text/parakeet.py::server_app
# ```

# 2. In a separate terminal, run the client:
# ```bash
# pip install requests websockets numpy
# modal run 06_gpu_and_ml/audio-to-text/parakeet.py::client_app --modal-profile=$(modal profile current)
# ```

# See [Troubleshooting](https://modal.com/docs/examples/parakeet#client) at the bottom if you run into issues.

# Here's what your final output might look like:

# ```

# 🌐 Downloading audio file...
# 🎧 Downloaded 6331478 bytes
# 🔗 Streaming data to WebSocket:
# wss://modal-labs--parakeet-websocket-parakeet-web-dev.modal.run/ws
# ☀️ Waking up model, this may take a few seconds on cold start...
# 📝 Transcription: A Dream Within a Dream
# 📝 Transcription: Edgar Allan Poe.
# 📝 Transcription:
# 📝 Transcription:
# 📝 Transcription: Take this kiss upon the brow.
# ```

# ## Setup
import modal

app_name = "parakeet-websocket"

server_app = modal.App(app_name)
BUFFER_SIZE = 8000

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
    )
)

with image.imports():
    import nemo.collections.asr as nemo_asr
# ## Implementing real-time audio transcription on Modal

# Now, we're ready to implement the transcription model. We wrap inference in a [Modal Cls](https://modal.com/docs/guide/lifecycle-functions) that
# ensures models are loaded and then moved to the GPU once when a new container starts. Couple of notes:

# - The `load` method loads the model at start, instead of during inference, using [`modal.enter()`](https://modal.com/docs/reference/modal.enter#modalenter).
# - The `transcribe` method takes a numpy array of audio data, and returns the transcribed text.
# - The `web` method creates a FastAPI app using [`modal.asgi_app`](https://modal.com/docs/reference/modal.asgi_app#modalasgi_app) that serves a
# [websocket](https://modal.com/docs/guide/webhooks#websockets) endpoint for real-time audio transcription.

# Additionally, since this app is running in its own container, we make a `.remote` call to invoke
# the `transcribe` method. This allows us to use the GPU for inference.

class_name = "parakeet"


@server_app.cls(volumes={"/cache": model_cache}, gpu="a10g", image=image)
class Parakeet:
    @modal.enter()
    def load(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import numpy as np

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        output = self.model.transcribe([audio_data])
        return output[0].text

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            buffer = bytearray()

            try:
                while True:
                    chunk = await ws.receive_bytes()
                    buffer.extend(chunk)

                    if len(buffer) > BUFFER_SIZE:
                        audio_bytes = bytes(buffer)
                        buffer.clear()

                        try:
                            text = self.transcribe.remote(audio_bytes)
                            await ws.send_text(text)
                        except Exception as e:
                            print("❌ Transcription error:", e)
                            await ws.close(code=1011, reason="Internal server error")
            except WebSocketDisconnect:
                print("WebSocket disconnected")

        return web_app


# ## Client
# Next, let's test the model with a simple client that streams audio data to the server, and prints
# out the transcriptions in real-time to our terminal. We can also run this using Modal!

# ## Image
# Using a secondary image for the client allows us to keep the client dependencies separate from the server dependencies.
# We'll use  python's `websockets` library to create a websocket client
# that sends audio data to the server and receives transcriptions in real-time.


client_image = modal.Image.debian_slim(python_version="3.12")
client_app = modal.App("parakeet-client", image=client_image)


WS_ENDPOINT = "/ws"

AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"


@client_app.local_entrypoint()
def main(modal_profile: str, audio_url: str = AUDIO_URL):
    import asyncio

    import requests

    CHUNK_SIZE = 64000
    is_dev = True
    url = f"wss://{modal_profile}--{app_name}-{class_name}-web{'-dev' if is_dev else ''}.modal.run"
    ws_url = f"{url}{WS_ENDPOINT}"

    def convert_to_mono_16khz(audio_bytes: bytes) -> bytes:
        import io
        import wave

        import numpy as np

        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
            n_channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            frame_rate = wav_in.getframerate()
            n_frames = wav_in.getnframes()
            frames = wav_in.readframes(n_frames)

        # Determine dtype from sample width
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert frames to NumPy array
        audio_data = np.frombuffer(frames, dtype=dtype)

        # Downmix to mono if needed
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)
            audio_data = audio_data.mean(axis=1).astype(dtype)

        # Resample to 16kHz if needed
        if frame_rate != 16000:
            ratio = 16000 / frame_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(
                indices, np.arange(len(audio_data)), audio_data
            ).astype(dtype)

        return audio_data.tobytes()

    def chunk_audio(data: bytes, chunk_size: int):
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async def send_audio(ws, audio_bytes):
        for chunk in chunk_audio(audio_bytes, CHUNK_SIZE):
            await ws.send(chunk)
            await asyncio.sleep(0.01)  # simulate real-time pacing

    async def receive_transcriptions(ws):
        async for message in ws:
            print("📝 Transcription:", message)

    async def run(ws_url, audio_bytes):
        import websockets

        async with websockets.connect(
            ws_url, ping_interval=None, open_timeout=240
        ) as ws:
            send_task = asyncio.create_task(send_audio(ws, audio_bytes))
            receive_task = asyncio.create_task(receive_transcriptions(ws))
            await asyncio.gather(send_task, receive_task)

    print("🌐 Downloading audio file...")
    response = requests.get(audio_url)
    response.raise_for_status()
    audio_bytes = response.content
    print(f"🎧 Downloaded {len(audio_bytes)} bytes")

    audio_data = convert_to_mono_16khz(audio_bytes)

    print(f"🔗 Streaming data to WebSocket: {ws_url}")
    print("☀️ Waking up model, this may take a few seconds on cold start...")
    try:
        asyncio.run(run(ws_url, audio_data))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


# ## Troubleshooting
# - Make sure you have the latest version of the Modal CLI installed.
# - The server takes a few seconds to start up on cold start. If your client times out, try
#   restarting the client. If it continues to time out, try bumping the value of `open_timeout`.
# - If you run into websocket URL errors, this may be because you have a non-default environment
#   configured. You can override the `url` variable in the client above, with the correct value.
#   Similarly, if you're `modal deploy`ing the server, make sure to set the correct URL in the client.
