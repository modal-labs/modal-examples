# ---
# lambda-test: false
# cmd: ["modal", "serve", "06_gpu_and_ml/audio-to-text/parakeet.py::server_app"]
# ---
# # Real time audio transcription using Parakeet 🦜

# [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet) is the name of a family of ASR models from [NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) with a FastConformer Encoder and a CTC, RNN-T, or TDT decoder.
# We'll show you how to use Parakeet for real-time audio transcription,
# with a simple python client and a websocket server you can spin up easily in Modal.

# This example uses the `nvidia/parakeet-tdt-0.6b-v2` model, which, as of May 13, 2025, sits at the.
# top of Hugging Face's [ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# To run this example:

# 1. Start the server locally:
# ```bash
# modal serve 06_gpu_and_ml/audio-to-text/parakeet.py::server_app
# ```

# 2. In a separate terminal, run the client:
# ```bash
# modal run 06_gpu_and_ml/audio-to-text/parakeet.py::client_app --modal-profile=$(modal profile current)
# ```

# See troubleshooting section at the bottom if you run into issues.

# Here's what your final output might look like:

# ```
# ☀️ Waking up model, this may take a few seconds on cold start...

# Recording and streaming... Press Ctrl+C to stop.
# 📝 Transcription: Hi, how's it going?.
# 📝 Transcription: Doing well. Great day to be having fun with transcription models
# 📝 Transcription:
# 📝 Transcription: Good.
# ^C
# 🛑 Stopped by user.
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

# Additionally, we install `ffmpeg` for handling audio data and fastapi to create a web
# server for our websocket.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
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
    .run_commands(
        "uv pip install --system hf_transfer==0.1.9 huggingface_hub[hf-xet]==0.31.2 nemo_toolkit[asr]==2.3.0 cuda-python==12.9.0",
        "uv pip install --system 'numpy==1.26.4'",
        "uv pip install --system fastapi==0.115.12",
    )
)

with image.imports():
    import nemo.collections.asr as nemo_asr
    import numpy as np

# ## Implementing real-time audio transcription on Modal

# Now we're ready to implement the transcription model. We wrap inference in a Modal [Cls](https://modal.com/docs/guide/lifecycle-functions) that
# ensures models are loaded and then moved to the GPU once when a new container starts.

# The `load` method loads the model at start, instead of during inference, using [`modal.enter()`](https://modal.com/docs/reference/modal.enter#modalenter).
# The `transcribe` method takes a numpy array of audio data, and returns the transcribed text.
# The `web` method creates a FastAPI app using [`modal.asgi_app`](https://modal.com/docs/reference/modal.asgi_app#modalasgi_app) that serves a
# [websocket](https://modal.com/docs/guide/webhooks#websockets) endpoint for real-time audio transcription.

# Additionally, since this app is running in its own container, we make a `.remote` call to invoke
# the `transcribe` method. This allows us to use the GPU for inference.

class_name = "parakeet"


@server_app.cls(volumes={"/cache": model_cache}, gpu="a10g", image=image)
class Parakeet:
    import numpy as np

    @modal.enter()
    def load(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    @modal.method()
    def transcribe(self, audio_data: np.ndarray) -> str:
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
                        audio_np = np.frombuffer(bytes(buffer), dtype=np.int16)
                        buffer.clear()

                        try:
                            text = self.transcribe.remote(audio_np)
                            await ws.send_text(text)
                        except Exception as e:
                            print("❌ Transcription error:", e)
                            await ws.close(code=1011, reason="Internal server error")
            except WebSocketDisconnect:
                print("WebSocket disconnected")

        return web_app


# ## Client
# Let's test the model with a simple client that streams audio data to the server, and prints
# out the transcriptions in real-time to our terminal. We can also run this using modal!

## Image
# Using a secondary image for the client allows us to keep the dependencies separate from the server.
# We can keep  python's `websockets` library to create a websocket client
# that sends audio data to the server and receives transcriptions in real-time.


client_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("portaudio19-dev")
    .pip_install("websockets==15.0.1", "sounddevice==0.5.1", "numpy==2.2.5")
)
client_app = modal.App("parakeet-client", image=client_image)


WS_ENDPOINT = "/ws"

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DTYPE = "int16"  # must be string for deferred eval


@client_app.local_entrypoint()
def main(modal_profile: str):
    import asyncio

    import sounddevice as sd
    import websockets

    def make_audio_callback(audio_queue, loop):
        def callback(indata, frames, time, status):
            if status:
                print("Input stream status:", status)
            audio_chunk = indata.copy().astype(DTYPE).tobytes()
            asyncio.run_coroutine_threadsafe(audio_queue.put(audio_chunk), loop)

        return callback

    async def send_audio(websocket, audio_queue):
        loop = asyncio.get_running_loop()
        callback = make_audio_callback(audio_queue, loop)
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=DTYPE,
            callback=callback,
            blocksize=CHUNK_SIZE,
        ):
            print("🎙️ Recording and streaming... Press Ctrl+C to stop.")
            while True:
                audio_chunk = await audio_queue.get()
                await websocket.send(audio_chunk)

    async def receive_transcriptions(websocket):
        async for message in websocket:
            print("📝 Transcription:", message)

    async def run(ws_url):
        audio_queue = asyncio.Queue()
        async with websockets.connect(
            ws_url, open_timeout=240, ping_interval=None
        ) as websocket:
            send_task = asyncio.create_task(send_audio(websocket, audio_queue))
            receive_task = asyncio.create_task(receive_transcriptions(websocket))
            await asyncio.gather(send_task, receive_task)

    is_dev = True  # set to False if running modal deploy

    url = f"wss://{modal_profile}--{app_name}-{class_name}-web{'-dev' if is_dev else ''}.modal.run"
    ws_url = f"{url}{WS_ENDPOINT}"
    print(f"🌐 Using WebSocket URL: {ws_url}")
    print("☀️ Waking up model, this may take a few seconds on cold start...\n")
    try:
        asyncio.run(run(ws_url))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


# ## Troubleshooting
# - Make sure you have the latest version of the Modal CLI installed.
# - The server takes a few seconds to start up on cold start. If your client times out, try
#   restarting the client.
# - If you run into websocket URL errors, this may be because you have a non-environment
#   set. You can override the `url` variable in the client above, with the correct value. (the example uses `main`)
#   Similarly, if you're `modal deploy`ing the server, make sure to set the correct URL in the client.
