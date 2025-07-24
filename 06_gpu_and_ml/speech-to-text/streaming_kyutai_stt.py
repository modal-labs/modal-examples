# ---
# mypy: ignore-errors
# deploy: true
# ---

# # Stream transcriptions with Kyutai STT

# This example demonstrates the deployment of a streaming audio transcription service with Kyutai STT on Modal.

# [Kyutai STT](https://kyutai.org/next/stt) is an automated speech recognition/transcription model
# that is designed to operate on streams of audio, rather than on complete audio files.
# See the linked blog post for details on their "delayed streams" architecture.

# ## Setup

# We start by importing some basic packages and the Modal SDK.

import asyncio
import base64
import time
from pathlib import Path

import modal

# Then we define a Modal App and an
# [Image](https://modal.com/docs/guide/images)
# with the dependencies of our speech-to-text system.

app = modal.App(name="example-streaming-kyutai-stt")

stt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "moshi==0.2.9", "fastapi==0.116.1", "hf_transfer==0.1.9", "julius==0.2.7"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# One dependency is missing: the model weights.

# Instead of including them in the Image or loading them every time the Function starts,
# we add them to a Modal [Volume](https://modal.com/docs/guide/volumes).
# Volumes are like a shared disk that all Modal Functions can access.

# For more details on patterns for handling model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

MODEL_NAME = "kyutai/stt-1b-en_fr"

hf_cache_vol = modal.Volume.from_name(f"{app.name}-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")
volumes = {hf_cache_vol_path: hf_cache_vol}

# ## Run Kyutai STT inference on Modal

# Now we're ready to add the code that runs the speech-to-text model.

# We use a Modal [Cls](https://modal.com/docs/guide/lifecycle-functions)
# so that we can separate out the model loading and setup code from the inference.

# For more on lifecycle management with Clses and cold start penalty reduction on Modal, see
# [this guide](https://modal.com/docs/guide/cold-start).

# We also define multiple ways to access the underlying streaming STT service --
# via a [WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API),
# for Web clients like browsers,
# and via a Modal [Queue](https://modal.com/docs/guide/queues)
# for Python clients.

# That plus the code for manipulating the streams of audio bytes and output text
# leads to a pretty big class! But there's not anything too complex here.

MINUTES = 60


@app.cls(image=stt_image, gpu="l40s", volumes=volumes, timeout=10 * MINUTES)
class STT:
    BATCH_SIZE = 1

    @modal.enter()
    def enter(self):
        import torch
        from huggingface_hub import snapshot_download
        from moshi.models import LMGen, loaders

        start_time = time.monotonic_ns()

        print("Loading model...")
        snapshot_download(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        self.moshi = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(self.moshi, temp=0, temp_text=0)

        self.mimi.streaming_forever(self.BATCH_SIZE)
        self.lm_gen.streaming_forever(self.BATCH_SIZE)

        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        self.audio_silence_prefix_seconds = checkpoint_info.stt_config.get(
            "audio_silence_prefix_seconds", 1.0
        )
        self.audio_delay_seconds = checkpoint_info.stt_config.get(
            "audio_delay_seconds", 5.0
        )
        self.padding_token_id = checkpoint_info.raw_config.get(
            "text_padding_token_id", 3
        )

        # warmup gpus
        for _ in range(4):
            codes = self.mimi.encode(
                torch.zeros(self.BATCH_SIZE, 1, self.frame_size).to(self.device)
            )
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
        torch.cuda.synchronize()

        print(f"Model loaded in {round((time.monotonic_ns() - start_time) / 1e9, 2)}s")

    def reset_state(self):
        # reset llm chat history for this input
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    async def transcribe(self, pcm, all_pcm_data):
        import numpy as np
        import torch

        if pcm is None:
            yield all_pcm_data
            return
        if len(pcm) == 0:
            yield all_pcm_data
            return

        if pcm.shape[-1] == 0:
            yield all_pcm_data
            return

        if all_pcm_data is None:
            all_pcm_data = pcm
        else:
            all_pcm_data = np.concatenate((all_pcm_data, pcm))

        # infer on each frame
        while all_pcm_data.shape[-1] >= self.frame_size:
            chunk = all_pcm_data[: self.frame_size]
            all_pcm_data = all_pcm_data[self.frame_size :]

            with torch.no_grad():
                chunk = torch.from_numpy(chunk)
                chunk = chunk.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_size)
                chunk = chunk.expand(
                    self.BATCH_SIZE, -1, -1
                )  # (batch_size, 1, frame_size)
                chunk = chunk.to(device=self.device)

                # inference on audio chunk
                codes = self.mimi.encode(chunk)

                # language model inference against encoded audio
                for c in range(codes.shape[-1]):
                    text_tokens, vad_heads = self.lm_gen.step_with_extra_heads(
                        codes[:, :, c : c + 1]
                    )
                    if text_tokens is None:
                        # model is silent
                        yield all_pcm_data
                        return
                    if vad_heads:
                        pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                        if pr_vad > 0.5:
                            # end of turn detected
                            yield all_pcm_data
                            return

                    assert text_tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

                    text_token = text_tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        text = self.text_tokenizer.id_to_piece(text_token)
                        text = text.replace("▁", " ")
                        yield text

        yield all_pcm_data

    @modal.asgi_app()
    def api(self):
        import sphn
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def transcribe_websocket(ws: WebSocket):
            await ws.accept()

            opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)
            transcription_queue = asyncio.Queue()

            print("Session started")
            tasks = []

            # asyncio to run multiple loops concurrently within single websocket connection
            async def recv_loop():
                """
                Receives Opus stream across websocket, appends into inbound queue.
                """
                nonlocal opus_stream_inbound
                while True:
                    data = await ws.receive_bytes()

                    if not isinstance(data, bytes):
                        print("received non-bytes message")
                        continue
                    if len(data) == 0:
                        print("received empty message")
                        continue
                    opus_stream_inbound.append_bytes(data)

            async def inference_loop():
                """
                Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                """
                nonlocal opus_stream_inbound, transcription_queue
                all_pcm_data = None

                while True:
                    await asyncio.sleep(0.001)

                    pcm = opus_stream_inbound.read_pcm()
                    async for msg in self.transcribe(pcm, all_pcm_data):
                        if isinstance(msg, str):
                            transcription_queue.put_nowait(msg)
                        else:
                            all_pcm_data = msg

            async def send_loop():
                """
                Reads outbound data, and sends it across websocket
                """
                nonlocal transcription_queue
                while True:
                    data = await transcription_queue.get()

                    if data is None:
                        continue

                    msg = b"\x01" + bytes(
                        data, encoding="utf8"
                    )  # prepend "\x01" as a tag to indicate text
                    await ws.send_bytes(msg)

            # run all loops concurrently
            try:
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(inference_loop()),
                    asyncio.create_task(send_loop()),
                ]
                await asyncio.gather(*tasks)

            except WebSocketDisconnect:
                print("WebSocket disconnected")
                await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)  # internal error
                raise e
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.reset_state()

        return web_app

    @modal.method()
    async def transcribe_queue(self, q: modal.Queue):
        import tempfile

        import sphn

        all_pcm_data = None

        while True:
            chunk = await q.get.aio(partition="audio")
            if chunk is None:
                await q.put.aio(None, partition="transcription")
                break

            # to avoid having to encode the audio and retrieve with OpusStreamReader:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(chunk)
                tmp.flush()
                pcm, _ = sphn.read(tmp.name)
                pcm = pcm.squeeze(0)

            async for msg in self.transcribe(pcm, all_pcm_data):
                if isinstance(msg, str):
                    await q.put.aio(msg, partition="transcription")
                else:
                    all_pcm_data = msg


# ## Run a local Python client to test streaming STT

# We can test this code on the same production Modal infra
# that we'll be deploying it on by writing a quick `local_entrypoint` for testing.

# We just need a few helper functions to control the streaming of audio bytes
# and transcribed text from local Python.

# These communicate asynchronously with the deployed Function using a Modal Queue.


async def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


async def send_audio(audio_bytes: bytes, q: modal.Queue, chunk_size: int, rtf: int):
    async for chunk in chunk_audio(audio_bytes, chunk_size):
        await q.put.aio(chunk, partition="audio")
        await asyncio.sleep(chunk_size / chunk_size / rtf)
    await q.put.aio(None, partition="audio")


async def receive_text(q: modal.Queue):
    break_counter, break_every = 0, 20
    while True:
        data = await q.get.aio(partition="transcription")
        if data is None:
            break
        print(data, end="")
        break_counter += 1
        if break_counter >= break_every:
            print()
            break_counter = 0


# Now we write our quick test, which loads in audio from a URL
# and then passes it to the remote Function via a

# If you run this example with

# ```bash
# modal run streaming_kyutai_stt.py
# ```

# you will

# 1. deploy the latest version of the code on Modal
# 2. spin up a new GPU to handle transcription
# 3. load the model from Hugging Face or the Modal Volume cache
# 4. send the audio out to the new GPU container, transcribe it, and receive it locally to be printed.

# Not bad for a single Python file with no dependencies except Modal!


@app.local_entrypoint()
async def test(
    chunk_size: int = 24_000,  # bytes
    rtf: int = 1000,
    audio_url: str = "https://github.com/kyutai-labs/delayed-streams-modeling/raw/refs/heads/main/audio/bria.mp3",
):
    from urllib.request import urlopen

    print(f"Downloading audio file from {audio_url}")
    audio_bytes = urlopen(audio_url).read()
    print(f"Downloaded {len(audio_bytes)} bytes")

    print("Starting transcription")
    start_time = time.monotonic_ns()
    with modal.Queue.ephemeral() as q:
        STT().transcribe_queue.spawn(q)
        send = asyncio.create_task(send_audio(audio_bytes, q, chunk_size, rtf))
        recv = asyncio.create_task(receive_text(q))
        await asyncio.gather(send, recv)
    print(
        f"\nTranscription complete in {round((time.monotonic_ns() - start_time) / 1e9, 2)}s"
    )


# ## Deploy a streaming STT service on the Web

# We've already written a Web backend for our streaming STT service --
# that's the FastAPI API with the WebSocket in the Modal Cls above.

# We can also deploy a Web frontend. To keep things almost entirely "pure Python",
# we here use the [FastHTML](https://www.fastht.ml/) library,
# but you can also deploy a JavaScript frontend with a FastAPI or Node backend.

# We do use a bit of JS for the audio processing in the browser.
# We add it to the Modal Image using `add_local_dir`.
# You can find the frontend files [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/speech-to-text/streaming-kyutai-stt-frontend).

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("python-fasthtml==0.12.20")
    .add_local_dir(
        Path(__file__).parent / "streaming-kyutai-stt-frontend", "/root/frontend"
    )
)

# You can deploy this frontend with

# ```bash
# modal deploy streaming_kyutai_stt.py
# ```

# and then interact with it at the printed `ui` URL.


@app.function(image=web_image, timeout=10 * MINUTES)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def ui():
    import fasthtml.common as fh

    modal_logo_svg = open("/root/frontend/modal-logo.svg").read()
    modal_logo_base64 = base64.b64encode(modal_logo_svg.encode()).decode()
    app_js = open("/root/frontend/audio.js").read()

    fast_app, rt = fh.fast_app(
        hdrs=[
            # audio recording libraries
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/recorder.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/ogg-opus-decoder/dist/ogg-opus-decoder.min.js"
            ),
            # styling
            fh.Link(
                href="https://fonts.googleapis.com/css?family=Inter:300,400,600",
                rel="stylesheet",
            ),
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.Script("""
                tailwind.config = {
                    theme: {
                        extend: {
                            colors: {
                                ground: "#0C0F0B",
                                primary: "#9AEE86",
                                "accent-pink": "#FC9CC6",
                                "accent-blue": "#B8E4FF",
                            },
                        },
                    },
                };
            """),
        ],
    )

    @rt("/")
    def get():
        return (
            fh.Title("Kyutai Streaming STT"),
            fh.Body(
                fh.Div(
                    fh.Div(
                        fh.Div(
                            id="text-output",
                            cls="flex flex-col-reverse overflow-y-auto max-h-64 pr-2",
                        ),
                        cls="w-full overflow-y-auto max-h-64",
                    ),
                    cls="bg-gray-800 rounded-lg shadow-lg w-full max-w-xl p-6",
                ),
                fh.Footer(
                    fh.Span(
                        "Built with ",
                        fh.A(
                            "Kyutai",
                            href="https://github.com/kyutai-labs/delayed-streams-modeling",
                            target="_blank",
                            rel="noopener noreferrer",
                            cls="underline",
                        ),
                        " and",
                        cls="text-sm font-medium text-gray-300 mr-2",
                    ),
                    fh.A(
                        fh.Img(
                            src=f"data:image/svg+xml;base64,{modal_logo_base64}",
                            alt="Modal logo",
                            cls="w-24",
                        ),
                        cls="flex items-center p-2 rounded-lg bg-gray-800 shadow-lg hover:bg-gray-700 transition-colors duration-200",
                        href="https://modal.com",
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    cls="fixed bottom-4 inline-flex items-center justify-center",
                ),
                fh.Script(app_js),
                cls="relative bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-4",
            ),
        )

    return fast_app
