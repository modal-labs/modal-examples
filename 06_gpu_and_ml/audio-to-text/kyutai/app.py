import asyncio
import base64
import time
from pathlib import Path

import modal

app = modal.App(name="kyutai")

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode python-fasthtml==0.12.20",
    )
    .add_local_dir(Path(__file__).parent / "frontend", "/root/frontend")
)

stt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode moshi==0.2.6 fastapi==0.115.14 hf_transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "kyutai/stt-1b-en_fr"
MODEL_REVISION = "40b03403247f4adc9b664bc1cbdff78a82d31085"

hf_cache_vol = modal.Volume.from_name(f"{app.name}-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")
volumes = {hf_cache_vol_path: hf_cache_vol}


@app.function(
    scaledown_window=600,
    timeout=600,
    image=web_image,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import fasthtml.common as fh

    modal_logo_svg = open("/root/frontend/modal-logo.svg").read()
    modal_logo_base64 = base64.b64encode(modal_logo_svg.encode()).decode()
    app_js = open("/root/frontend/audio.js").read()

    fast_app, rt = fh.fast_app(
        hdrs=[
            # Audio recording libraries
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/recorder.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/ogg-opus-decoder/dist/ogg-opus-decoder.min.js"
            ),
            # Styling
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
            fh.Title(
                "Kyutai STT",
            ),
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
                            "Kyutai STT",
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


MINUTES = 60


@app.cls(
    image=stt_image,
    gpu="l40s:1",
    volumes=volumes,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=64)
class STT:
    @modal.enter()
    def enter(self):
        import torch
        from huggingface_hub import snapshot_download
        from moshi.models import LMGen, loaders

        start_time = time.monotonic_ns()

        print("Downloading model if necessary...")
        snapshot_download(MODEL_NAME, revision=MODEL_REVISION)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        self.moshi = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(
            self.moshi,
            # sampling params
            temp=0,
            temp_text=0,
        )

        self.batch_size = 1
        self.mimi.streaming_forever(self.batch_size)
        self.lm_gen.streaming_forever(self.batch_size)

        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        # warmup gpus
        for chunk in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
        torch.cuda.synchronize()

        print(f"Model loaded in {round((time.monotonic_ns() - start_time) / 1e9, 2)}s")

    def reset_state(self):
        import sphn

        # use opus format for audio across websocket to safely stream/decode in real-time
        self.opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)

        # queue for out-bound transcription
        self.transcription_queue = asyncio.Queue()

        # reset llm chat history on each connection
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @modal.asgi_app()
    def web(self):
        import numpy as np
        import torch
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()

            # clear llm chat history + buffered audio
            self.reset_state()

            print("Session started")
            tasks = []

            # shared state between loops
            is_first_token = True
            stream_start_time = None
            last_audio_receive_time = None
            audio_receive_lock = asyncio.Lock()

            # asyncio to run multiple loops concurrently within single websocket connection
            async def recv_loop():
                """
                Receives Opus stream across websocket, appends into opus_stream_inbound.
                """
                nonlocal stream_start_time
                nonlocal last_audio_receive_time

                while True:
                    data = await ws.receive_bytes()

                    if not isinstance(data, bytes):
                        print("received non-bytes message")
                        continue
                    if len(data) == 0:
                        print("received empty message")
                        continue

                    if stream_start_time is None:
                        stream_start_time = time.monotonic_ns()

                    # track when we received audio
                    async with audio_receive_lock:
                        last_audio_receive_time = time.monotonic_ns()

                    self.opus_stream_inbound.append_bytes(data)

            async def inference_loop():
                """
                Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                """
                nonlocal is_first_token
                nonlocal stream_start_time
                nonlocal last_audio_receive_time
                all_pcm_data = None

                while True:
                    await asyncio.sleep(0.001)
                    pcm = self.opus_stream_inbound.read_pcm()
                    if pcm is None:
                        continue
                    if len(pcm) == 0:
                        continue

                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))

                    # infer on each frame
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        # get the audio receive time for this frame
                        async with audio_receive_lock:
                            frame_audio_time = last_audio_receive_time

                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size :]

                        with torch.no_grad():
                            chunk = torch.from_numpy(chunk)
                            chunk = chunk.to(device=self.device)[None, None]

                            # inference on audio chunk
                            codes = self.mimi.encode(chunk)

                            # language model inference against encoded audio
                            for c in range(codes.shape[-1]):
                                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                                if tokens is None:
                                    # model is silent
                                    continue

                                assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

                                text_token = tokens[0, 0, 0].item()
                                if text_token not in (0, 3):
                                    text = self.text_tokenizer.id_to_piece(text_token)
                                    text = text.replace("▁", " ")
                                    self.transcription_queue.put_nowait(text)

                                    # calculate latency from when audio was received
                                    latency_ms = None
                                    if frame_audio_time is not None:
                                        current_time = time.monotonic_ns()
                                        latency_ms = round(
                                            (current_time - frame_audio_time) / 1e6, 2
                                        )
                                        if is_first_token:
                                            is_first_token = False
                                            ttft_ms = round(
                                                (current_time - stream_start_time)
                                                / 1e6,
                                                2,
                                            )
                                            print(f"TTFT: {ttft_ms}ms")
                                        else:
                                            print(f"Latency: {latency_ms}ms")

            async def send_loop():
                """
                Reads outbound data, and sends it across websocket
                """
                while True:
                    await asyncio.sleep(0.001)
                    try:
                        text = self.transcription_queue.get_nowait()
                        if text is None:
                            continue
                        msg = b"\x00" + bytes(
                            text, encoding="utf8"
                        )  # prepend "\x00" as a tag to indicate text
                        await ws.send_bytes(msg)
                    except asyncio.queues.QueueEmpty:
                        continue

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
