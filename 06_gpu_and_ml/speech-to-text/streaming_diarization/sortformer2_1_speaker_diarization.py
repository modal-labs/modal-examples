# ---
# lambda-test: false
# pytest: false
# ---

from pathlib import Path
from typing import Literal

import modal

app = modal.App("sortformer2-1-speaker-diarization")

CACHE_PATH = "/model"
cache_vol = modal.Volume.from_name("sortformer2_1-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,  # cache directory for Hugging Face models
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": CACHE_PATH,
        }
    )
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "cuda-python==13.0.1",
        "numpy<2",
        "fastapi",
        "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@main",
    )
)

with image.imports():
    import asyncio
    import json
    import time

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState

    from .sortformer2_1 import DiarizationConfig, NeMoStreamingDiarizer


@app.cls(
    image=image,
    volumes={CACHE_PATH: cache_vol},
    gpu="L4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Sortformer2_1_Speaker_Diarization:
    @modal.enter()
    def enter(self):
        self._LOW_LATENCY_CONFIG = DiarizationConfig(
            max_num_speakers=4,
            chunk_len=6,
            chunk_right_context=7,
            fifo_len=188,
            spkcache_refresh_rate=144,
            spkcache_len=188,
        )
        self._HIGH_LATENCY_CONFIG = DiarizationConfig(
            max_num_speakers=4,
            chunk_len=340,
            chunk_right_context=40,
            fifo_len=40,
            spkcache_refresh_rate=300,
            spkcache_len=188,
        )
        self.latency: Literal["low", "high"] = "low"
        self._SORTFORMER_FRAME_SIZE_BYTES = (
            16000 * 0.08 * 2
        )  # sample rate * frame size in seconds * 2 bytes (16 bit)
        if self.latency == "low":
            self.config = self._LOW_LATENCY_CONFIG
        else:
            self.config = self._HIGH_LATENCY_CONFIG
        # load model from Hugging Face model card directly (You need a Hugging Face token)
        self.diarizer = NeMoStreamingDiarizer(
            cfg=self.config, model="nvidia/diar_streaming_sortformer_4spk-v2.1"
        )

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            audio_queue = asyncio.Queue()
            output_queue = asyncio.Queue()

            async def recv_loop(ws, audio_queue):
                audio_buffer = bytearray()
                while True:
                    data = await ws.receive_bytes()
                    audio_buffer.extend(data)
                    if len(audio_buffer) > self._SORTFORMER_FRAME_SIZE_BYTES:
                        await audio_queue.put(audio_buffer)
                        audio_buffer = bytearray()

            async def inference_loop(audio_queue, output_queue):
                while True:
                    audio_data = await audio_queue.get()

                    start_time = time.perf_counter()
                    diar_result = self.diarizer.diarize(audio_data)

                    probs = self._get_speaker_probabilities(diar_result)
                    await output_queue.put(json.dumps(probs))

                    end_time = time.perf_counter()
                    print(
                        f"time taken to diarize audio segment: {end_time - start_time} seconds"
                    )

            async def send_loop(output_queue, ws):
                while True:
                    output = await output_queue.get()
                    print(f"sending diarize result: {output}")
                    await ws.send_text(output)

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
                    asyncio.create_task(inference_loop(audio_queue, output_queue)),
                    asyncio.create_task(send_loop(output_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                ws = None
            except Exception as e:
                print("Exception:", e)
            finally:
                self.diarizer.reset_state()
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close(code=1011)  # internal error
                    ws = None
                for task in tasks:
                    if not task.done():
                        try:
                            task.cancel()
                            await task
                        except asyncio.CancelledError:
                            pass

    @modal.asgi_app()
    def webapp(self):
        return self.web_app

    def _get_speaker_probabilities(self, spk_pred):
        # spk_pred is a 6x4 matrix of probabilities
        # We want to return a 1x4 vector of probabilities for the total time window
        # We can take the mean across the time dimension (axis 0)
        return spk_pred.mean(axis=0).tolist()


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi")
    .add_local_dir(
        Path(__file__).parent / "streaming-diarization-frontend", "/root/frontend"
    )
)

with web_image.imports():
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles


@app.cls(image=web_image)
@modal.concurrent(max_inputs=20)
class WebServer:
    @modal.asgi_app()
    def web(self):
        web_app = FastAPI()
        web_app.mount("/static", StaticFiles(directory="frontend"))

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        # serve frontend
        @web_app.get("/")
        async def index():
            html_content = open("frontend/index.html").read()

            # Get the base WebSocket URL (without transcriber parameters)
            cls_instance = modal.Cls.from_name(
                "sortformer2-1-speaker-diarization", "Sortformer2_1_Speaker_Diarization"
            )()
            ws_base_url = (
                cls_instance.webapp.get_web_url().replace("http", "ws") + "/ws"
            )
            script_tag = f'<script>window.WS_BASE_URL = "{ws_base_url}";</script>'
            html_content = html_content.replace(
                '<script src="/static/sortformer2_1.js"></script>',
                f'{script_tag}\n<script src="/static/sortformer2_1.js"></script>',
            )
            return HTMLResponse(content=html_content)

        return web_app
