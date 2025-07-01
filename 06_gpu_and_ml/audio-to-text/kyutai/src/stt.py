"""
STT websocket web service.
"""

import asyncio
from pathlib import Path

import modal

from .common import app

image = (
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

with image.imports():
    import numpy as np
    import sphn
    import torch
    from huggingface_hub import snapshot_download
    from moshi.models import LMGen, loaders


N_GPU = 1
GPU_CONFIG = f"l40s:{N_GPU}"
MINUTES = 60


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes=volumes,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=64)
class STT:
    @modal.enter()
    def download_model(self):
        print("Downloading model if necessary...")
        snapshot_download(MODEL_NAME, revision=MODEL_REVISION)

    @modal.enter()
    def enter(self):
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

    def reset_state(self):
        # use opus format for audio across websocket to safely stream/decode in real-time
        self.opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)

        # reset llm chat history on each connection
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            with torch.no_grad():
                await ws.accept()

                # clear llm chat history + buffered audio
                self.reset_state()

                print("Session started")
                tasks = []

                # asyncio to run multiple loops concurrently within single websocket connection
                async def recv_loop():
                    """
                    Receives Opus stream across websocket, appends into opus_stream_inbound.
                    """
                    while True:
                        data = await ws.receive_bytes()

                        if not isinstance(data, bytes):
                            print("received non-bytes message")
                            continue
                        if len(data) == 0:
                            print("received empty message")
                            continue
                        self.opus_stream_inbound.append_bytes(data)

                async def inference_loop():
                    """
                    Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                    """
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
                            chunk = all_pcm_data[: self.frame_size]
                            all_pcm_data = all_pcm_data[self.frame_size :]

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
                                    msg = b"\x00" + bytes(
                                        text, encoding="utf8"
                                    )  # prepend "\x00" as a tag to indicate text
                                    await ws.send_bytes(msg)

                # run all loops concurrently
                try:
                    tasks = [
                        asyncio.create_task(recv_loop()),
                        asyncio.create_task(inference_loop()),
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
