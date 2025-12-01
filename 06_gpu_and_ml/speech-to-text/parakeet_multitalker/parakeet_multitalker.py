# ---
# lambda-test: false
# pytest: false
# ---

# # Parakeet Multi-talker Speech-to-Text

# This example shows how to run a streaming multi-talker speech-to-text service
# using NVIDIA's Parakeet Multi-talker model and Sortformer diarization model.
# The application transcribes audio in real-time while identifying different speakers.

# Click the "View on GitHub" button to see the code. And [sign up for a Modal account](https://modal.com/signup) if you haven't already.

# ## Setup

# We start by importing the necessary dependencies and defining the Modal App and Image.
# We use a persistent Volume to cache the models.

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import modal

app = modal.App("parakeet-multitalker")
model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)
CACHE_PATH = "/cache"
hf_secret = modal.Secret.from_name("huggingface-secret")

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

SAMPLE_RATE = 16000
NUM_REQUIRED_BUFFER_FRAMES = 13
BYTES_PER_SAMPLE = 2
FRAME_LEN_SEC = 0.080
PARAKEET_RT_STREAMING_CHUNK_SIZE = (
    int(FRAME_LEN_SEC * SAMPLE_RATE) * BYTES_PER_SAMPLE * NUM_REQUIRED_BUFFER_FRAMES
)


def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# ## Configuration

# This dataclass holds all the configuration parameters for the transcription and diarization models.


@dataclass
class MultitalkerTranscriptionConfig:
    """
    Configuration for Multi-talker transcription with an ASR model and a diarization model.
    """

    # Required configs
    diar_model: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    max_num_of_spks: Optional[int] = 4  # maximum number of speakers
    parallel_speaker_strategy: bool = True  # whether to use parallel speaker strategy
    masked_asr: bool = True  # whether to use masked ASR
    mask_preencode: bool = False  # whether to mask preencode or mask features
    cache_gating: bool = True  # whether to use cache gating
    cache_gating_buffer_size: int = 2  # buffer size for cache gating
    single_speaker_mode: bool = False  # whether to use single speaker mode

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    num_workers: int = 8
    random_seed: Optional[int] = (
        None  # seed number going to be used in seed_everything()
    )
    log: bool = True  # If True,log will be printed

    # Streaming diarization configs
    streaming_mode: bool = True  # If True, streaming diarization will be used.
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 1
    chunk_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = True
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = True
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None  # ["ctc", "rnnt"]
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    generate_realtime_scripts: bool = True

    word_window: int = 50
    sent_break_sec: float = 30.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = True
    print_path: Optional[str] = "./"

    ignored_initial_frame_steps: int = 5
    verbose: bool = True

    feat_len_sec: float = 0.01
    finetune_realtime_ratio: float = 0.01

    spk_supervision: str = "diar"  # ["diar", "rttm"]
    binary_diar_preds: bool = False
    deploy_mode: bool = True

    @staticmethod
    def init_diar_model(cfg, diar_model):
        # Set streaming mode diar_model params (matching the diarization setup from lines 263-271 of reference file)
        diar_model.streaming_mode = cfg.streaming_mode
        diar_model.sortformer_modules.chunk_len = (
            cfg.chunk_len if cfg.chunk_len > 0 else 6
        )
        diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = (
            cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
        )
        diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        diar_model.sortformer_modules.log = cfg.log
        diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
        return diar_model


with image.imports():
    import logging
    from urllib.request import urlopen

    import numpy as np
    import torch
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
        SpeakerTaggedASR,
    )
    from omegaconf import OmegaConf
    from starlette.websockets import WebSocketState

    from .asr_utils import int2float, preprocess_audio
    from .cache_aware_buffer import CacheAwareStreamingAudioBuffer


# ## Transcriber Service

# We define the main `Transcriber` class as a Modal Cls.
# This class loads the models into GPU memory and handles the streaming inference.
# For more on lifecycle management with Cls and cold start penalty reduction on Modal, see
# [this guide](https://modal.com/docs/guide/cold-start). In particular, this model
# is amenable to GPU snapshots.


@app.cls(
    volumes={CACHE_PATH: model_cache},
    gpu=["A100"],
    image=image,
    secrets=[hf_secret] if hf_secret is not None else [],
)
class Transcriber:
    @modal.enter()
    # @modal.enter()
    async def load(self):
        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.diar_model = (
            SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2.1"
            )
            .eval()
            .to(torch.device("cuda"))
        )
        self.asr_model = (
            ASRModel.from_pretrained("nvidia/multitalker-parakeet-streaming-0.6b-v1")
            .eval()
            .to(torch.device("cuda"))
        )

        self.cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        self.diar_model = MultitalkerTranscriptionConfig.init_diar_model(
            self.cfg, self.diar_model
        )
        self.multispk_asr_streamer = SpeakerTaggedASR(
            self.cfg, self.asr_model, self.diar_model
        )

        self._chunk_size = PARAKEET_RT_STREAMING_CHUNK_SIZE

        # warm up gpu
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = urlopen(AUDIO_URL).read()
        audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)

        # We use a `CacheAwareStreamingAudioBuffer` to manage the audio stream.
        # This buffer handles the streaming input and output, ensuring that the model receives
        # the correct amount of audio data for each inference step.

        self.streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr_model,
            online_normalization=self.cfg.online_normalization,
            pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded,
        )

        self.streaming_buffer.reset_buffer()

        step_num = 0
        stream_id = -1
        for audio_data in chunk_audio(audio_bytes, PARAKEET_RT_STREAMING_CHUNK_SIZE):
            transcript, stream_id = await self.transcribe(
                audio_data, step_num, stream_id
            )
            step_num += 1
            stream_id = 0
            print(f"transcript: {transcript}")
            print(f"stream_id: {stream_id}")

        self.streaming_buffer.reset_buffer()

        # ### WebSocket Handling

        # We use FastAPI's WebSocket support to handle the audio stream.
        # Incoming audio bytes are buffered and processed in chunks, and
        # transcriptions are sent back to the client as they become available.

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()

            self.streaming_buffer.reset_buffer()

            async def recv_loop(ws, audio_queue):
                audio_buffer = bytearray()
                while True:
                    data = await ws.receive_bytes()
                    audio_buffer.extend(data)
                    if len(audio_buffer) > self._chunk_size:
                        print("sending audio data")
                        await audio_queue.put(audio_buffer)
                        audio_buffer = bytearray()

            async def inference_loop(audio_queue, transcription_queue):
                step_num = 0
                stream_id = -1
                while True:
                    audio_data = await audio_queue.get()

                    start_time = time.perf_counter()
                    print("transcribing audio data")
                    transcript, stream_id = await self.transcribe(
                        audio_data, step_num, stream_id
                    )
                    step_num += 1
                    stream_id = 0
                    print(f"transcript: {transcript}")
                    if transcript:
                        await transcription_queue.put(transcript)

                    end_time = time.perf_counter()
                    print(
                        f"time taken to transcribe audio segment: {end_time - start_time} seconds"
                    )

            async def send_loop(transcription_queue, ws):
                while True:
                    transcript = await transcription_queue.get()
                    print(f"sending transcription data: {transcript}")
                    await ws.send_text(transcript)

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
                    asyncio.create_task(
                        inference_loop(audio_queue, transcription_queue)
                    ),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                ws = None
            except Exception as e:
                print("Exception:", e)
            finally:
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

    async def transcribe(self, audio_data, step_num, stream_id=-1) -> str:
        print(f"transcribing audio data: {len(audio_data)} bytes")

        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not self.cfg.pad_and_drop_preencoded
            else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        # convert to numpy
        audio_data = int2float(np.frombuffer(audio_data, dtype=np.int16))
        processed_signal, processed_signal_length, stream_id = (
            self.streaming_buffer.append_audio(audio_data, stream_id=stream_id)
        )

        result = self.streaming_buffer.get_next_chunk()
        if result is not None:
            audio_chunk, chunk_lengths = result
        else:
            return None, stream_id

        with torch.inference_mode():
            with torch.amp.autocast(self.diar_model.device.type, enabled=True):
                with torch.no_grad():
                    result = (
                        self.multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                            step_num=step_num,
                            chunk_audio=audio_chunk,
                            chunk_lengths=chunk_lengths,
                            is_buffer_empty=False,
                            drop_extra_pre_encoded=drop_extra_pre_encoded,
                        )
                    )
        if result:
            return result[0], stream_id
        return None, stream_id

    @modal.asgi_app()
    def webapp(self):
        return self.web_app

    @modal.method()
    def ping(self):
        return "pong"


# ## Frontend Service

# We serve a simple HTML/JS frontend to interact with the transcriber.
# The frontend captures microphone input and streams it to the WebSocket endpoint.

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi")
    .add_local_dir(Path(__file__).parent / "multitalker-frontend", "/root/frontend")
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
            cls_instance = Transcriber()
            ws_base_url = (
                cls_instance.webapp.get_web_url().replace("http", "ws") + "/ws"
            )
            script_tag = f'<script>window.WS_BASE_URL = "{ws_base_url}"; window.TRANSCRIPTION_MODE = "replace";</script>'
            html_content = html_content.replace(
                '<script src="/static/parakeet.js"></script>',
                f'{script_tag}\n<script src="/static/parakeet.js"></script>',
            )
            return HTMLResponse(content=html_content)

        return web_app


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
