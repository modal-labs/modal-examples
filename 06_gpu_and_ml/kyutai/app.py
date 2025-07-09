import asyncio
import base64
import dataclasses
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
        "uv pip install --system --compile-bytecode moshi==0.2.9 fastapi==0.115.14 hf_transfer==0.1.9 julius==0.2.7",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(Path(__file__).parent / "bria.mp3", "/root/bria.mp3")
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


@dataclasses.dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]

    def __str__(self):
        return f"{self.text} ({self.timestamp[0]:.2f}:{self.timestamp[1]:.2f})"


@app.cls(
    image=stt_image,
    gpu="l40s:1",
    volumes=volumes,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
class STT:
    BATCH_SIZE = 1

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

    @modal.asgi_app()
    def web(self):
        import numpy as np
        import sphn
        import torch
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()

            opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)
            transcription_queue = asyncio.Queue()
            is_first_token = True
            stream_start_time = None
            last_audio_receive_time = None

            print("Session started")
            tasks = []

            # asyncio to run multiple loops concurrently within single websocket connection
            async def recv_loop():
                """
                Receives Opus stream across websocket, appends into inbound queue.
                """
                nonlocal opus_stream_inbound, stream_start_time, last_audio_receive_time
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
                    last_audio_receive_time = time.monotonic_ns()
                    opus_stream_inbound.append_bytes(data)

            async def inference_loop():
                """
                Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                """
                nonlocal \
                    opus_stream_inbound, \
                    transcription_queue, \
                    is_first_token, \
                    stream_start_time, \
                    last_audio_receive_time
                all_pcm_data = None

                while True:
                    await asyncio.sleep(0.001)

                    pcm = opus_stream_inbound.read_pcm()
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
                        if is_first_token and stream_start_time is not None:
                            ttft_ms = round(
                                (time.monotonic_ns() - stream_start_time) / 1e6, 2
                            )
                            is_first_token = False
                            stream_start_time = None
                            print(f"TTFT: {ttft_ms}ms")

                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size :]

                        with torch.no_grad():
                            chunk = torch.from_numpy(chunk)
                            chunk = chunk.unsqueeze(0).unsqueeze(
                                0
                            )  # (1, 1, frame_size)
                            chunk = chunk.expand(
                                self.BATCH_SIZE, -1, -1
                            )  # (batch_size, 1, frame_size)
                            chunk = chunk.to(device=self.device)

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

                                    transcription_queue.put_nowait(text)

                                    latency_ms = round(
                                        (time.monotonic_ns() - last_audio_receive_time)
                                        / 1e6,
                                        2,
                                    )
                                    print(f"Latency: {latency_ms}ms")

            async def send_loop():
                """
                Reads outbound data, and sends it across websocket
                """
                nonlocal transcription_queue
                while True:
                    await asyncio.sleep(0.001)

                    try:
                        text = transcription_queue.get_nowait()
                    except asyncio.queues.QueueEmpty:
                        continue

                    if text is None:
                        continue
                    msg = b"\x00" + bytes(
                        text, encoding="utf8"
                    )  # prepend "\x00" as a tag to indicate text
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

    def tokens_to_timestamped_text(
        self,
        text_tokens,
        end_of_padding_id,
        offset_seconds,
    ) -> list[TimestampedText]:
        import torch

        text_tokens = text_tokens.cpu().view(-1)

        # Normally `end_of_padding` tokens indicate word boundaries.
        # Everything between them should be a single word;
        # the time offset of the those tokens correspond to word start and
        # end timestamps (minus silence prefix and audio delay).
        #
        # However, in rare cases some complexities could arise. Firstly,
        # for words that are said quickly but are represented with
        # multiple tokens, the boundary might be omitted. Secondly,
        # for the very last word the end boundary might not happen.
        # Below is a code snippet that handles those situations a bit
        # more carefully.

        sequence_timestamps = []

        def _tstmp(start_position, end_position):
            return (
                max(0, start_position / self.mimi.frame_rate - offset_seconds),
                max(0, end_position / self.mimi.frame_rate - offset_seconds),
            )

        def _decode(t):
            t = t[t > self.padding_token_id]
            return self.text_tokenizer.decode(t.numpy().tolist())

        def _decode_segment(start, end):
            nonlocal text_tokens
            nonlocal sequence_timestamps

            text = _decode(text_tokens[start:end])
            words_inside_segment = text.split()

            if len(words_inside_segment) == 0:
                return
            if len(words_inside_segment) == 1:
                # Single word within the boundaries, the general case
                sequence_timestamps.append(
                    TimestampedText(text=text, timestamp=_tstmp(start, end))
                )
            else:
                # We're in a rare situation where multiple words are so close they are not separated by `end_of_padding`.
                # We tokenize words one-by-one; each word is assigned with as many frames as much tokens it has.
                for adjacent_word in words_inside_segment[:-1]:
                    n_tokens = len(self.text_tokenizer.encode(adjacent_word))
                    sequence_timestamps.append(
                        TimestampedText(
                            text=adjacent_word,
                            timestamp=_tstmp(start, start + n_tokens),
                        )
                    )
                    start += n_tokens

                # The last word takes everything until the boundary
                adjacent_word = words_inside_segment[-1]
                sequence_timestamps.append(
                    TimestampedText(text=adjacent_word, timestamp=_tstmp(start, end))
                )

        (segment_boundaries,) = torch.where(text_tokens == end_of_padding_id)

        if not segment_boundaries.numel():
            return []

        for i in range(len(segment_boundaries) - 1):
            segment_start = int(segment_boundaries[i]) + 1
            segment_end = int(segment_boundaries[i + 1])

            _decode_segment(segment_start, segment_end)

        last_segment_start = segment_boundaries[-1] + 1

        boundary_token = torch.tensor([self.text_tokenizer.eos_id()])
        (end_of_last_segment,) = torch.where(
            torch.isin(text_tokens[last_segment_start:], boundary_token)
        )

        if not end_of_last_segment.numel():
            # upper-bound either end of the audio or 1 second duration, whicher is smaller
            last_segment_end = min(
                text_tokens.shape[-1], last_segment_start + self.mimi.frame_rate
            )
        else:
            last_segment_end = last_segment_start + end_of_last_segment[0]
        _decode_segment(int(last_segment_start), int(last_segment_end))

        return sequence_timestamps

    @modal.method()
    def process_audio_file(self, audio_file: str):
        import itertools
        import math

        import julius
        import sphn
        import torch

        audio, input_sample_rate = sphn.read(audio_file)
        audio = torch.from_numpy(audio).to(self.device)
        audio = julius.resample_frac(audio, input_sample_rate, self.mimi.sample_rate)
        if audio.shape[-1] % self.mimi.frame_size != 0:
            to_pad = self.mimi.frame_size - audio.shape[-1] % self.mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, to_pad))

        text_tokens_accum = []

        n_prefix_chunks = math.ceil(
            self.audio_silence_prefix_seconds * self.mimi.frame_rate
        )
        n_suffix_chunks = math.ceil(self.audio_delay_seconds * self.mimi.frame_rate)
        silence_chunk = torch.zeros(
            (1, 1, self.mimi.frame_size), dtype=torch.float32, device=self.device
        )

        chunks = itertools.chain(
            itertools.repeat(silence_chunk, n_prefix_chunks),
            torch.split(audio[:, None], self.mimi.frame_size, dim=-1),
            itertools.repeat(silence_chunk, n_suffix_chunks),
        )

        start_time = time.time()
        nchunks = 0
        last_print_was_vad = False
        for audio_chunk in chunks:
            nchunks += 1
            audio_tokens = self.mimi.encode(audio_chunk)
            text_tokens, vad_heads = self.lm_gen.step_with_extra_heads(audio_tokens)
            if vad_heads:
                pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                if pr_vad > 0.5 and not last_print_was_vad:
                    print(" [end of turn detected]")
                    last_print_was_vad = True
            text_token = text_tokens[0, 0, 0].cpu().item()
            if text_token not in (0, 3):
                _text = self.text_tokenizer.id_to_piece(
                    text_tokens[0, 0, 0].cpu().item()
                )  # type: ignore
                _text = _text.replace("▁", " ")
                print(_text, end="", flush=True)
                last_print_was_vad = False
            text_tokens_accum.append(text_tokens)

        utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
        dt = time.time() - start_time
        print(
            f"\nprocessed {nchunks} chunks in {dt:.2f} seconds, steps per second: {nchunks / dt:.2f}"
        )
        timed_text = self.tokens_to_timestamped_text(
            utterance_tokens,
            end_of_padding_id=0,
            offset_seconds=int(n_prefix_chunks / self.mimi.frame_rate)
            + self.audio_delay_seconds,
        )

        decoded = " ".join([str(t) for t in timed_text])
        print(decoded)


@app.local_entrypoint()
def test(audio_file: str = None):
    if audio_file is None:
        audio_file = "/root/bria.mp3"
        print(f"Using default test audio: {audio_file}")
    else:
        print(f"Using provided audio: {audio_file}")

    stt = STT()
    stt.process_audio_file.remote(audio_file)
