import asyncio
import base64
import time
from pathlib import Path

import modal

app = modal.App(name="orpheus")

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode python-fasthtml==0.12.20",
    )
    .add_local_dir(Path(__file__).parent / "frontend", "/root/frontend")
)


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
            # Audio playback libraries
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
                "Orpheus TTS",
            ),
            fh.Body(
                fh.Div(
                    fh.Select(
                        fh.Option("Tara", value="tara", selected=True),
                        fh.Option("Leah", value="leah"),
                        fh.Option("Jess", value="jess"),
                        fh.Option("Leo", value="leo"),
                        fh.Option("Dan", value="dan"),
                        fh.Option("Mia", value="mia"),
                        fh.Option("Zac", value="zac"),
                        fh.Option("Zoe", value="zoe"),
                        id="voice-select",
                        cls="bg-gray-700 text-white rounded py-3 focus:outline-none focus:ring-2 focus:ring-primary text-center",
                    ),
                    fh.Textarea(
                        id="text-input",
                        placeholder="Enter text to synthesize...",
                        cls="w-full p-4 bg-gray-700 text-white rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary",
                        rows="4",
                    ),
                    fh.Div(
                        "💡 Tip: Use emotive tags like ",
                        fh.Code(
                            "<laugh>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<chuckle>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<sigh>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<cough>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<sniffle>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<groan>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", ",
                        fh.Code(
                            "<yawn>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        ", and ",
                        fh.Code(
                            "<gasp>",
                            cls="text-sm px-1 bg-gray-700 rounded text-accent-pink",
                        ),
                        " to add emotion to your text.",
                        cls="text-gray-400",
                    ),
                    fh.Div(
                        fh.Button(
                            "Synthesize",
                            id="synthesize-btn",
                            cls="px-6 py-2 bg-primary text-black font-semibold rounded-lg hover:bg-green-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
                            disabled=True,
                        ),
                        fh.Button(
                            "Stop",
                            id="stop-btn",
                            cls="px-6 py-2 bg-gray-600 text-white font-semibold rounded-lg hover:bg-gray-500 transition-colors ml-2 disabled:opacity-50 disabled:cursor-not-allowed",
                            disabled=True,
                        ),
                        cls="flex justify-center items-center gap-2",
                    ),
                    fh.Div(
                        id="status",
                        cls="text-gray-400",
                    ),
                    cls="bg-gray-800 rounded-lg shadow-lg w-full max-w-2xl p-6 flex flex-col items-center gap-8",
                ),
                fh.Footer(
                    fh.Span(
                        "Built with ",
                        fh.A(
                            "Orpheus TTS",
                            href="https://github.com/canopyai/Orpheus-TTS",
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


tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands("git clone https://github.com/canopyai/Orpheus-TTS.git")
    .run_commands(
        "cd Orpheus-TTS/orpheus_tts_pypi && uv pip install --system --compile-bytecode ."
    )
    .run_commands(
        "uv pip install --system --compile-bytecode --force-reinstall vllm==0.7.3 hf_transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "canopylabs/orpheus-3b-0.1-ft"
MODEL_REVISION = "4206a56e5a68cf6cf96900a8a78acd3370c02eb6"

hf_cache_vol = modal.Volume.from_name(f"{app.name}-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")
volumes = {hf_cache_vol_path: hf_cache_vol}

secrets = [modal.Secret.from_name("hf-token")]

MINUTES = 60


@app.cls(
    image=tts_image,
    gpu="l40s:1",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
class TTS:
    BATCH_SIZE = 1
    FRAME_RATE = 24000

    @modal.enter()
    def enter(self):
        import torch
        from huggingface_hub import snapshot_download
        from orpheus_tts import OrpheusModel

        start_time = time.monotonic_ns()

        print("Downloading model if necessary...")
        snapshot_download(MODEL_NAME, revision=MODEL_REVISION)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = OrpheusModel(
            model_name=MODEL_NAME,
        )

        print(f"Model loaded in {round((time.monotonic_ns() - start_time) / 1e9, 2)}s")

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

            input_text_queue = asyncio.Queue()
            output_audio_queue = asyncio.Queue()

            print("Session started")
            tasks = []

            async def recv_loop():
                """
                Receives text input across websocket, and appends it to the outbound stream.
                """
                nonlocal input_text_queue
                while True:
                    data = await ws.receive_json()
                    input_text_queue.put_nowait(data)

            async def inference_loop():
                """
                Runs streaming inference on text input and appends response audio to the outbound stream.
                """
                nonlocal input_text_queue, output_audio_queue

                while True:
                    await asyncio.sleep(0.001)

                    try:
                        input_text = input_text_queue.get_nowait()
                    except asyncio.queues.QueueEmpty:
                        continue

                    for audio_chunk in self.model.generate_speech(
                        prompt=input_text["text"],
                        voice=input_text["voice"],
                    ):
                        output_audio_queue.put_nowait(audio_chunk)

            async def send_loop():
                """
                Reads outbound data, and sends it across websocket
                """
                nonlocal output_audio_queue
                while True:
                    await asyncio.sleep(0.001)

                    try:
                        msg = output_audio_queue.get_nowait()
                    except asyncio.queues.QueueEmpty:
                        continue

                    msg = b"\x01" + msg  # prepend "\x01" as a tag to indicate audio
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

        return web_app

    @modal.method()
    def local_test(self, text: str, voice: str):
        for token in self.model.generate_speech(
            prompt=text,
            voice=voice,
        ):
            yield token

    @modal.method()
    def boot(self):
        pass


@app.local_entrypoint()
async def test(text: str = None, voice: str = None):
    import wave

    if text is None:
        text = "Hello, how are you?"
        print(f"Using default test text: {text}")
    if voice is None:
        voice = "tara"
        print(f"Using default test voice: {voice}")

    tts = TTS()
    tts.boot.remote()

    start_time = time.monotonic()

    save_path = Path("/tmp/orpheus/output.wav")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(save_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.FRAME_RATE)

        total_frames = 0
        chunk_counter = 0

        async for audio_chunk in tts.local_test.remote_gen.aio(text, voice):
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

    duration = total_frames / wf.getframerate()
    print(
        f"It took {time.monotonic() - start_time} seconds to generate {duration:.2f} seconds of audio"
    )
    print(f"Saved to {save_path}")
