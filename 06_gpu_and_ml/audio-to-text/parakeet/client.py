# ---
# lambda-test: false
# ---

# Reference client code for parakeet example

import asyncio
import websockets
import sounddevice as sd
import numpy as np
import argparse

profile = "modal-labs"
app_name = "parakeet-websocket"
class_name = "parakeet"
is_dev = False

DEFAULT_URL = (
    f"wss://{profile}--{app_name}-{class_name}-web{'-dev' if is_dev else ''}.modal.run"
)
WS_ENDPOINT = "/ws"

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DTYPE = np.int16


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
        print("Recording and streaming... Press Ctrl+C to stop.")
        while True:
            audio_chunk = await audio_queue.get()
            await websocket.send(audio_chunk)


async def receive_transcriptions(websocket):
    async for message in websocket:
        print("📝 Transcription:", message)


async def main(ws_url):
    audio_queue = asyncio.Queue()
    async with websockets.connect(
        ws_url, open_timeout=120, ping_interval=None
    ) as websocket:
        send_task = asyncio.create_task(send_audio(websocket, audio_queue))
        receive_task = asyncio.create_task(receive_transcriptions(websocket))
        await asyncio.gather(send_task, receive_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream audio to a WebSocket ASR server."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        help="WebSocket server URL (e.g., wss://example.modal.run)",
    )
    args = parser.parse_args()

    if args.url:
        url = args.url
    else:
        print(
            f"Using default WebSocket base URL: {DEFAULT_URL} (you can override with --url)"
        )
        url = DEFAULT_URL

    ws_url = f"{url}{WS_ENDPOINT}"

    print("☀️ Waking up model, this may take a few seconds on cold start...\n")
    try:
        asyncio.run(main(ws_url))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
