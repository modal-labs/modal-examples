import asyncio
import pathlib
import re
import tempfile
import time
import urllib
from typing import Iterator

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "fastapi==0.116.1",
        "ffmpeg-python==0.2.0",
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "numpy<2",
    )
)
app = modal.App(name="example-streaming-whisper", image=image)
SAMPLE_URL = (
    "https://modal-cdn.com/history-of-rome-podcast-duncan-001-in-the-beginning.mp3"
)

CACHE_DIR = "/root/.cache/whisper"
whisper_cache = modal.Volume.from_name("whisper-cache", create_if_missing=True)


def load_audio(data: bytes, start=None, end=None, sr: int = 16000):
    import ffmpeg
    import numpy as np

    try:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fp.write(data)
        fp.close()
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        if start is None and end is None:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        else:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .filter("atrim", start=start, end=end)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
) -> Iterator[tuple[float, float]]:
    """
    Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds.

    Parameters
    ----------
    path: str
        path to the audio file on disk.
    min_segment_length : float
        The minimum acceptable length for an audio segment in seconds. Lower values
        allow for more splitting and increased parallelizing, but decrease transcription
        accuracy. Whisper models expect to transcribe in 30 second segments.
    min_silence_length : float
        Minimum silence to detect and split on, in seconds. Lower values are more likely to split
        audio in middle of phrases and degrade transcription accuracy.
    """
    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    print(f"Split {path} into {num_segments} segments")


@app.function(gpu="a10", volumes={CACHE_DIR: whisper_cache})
def transcribe_segment(start: float, end: float, audio_data: bytes, model: str):
    import torch
    import whisper

    print(
        f"Transcribing segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration)"
    )

    t0 = time.time()
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model, device=device)
    np_array = load_audio(audio_data, start=start, end=end)
    result = model.transcribe(np_array, language="en", fp16=use_gpu)  # type: ignore
    print(
        f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


async def stream_whisper(audio_data: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_data)
        f.flush()
        segment_gen = split_silences(f.name)

    async for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_data=audio_data, model="base.en")
    ):
        # Must cooperatively yield here otherwise `StreamingResponse` will not iteratively return stream parts.
        # see: https://github.com/python/asyncio/issues/284#issuecomment-154162668
        await asyncio.sleep(0)
        yield result["text"]


@app.function()
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    web_app = FastAPI()

    @web_app.get("/transcribe")
    async def transcribe(url: str):
        """
        Usage:

        ```sh
        curl --no-buffer \
            https://modal-labs-examples--example-streaming-whisper-api.modal.run/transcribe?url=https://modal-cdn.com/history-of-rome-podcast-duncan-001-in-the-beginning.mp3
        ```

        This endpoint will stream back the audio transcription as it makes progress.
        """
        print(f"downloading {url}")
        try:
            with urllib.request.urlopen(url) as response:
                assert response.getcode() == 200, response.getcode()
                audio_data = response.read()
        except AssertionError:
            raise HTTPException(status_code=422, detail=f"Could not process url {url}")
        print(f"streaming transcription of {url} audio to client...")
        return StreamingResponse(
            stream_whisper(audio_data), media_type="text/event-stream"
        )

    return web_app


@app.function()
async def transcribe_cli(data: bytes, suffix: str):
    async for result in stream_whisper(data):
        print(result)


@app.local_entrypoint()
def main(path: str = SAMPLE_URL):
    if path.startswith("http"):
        with urllib.request.urlopen(path) as response:
            assert response.getcode() == 200, response.getcode()
            data = response.read()
        suffix = path.rsplit(".")[-1]
    else:
        filepath = pathlib.Path(path)
        data = filepath.read_bytes()
        suffix = filepath.suffix
    transcribe_cli.remote(data, suffix=suffix)
