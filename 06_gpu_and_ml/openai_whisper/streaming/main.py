# ---
# runtimes: ["runc", "gvisor"]
# ---
import asyncio
import io
import logging
import pathlib
import re
import tempfile
import time
from typing import Iterator

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
        "pytube @ git+https://github.com/felipeucelli/pytube",
    )
)
app = modal.App(name="example-whisper-streaming", image=image)
web_app = FastAPI()
CHARLIE_CHAPLIN_DICTATOR_SPEECH_URL = (
    "https://www.youtube.com/watch?v=J7GY1Xg6X20"
)


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
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        else:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .filter("atrim", start=start, end=end)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
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
        accuracy. Whisper models expect to transcribe in 30 second segments, so this is the
        default minimum.
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


@app.function()
def download_mp3_from_youtube(youtube_url: str) -> bytes:
    from pytube import YouTube

    logging.getLogger("pytube").setLevel(logging.INFO)
    yt = YouTube(youtube_url)
    video = yt.streams.filter(only_audio=True).first()
    buffer = io.BytesIO()
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer.read()


@app.function(cpu=2)
def transcribe_segment(
    start: float,
    end: float,
    audio_data: bytes,
    model: str,
):
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


@web_app.get("/transcribe")
async def transcribe(url: str):
    """
    Usage:

    ```sh
    curl --no-buffer \
        https://modal-labs--example-whisper-streaming-web.modal.run/transcribe?url=https://www.youtube.com/watch?v=s_LncVnecLA"
    ```

    This endpoint will stream back the Youtube's audio transcription as it makes progress.

    Some example Youtube videos for inspiration:

    1. Churchill's 'We shall never surrender' speech - https://www.youtube.com/watch?v=s_LncVnecLA
    2. Charlie Chaplin's final speech from The Great Dictator - https://www.youtube.com/watch?v=J7GY1Xg6X20
    """
    import pytube.exceptions

    print(f"downloading {url}")
    try:
        audio_data = download_mp3_from_youtube.remote(url)
    except pytube.exceptions.RegexMatchError:
        raise HTTPException(
            status_code=422, detail=f"Could not process url {url}"
        )
    print(f"streaming transcription of {url} audio to client...")
    return StreamingResponse(
        stream_whisper(audio_data), media_type="text/event-stream"
    )


@app.function()
@modal.asgi_app()
def web():
    return web_app


@app.function()
async def transcribe_cli(data: bytes, suffix: str):
    async for result in stream_whisper(data):
        print(result)


@app.local_entrypoint()
def main(path: str = CHARLIE_CHAPLIN_DICTATOR_SPEECH_URL):
    if path.startswith("https"):
        data = download_mp3_from_youtube.remote(path)
        suffix = ".mp3"
    else:
        filepath = pathlib.Path(path)
        data = filepath.read_bytes()
        suffix = filepath.suffix
    transcribe_cli.remote(
        data,
        suffix=suffix,
    )
