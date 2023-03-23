import asyncio
import json
import io
import logging
import pathlib
import tempfile
from typing import Iterator

import modal

from fastapi import FastAPI, Header, File, UploadFile, Request
from fastapi.responses import StreamingResponse

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "python-multipart==0.0.6",
        "git+https://github.com/shirayu/whispering.git@v0.6.6",
        "ffmpeg-python",
        "pytube~=12.1.2",
    )
)
stub = modal.Stub(name="example-whisper-streaming", image=image)

web_app = FastAPI()


def load_audio(data: bytes, start=None, end=None, sr: int = 16000):
    import tempfile
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
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re
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


@stub.function
def download_mp3_from_youtube(youtube_url: str) -> bytes:
    import logging
    from pytube import YouTube
    logging.getLogger("pytube").setLevel(logging.DEBUG)
    yt = YouTube(youtube_url)
    video = yt.streams.filter(only_audio=True).first()
    buffer = io.BytesIO()
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer.read()


@stub.function(
    cpu=2,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_data: bytes,
    model: str,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper
    print(f"Transcribing segment {start:.2f} to {end:.2f}")

    t0 = time.time()
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model, device=device)
    result = model.transcribe(load_audio(audio_data, start=start, end=end), language="en", fp16=use_gpu)  # type: ignore
    print(
        f"Transcribed segment {start:.2f} to {end:.2f} of {end - start:.2f} in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


async def stream_whisper(audio_data: bytes):
    import logging
    import numpy as np

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_data)
        f.flush()
        file_len = len(pathlib.Path(f.name).read_bytes())
        print(f"file len is {file_len}")
        segment_gen = split_silences(f.name)

        for result in transcribe_segment.starmap(
            segment_gen, kwargs=dict(audio_data=audio_data, model="base.en"),
            order_outputs=True,
        ):
            yield result["text"]


@web_app.post("/transcribe")
async def transcribe(request: Request):
    """
    Transcribe an 'example.wav' file in your current directory with the following
    `curl` request:

    ```sh
curl --verbose -X POST https://modal-labs--example-whisper-streaming-fastapi-app-th-7ade17-dev.modal.run/transcribe \
    -H  "accept: application/json" \
    -H  "Content-Type: multipart/form-data" \
    -F "audio=@harvard.wav;type=audio/wav"
    ```

    This endpoint will stream back the audio file's transcription as it makes progress.
    """
    body = await request.body()
    # FastAPI's built-in `File` type does not work with `curl`.
    # https://github.com/encode/starlette/issues/1059#issuecomment-696419776
    request._body = body.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    inp = await request.form()
    audio = inp["audio"]
    # content_size = int(request.headers['content-length'])
    print(audio)
    print(dir(audio))
    print(audio.headers)
    print(f"{audio.file=}")
    print("reading from file, which should be 12034158 for churchill.mp3")
    audio_data = audio.file.read(12034158)
    print(len(audio_data))
    # return
    return StreamingResponse(stream_whisper(audio_data))


@stub.asgi(gpu="any")
def fastapi_app():
    return web_app


@stub.function(gpu="any", timeout=600)
async def transcribe_cli(data: bytes, suffix: str):
    async for result in stream_whisper(data):
        print(result)


@stub.local_entrypoint
def main(filepath: str):
    data = download_mp3_from_youtube.call("https://www.youtube.com/watch?v=s_LncVnecLA")
    filepath = pathlib.Path(filepath)
    # data = filepath.read_bytes()
    transcribe_cli.call(
        data,
        suffix=filepath.suffix,
    )
