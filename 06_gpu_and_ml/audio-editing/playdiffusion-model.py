# ---
# output-directory: "/tmp/playdiffusion"
# cmd: ["modal", "run", "06_gpu_and_ml/audio-editing/playdiffusion-model.py"]
# args: ["--audio-url", "https://modal-public-assets.s3.us-east-1.amazonaws.com/mono_44100_127389__acclivity__thetimehascome.wav", "--output-text", "November, '9 PM. I'm standing in alley. After waiting several hours, the time has come. A man with long dark hair approaches. I have to act and fast before he realizes what has happened. I must find out.", "--output-path", "/tmp/playdiffusion/output.wav"]
# ---


# # Run PlayDiffusion on Modal

# This example demonstrates how to run the [PlayDiffusion](https://huggingface.co/PlayHT/PlayDiffusion) audio editing model on Modal.
# PlayDiffusion is a model that takes an input audio and a desired output text, and then modifies the audio to say the output text.
# The function accepts text prompts and input audio as WAV files and returns generated audio as WAV files.
# We use Modal's class-based approach with GPU acceleration to provide fast, scalable inference.

# ## Setup

# Import the necessary modules
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

# ## Define a container image

# We start with Modal's baseline `debian_slim` image and install the required packages.
# - `openai`: PlayDiffusion requires a transcript as input. You can either provide the transcript yourself as input, or use a transcription model to transcribe the audio on the fly. In this case we use openai's whisper api, but you can use any model of your choice.
AUDIO_URL: str = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"

# The python version [needs to be](https://github.com/playht/PlayDiffusion/blob/d3995b9e2cd8a80b88be6aeeb4e35fd282b2d255/pyproject.toml) `3.11`
image: modal.Image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("openai==1.91.0")
    .run_commands(
        "pip install git+https://github.com/playht/PlayDiffusion.git@d3995b9e2cd8a80b88be6aeeb4e35fd282b2d255"
    )
)
app: modal.App = modal.App("example-playdiffusion-model", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing and the TTS model itself.

with image.imports():
    import os
    from urllib.request import urlopen

    import soundfile as sf
    from openai import OpenAI
    from playdiffusion import InpaintInput, PlayDiffusion


# ## The model class


# The service is implemented using Modal's class syntax with GPU acceleration.
# We configure the class to use an A10G GPU with additional parameters:


# - `scaledown_window=60 * 5`: Keep containers alive for 5 minutes after last request
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per containerå
@app.cls(gpu="a10g", scaledown_window=60 * 5)
@modal.concurrent(max_inputs=10)
class PlayDiffusionModel:
    @modal.enter()
    def load(self) -> None:
        self.inpainter = PlayDiffusion()

    @modal.method()
    def generate(
        self,
        audio_url: str,
        input_text: str,
        output_text: str,
        word_times: List[Dict[str, Any]],
    ) -> bytes:
        # Create a temporary file to store the audio
        temp_file_path: str = write_to_tempfile(audio_url)

        # Get the audio data and sample rate from inpainter
        sample_rate: int
        output_audio_data: bytes
        sample_rate, output_audio_data = self.inpainter.inpaint(
            InpaintInput(
                input_text=input_text,
                output_text=output_text,
                input_word_times=word_times,
                audio=temp_file_path,
            )
        )

        # Create an in-memory buffer
        buffer: io.BytesIO = io.BytesIO()

        # Write the audio data to the buffer as WAV
        sf.write(buffer, output_audio_data, sample_rate, format="WAV")

        # Reset buffer position to beginning
        buffer.seek(0)

        return buffer.getvalue()


# PlayDiffusion requires a transcript as input. You can either provide the transcript yourself as input, or use a transcription model
# to transcribe the audio on the fly. In this case we use openai's whisper api, but you can use any model of your choice.
@app.function(
    secrets=[modal.Secret.from_name("openai-secret", environment_name="main")]
)
def run_asr(audio_url: str) -> Tuple[str, List[Dict[str, Any]]]:
    temp_file_path: str = write_to_tempfile(audio_url)
    audio_file = open(temp_file_path, "rb")
    whisper_client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    transcript = whisper_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    word_times: List[Dict[str, Dict[str, str]]] = [
        {"word": word.word, "start": word.start, "end": word.end}
        for word in transcript.words
    ]

    return transcript.text, word_times


# Finally, we define a local entrypoint
@app.local_entrypoint()
def main(audio_url: str, output_text: str, output_path: str) -> None:
    # Parse output_path and create parent directory if needed
    output_path_obj: Path = Path(output_path)
    input_text: str
    word_times: List[Dict[str, Dict[str, str]]]

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    input_text, word_times = run_asr.remote(audio_url)
    playdiffusion_model: PlayDiffusionModel = PlayDiffusionModel()
    output_audio: bytes = playdiffusion_model.generate.remote(
        audio_url, input_text, output_text, word_times
    )

    # Save the output audio to the specified path
    with open(output_path, "wb") as f:
        f.write(output_audio)


# Example command line invocation:
# `modal run playdiffusion-model.py --audio-url "https://modal-public-assets.s3.us-east-1.amazonaws.com/mono_44100_127389__acclivity__thetimehascome.wav" --output-text "November, '9 PM. I'm standing in alley. After waiting several hours, the time has come. A man with long dark hair approaches. I have to act and fast before he realizes what has happened. I must find out." --output-path "/tmp/playdiffusion/output.wav"`


# Some utility functions
def write_to_tempfile(audio_url: str) -> Tuple[bytes, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Download and write the audio to the temporary file
        audio_bytes: bytes = urlopen(audio_url).read()
        temp_file.write(audio_bytes)
        temp_file_path: str = temp_file.name
    return temp_file_path
