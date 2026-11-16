# ---
# output-directory: "/tmp/chatterbox-tts"
# ---

# # Create a Chatterbox TTS API on Modal

# This example demonstrates how to deploy a text-to-speech (TTS) API using the open source model Chatterbox Turbo on Modal.

# Chatterbox Turbo is a state-of-the-art TTS model that can generate natural, expressive speech that rivals proprietary models.
# Prompts can include paralinguistic tags like `[chuckle]`, `[sigh]`, and `[gasp]`. Chatterbox also support voice cloning by passing
# a short (about 10 seconds) audio prompt of the target voice.
#
# Check out [Resemble AI's website](https://www.resemble.ai/) or
# the [Chatterbox Github](https://github.com/resemble-ai/chatterbox) repo for more details.

# ## Setup

# Import `modal`, the only required local dependency.

import modal

# ## Define a container image

# We start with Modal's baseline `debian_slim` image and install the required packages.
# - `chatterbox-tts`: The TTS model library
# - `fastapi`: Web framework for creating the API endpoint
# - "peft": Required for properly loading the model

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "chatterbox-tts==0.1.7",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
    "setuptools==81.0.0",
)

# We'll download Chatterbox's public [voice prompts](https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip)
# to the container image during its construction.

VOICE_PROMPT_PATH = "/chatterbox-tts/prompts/Lucy.wav"


def download_voice_prompt():
    import io
    import pathlib
    import urllib.request
    import zipfile

    with urllib.request.urlopen(
        "https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip", timeout=60
    ) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            prompts = [
                name
                for name in archive.namelist()
                if pathlib.PurePosixPath(name).name == "Lucy.wav"
            ]
            if len(prompts) != 1:
                raise ValueError(
                    f"Expected one Lucy.wav voice prompt, found {len(prompts)}"
                )
            path = pathlib.Path(VOICE_PROMPT_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(prompts[0]))


image = image.run_function(download_voice_prompt)

app = modal.App("example-chatterbox-tts", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing modules and the Chatterbox TTS module itself.

with image.imports():
    import io

    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from fastapi.responses import StreamingResponse

# ## The TTS model class

# The TTS service is implemented using Modal's class syntax with GPU acceleration.
# We configure the class to use an A10G GPU with additional parameters:

# - `scaledown_window=60 * 5`: Keep containers alive for 5 minutes after last request
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per container


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def api_endpoint(self, prompt: str):
        # Get the audio bytes from the generate method
        audio_bytes = self.generate.local(prompt)

        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
        )

    @modal.method()
    def generate(self, prompt: str) -> bytes:
        # Generate audio waveform from the input text
        wav = self.model.generate(
            prompt,
            audio_prompt_path=VOICE_PROMPT_PATH,
        )

        # Convert the waveform to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()


@app.local_entrypoint()
def test(
    prompt: str = "Chatterbox running on Modal [chuckle].",
    output_path: str = "/tmp/chatterbox-tts/output.wav",
):
    chatterbox = Chatterbox()
    audio_bytes = chatterbox.generate.remote(prompt=prompt)

    # Save the audio bytes to a file
    import pathlib

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    print(f"Audio saved to {output_path}")


# Now deploy the Chatterbox API from this file's directory:
#
# ```shell
# modal deploy -m 06_gpu_and_ml.text-to-audio.chatterbox_tts
# ```
#
# And query the endpoint with:
#
# ```shell
# mkdir -p /tmp/chatterbox-tts  # create tmp directory
#
# curl -X POST --get "<YOUR-ENDPOINT-URL>" \
#   --data-urlencode "prompt=Chatterbox running on Modal [chuckle]." \
#   --output /tmp/chatterbox-tts/output.wav
# ```
#
# You'll receive a WAV file named `/tmp/chatterbox-tts/output.wav` containing the generated audio.
