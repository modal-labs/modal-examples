# ---
# output-directory: "/tmp/chatterbox-tts"
# lambda-test: false
# cmd: ["modal", "serve", "-m", "06_gpu_and_ml.text-to-audio.chatterbox_tts"]
# ---

# # Create a Chatterbox TTS API on Modal

# This example demonstrates how to deploy a text-to-speech (TTS) API using the open source model, Chatterbox Turbo, on Modal.

# Chatterbox Turbo is a state-of-the-art TTS model that can generate natural, expressive speech that rivals proprietary models.
# Prompts can include paralinguistic tags like `[chuckle]`, `[sigh]`, and `[gasp]`. Chatterobx also support voice cloning by passing
# a short (about 10 seconds) audio prompt of the target voice.
#
# Check out[Resemble AI's website](https://www.resemble.ai/) or
# the [Chatterbox Github](https://github.com/resemble-ai/chatterbox) repo for more details.

# ## Setup

# Import the necessary modules for Modal deployment and TTS functionality.

import io
from pathlib import Path

import modal

# ## Define a container image

# We start with Modal's baseline `debian_slim` image and install the required packages.
# - `chatterbox-tts`: The TTS model library
# - `fastapi`: Web framework for creating the API endpoint

PROJECT_DIR = Path(__file__).parent

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "chatterbox-tts",
    "fastapi[standard]",
    "peft",
)

app = modal.App("example-chatterbox-turbo", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing and the TTS model itself.

with image.imports():
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from fastapi.responses import StreamingResponse

# ## The TTS model class

# The TTS service is implemented using Modal's class syntax with GPU acceleration.
# We configure the class to use an A10G GPU with additional parameters:

# - `scaledown_window=60 * 5`: Keep containers alive for 5 minutes after last request
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per container

# We'll also need to provide the Hugging Face token using a `modal.Secret` to access the model weights.


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=1,
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str):
        # Generate audio waveform from the input text
        wav = self.model.generate(
            prompt, audio_prompt_path=PROJECT_DIR / "prompts" / "Lucy.wav"
        )

        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        # Uses the model's sample rate and WAV format
        ta.save(buffer, wav, self.model.sr, format="wav")

        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
        )


# Now deploy the Chatterbox API with from the repo root directory:
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
#
# This app takes about 30 seconds to cold boot, mostly dominated by loading
# the Chatterbox model into GPU memory. It takes 2-3s to generate a 5s audio clip.
