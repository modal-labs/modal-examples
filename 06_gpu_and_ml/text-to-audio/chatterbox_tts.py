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

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "chatterbox-tts==0.1.6",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
)

# We'll also use Chatterbox's provided set of voice prompts which you can download [here](https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip).
# Unzip the file and upload it to a `modal.Volume` called `chatterbox-tts-voices` with the following CLI commands:
# ```shell
# modal volume create chatterbox-tts-voices
# modal volume put chatterbox-tts-voices <PATH-TO-UNZIPPED-VOICE-PROMPTS-DIRECTORY>
# ```
# Now we can instantiate the volume and use it with our app.

chatterbox_tts_voices_vol = modal.Volume.from_name("chatterbox-tts-voices")
VOICE_PROMPTS_DIR = "/chatterbox-tts/prompts"

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

# We'll also need to provide a Hugging Face token using a `modal.Secret` to access the model weights,
# and attach the `chatterbox-tts-voices` volume to the container.


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOICE_PROMPTS_DIR: chatterbox_tts_voices_vol},
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
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
            audio_prompt_path=VOICE_PROMPTS_DIR
            + "/chatterbox-tts-voices"
            + "/prompts"
            + "/Lucy.wav",
        )

        # Convert the waveform to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()


@app.local_entrypoint()
def test():
    chatterbox = Chatterbox()
    audio_bytes = chatterbox.generate.remote(
        prompt="Chatterbox running on Modal [chuckle]."
    )

    # Save the audio bytes to a file
    import pathlib

    output_path = pathlib.Path("/tmp/chatterbox-tts/output.wav")
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
