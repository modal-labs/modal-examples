# ---
# output-directory: "/tmp/chatterbox-tts"
# lambda-test: false
# cmd: ["modal", "serve", "06_gpu_and_ml/test-to-audio/chatterbox_tts.py"]
# ---


# # Create a Chatterbox TTS API on Modal

# This example demonstrates how to deploy a text-to-speech (TTS) API using the Chatterbox TTS model on Modal.
# The API accepts text prompts and returns generated audio as WAV files through a FastAPI endpoint.
# We use Modal's class-based approach with GPU acceleration to provide fast, scalable TTS inference.

# ## Setup

# Import the necessary modules for Modal deployment and TTS functionality.

import io

import modal

# ## Define a container image

# We start with Modal's baseline `debian_slim` image and install the required packages.
# - `chatterbox-tts`: The TTS model library
# - `fastapi`: Web framework for creating the API endpoint

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "chatterbox-tts==0.1.1", "fastapi[standard]"
)
app = modal.App("example-chatterbox-tts", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing and the TTS model itself.

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import StreamingResponse

# ## The TTS model class

# The TTS service is implemented using Modal's class syntax with GPU acceleration.
# We configure the class to use an A10G GPU with additional parameters:

# - `scaledown_window=60 * 5`: Keep containers alive for 5 minutes after last request
# - `enable_memory_snapshot=True`: Enable [memory snapshots](https://modal.com/docs/guide/memory-snapshot) to optimize cold boot times
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per container


@app.cls(gpu="a10g", scaledown_window=60 * 5, enable_memory_snapshot=True)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str):
        # Generate audio waveform from the input text
        wav = self.model.generate(prompt)

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


# Now deploy the Chatterbox API with:
#
# ```shell
# modal deploy chatterbox_tts.py
# ```
#
# And query the endpoint with:
#
# ```shell
# mkdir -p /tmp/chatterbox-tts  # create tmp directory
#
# curl -X POST --get "<YOUR-ENDPOINT-URL>" \
#   --data-urlencode "prompt=Chatterbox running on Modal"
#   --output /tmp/chatterbox-tts/output.wav
# ```
#
# You'll receive a WAV file named `/tmp/chatterbox-tts/output.wav` containing the generated audio.
#
# This app takes about 30 seconds to cold boot, mostly dominated by loading
# the Chatterbox model into GPU memory. It takes 2-3s to generate a 5s audio clip.
