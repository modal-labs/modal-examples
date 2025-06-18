# ---
# output-directory: "/tmp/playdiffusion"
# lambda-test: false
# cmd: ["modal", "serve", "06_gpu_and_ml/audio-editing/playdiffusion.py"]
# ---


# # Create a PlayDiffusion API on Modal

# This example demonstrates how to deploy a PlayDiffusion audio editing API using the PlayDiffusion model on Modal.
# The API accepts text prompts and input audio as WAV files and returns generated audio as WAV files through a FastAPI endpoint.
# We use Modal's class-based approach with GPU acceleration to provide fast, scalable inference.

# ## Setup

# Import the necessary modules for Modal deployment and TTS functionality.

import io
import tempfile
import modal

# ## Define a container image

# We start with Modal's baseline `debian_slim` image and install the required packages.
# - `fastapi`: Web framework for creating the API endpoint
AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"

# Needs to be 3.11 https://github.com/playht/PlayDiffusion/blob/d3995b9e2cd8a80b88be6aeeb4e35fd282b2d255/pyproject.toml
image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("fastapi[standard]", "openai").run_commands(
    "pip install git+https://github.com/playht/PlayDiffusion.git@d3995b9e2cd8a80b88be6aeeb4e35fd282b2d255"
)
app = modal.App("playdiffusion-api-example", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing and the TTS model itself.

with image.imports():
    import io
    import soundfile as sf
    from playdiffusion import PlayDiffusion, InpaintInput
    from fastapi.responses import StreamingResponse
    from urllib.request import urlopen
    from openai import OpenAI
    import os



# ## The model class

# The service is implemented using Modal's class syntax with GPU acceleration.
# We configure the class to use an A10G GPU with additional parameters:
# #
# - `scaledown_window=60 * 5`: Keep containers alive for 5 minutes after last request
# - `enable_memory_snapshot=True`: Enable [memory snapshots](https://modal.com/docs/guide/memory-snapshot) to optimize cold boot times
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per container


@app.cls(gpu="a10g", scaledown_window=60 * 5, enable_memory_snapshot=True)
@modal.concurrent(max_inputs=10)
class Model:
    @modal.enter()
    def load(self):
        self.inpainter = PlayDiffusion()

    @modal.method()
    def generate(self, audio_url: str, input_text: str, output_text: str, word_times): 
        # Create a temporary file to store the audio
        audio_bytes, temp_file_path = write_to_tempfile(audio_url)
        
        # Get the audio data and sample rate from inpainter
        sample_rate, audio_data = self.inpainter.inpaint(
            InpaintInput(
                input_text=input_text,
                output_text=output_text,
                input_word_times = word_times,
                audio=temp_file_path
            )
        )
        
        # Create an in-memory buffer
        buffer = io.BytesIO()
        
        # Write the audio data to the buffer as WAV
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        
        # Reset buffer position to beginning
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav"
        )

@app.function(secrets=[modal.Secret.from_name("openai-secret")])
def run_asr(audio_url):
    audio_bytes, _ = write_to_tempfile(audio_url)

    whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    audio_file = open(audio_bytes, "rb")
    transcript = whisper_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    word_times = [{
        "word": word.word,
        "start": word.start,
        "end": word.end
    } for word in transcript.words]

    return transcript.text, word_times


@app.local_entrypoint()
def main(audio_url, output_text, output_path):
    input_text, word_times = run_asr.remote(audio_url)
    playdiffusion_model = Model()
    output_audio = playdiffusion_model.generate(audio_url, input_text, output_text, word_times)
    # Create output directory if it doesn't exist
    os.makedirs("/tmp/playdiffusion", exist_ok=True)
    
    # Save the output audio to the specified path
    output_path = os.path.join("/tmp/playdiffusion", "output.wav")
    with open(output_path, "wb") as f:
        f.write(output_audio)


def write_to_tempfile(audio_url):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        # Download and write the audio to the temporary file
        audio_bytes = urlopen(audio_url).read()
        temp_file.write(audio_bytes)
        temp_file.flush()  # Ensure data is written to disk
        temp_file_path = temp_file.name
    
    return audio_bytes, temp_file_path


# Now deploy the PlayDiffusion API with:
#
# ```shell
# modal deploy play_diffusion.py
# ```
#
# And query the endpoint with:
#
# ```shell
# mkdir -p /tmp/playdiffusion  # create tmp directory

# We need to generate the word level timestamps and transcript of the modal


#
# curl -X POST --get https://modal-labs-advay-dev--playdiffusion-api-example-model-generate.modal.run \
#   --data-urlencode "audio_url=https://modal-public-assets.s3.us-east-1.amazonaws.com/mono_44100_127389__acclivity__thetimehascome.wav" \
#   --data-urlencode "output_text=November, '9 PM. I'm standing in  alley. After waiting several hours, the time has come. A man with long dark hair approaches. I have to act and fast before he realizes what has happened. I must find out." \
#   --output /tmp/playdiffusion/output.wav
# ```
#
# You'll receive a WAV file named `/tmp/playdiffusion/output.wav` containing the generated audio.