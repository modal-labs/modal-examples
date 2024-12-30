# ---
# deploy: true
# ---

# # Create your own music samples with MusicGen

# [MusicGen](https://github.com/facebookresearch/audiocraft) is the latest
# milestone language model in conditional music generation, with great results. In
# this example, we show you how you can run MusicGen on modal.

# We use [Audiocraft](https://github.com/facebookresearch/audiocraft), a PyTorch
# library that provides the code and models for MusicGen, and load both the 3.3B
# `large` and 1.5B `melody` models to use depending on the user input (large for
# just text, melody for text and melody). We can install all our dependencies and
# “bake” both models into our image to avoid downloading our models during
# inference and take advantage of Modal's incredibly fast cold-start times:

# ## Setting up the image and dependencies

from modal import Image, App, method, gpu, enter, asgi_app
from pathlib import Path
import io

app = App("musicgen")

MAX_SEGMENT_DURATION = 30


def download_models() -> None:
    from audiocraft.models import MusicGen
    MusicGen.get_pretrained("large")
    MusicGen.get_pretrained("melody")

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "soundfile",
        "pydub",
        "git+https://github.com/facebookresearch/audiocraft.git",
    )
    .run_function(download_models, gpu="any")
)

with image.imports():
    import torch
    import torchaudio

# ## Defining the model generation
# We then write our model code within Modal's
# [`@app.cls`](/docs/reference/modal.App#cls) decorator, with the
# [`generate`] function processing the user input and generating audio as bytes that we can
# save to a file later.

@app.cls(gpu=gpu.A10G(), image = image)
class Audiocraft:
    @enter()
    def init(self):
        from audiocraft.models import MusicGen

        self.model_large = MusicGen.get_pretrained("large")
        self.model_melody = MusicGen.get_pretrained("melody")

    # modified audiocraft.audio_write() to return bytes
    def audio_write_to_bytes(
        self,
        wav,
        sample_rate: int,
        format: str = "wav",
        normalize: bool = True,
        strategy: str = "peak",
        peak_clip_headroom_db: float = 1,
        rms_headroom_db: float = 18,
        loudness_headroom_db: float = 14,
        log_clipping: bool = True,
    ) -> io.BytesIO:
        from audiocraft.data.audio_utils import i16_pcm, normalize_audio
        import soundfile as sf
        from pydub import AudioSegment

        assert wav.dtype.is_floating_point, "wav is not floating point"
        if wav.dim() == 1:
            wav = wav[None]
        elif wav.dim() > 2:
            raise ValueError("Input wav should be at most 2 dimension.")
        assert wav.isfinite().all()

        wav = normalize_audio(
            wav,
            normalize,
            strategy,
            peak_clip_headroom_db,
            rms_headroom_db,
            loudness_headroom_db,
            log_clipping=log_clipping,
            sample_rate=sample_rate,
        )

        wav = i16_pcm(wav)
        wav_np = wav.numpy()
        audio = AudioSegment(
            data=wav_np.tobytes(),
            sample_width=wav_np.dtype.itemsize,
            frame_rate=sample_rate,
            channels=1,
        )

        audio_bytes = io.BytesIO()

        audio.export(audio_bytes, format=format)
        audio_bytes.seek(0)
        return audio_bytes

    def load_and_clip_melody(self, url: str):
        import requests

        # check file format
        _, file_extension = url.rsplit(".", 1)
        if file_extension.lower() not in ["mp3", "wav"]:
            raise ValueError(f"Invalid file format. Only .mp3 and .wav are supported.")

        _, filepath = url.rsplit("/", 1)
        response = requests.get(url)

        # checking if the request was successful (status code 200)
        if response.status_code == 200:
            with open(filepath, "wb") as file:
                file.write(response.content)
            print("File downloaded successfully.")
        else:
            raise Exception(f"Error: {response.status_code} - {response.reason}")
        melody_waveform, sr = torchaudio.load(filepath)

        # checking duration of audio and clipping to first 30 secs if too long
        melody_duration = melody_waveform.size(1) / sr
        if melody_duration > MAX_SEGMENT_DURATION:
            melody_waveform = melody_waveform[:, : MAX_SEGMENT_DURATION * sr]

        return melody_waveform, sr

    @method()
    def generate(
        self,
        prompt: str,
        duration: int = 10,
        format: str = "wav",
        melody_url: str = "",
    ):
        output = None
        segment_duration = (
            MAX_SEGMENT_DURATION if duration > MAX_SEGMENT_DURATION else duration
        )
        overlap = 10

        if melody_url != "":
            model = self.model_melody
            melody_waveform, sr = self.load_and_clip_melody(melody_url)
            self.model_melody.set_generation_params(
                duration=min(segment_duration, MAX_SEGMENT_DURATION)
            )
            output = self.model_melody.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody_waveform.unsqueeze(0),
                melody_sample_rate=sr,
                progress=True,
            )
            duration -= segment_duration
        else:
            model = self.model_large
            sr = self.model_large.sample_rate

        # looping to generate duration longer than model max of 30 secs
        while duration > 0:
            if output is not None:
                if (duration + overlap) < MAX_SEGMENT_DURATION:
                    segment_duration = duration + overlap
                else:
                    segment_duration = MAX_SEGMENT_DURATION

            model.set_generation_params(
                duration=min(segment_duration, MAX_SEGMENT_DURATION)
            )

            if output is None:  # generate first chunk
                next_segment = model.generate(descriptions=[prompt])
                duration -= segment_duration
            else:
                last_chunk = output[:, :, -overlap * sr :]
                next_segment = model.generate_continuation(
                    last_chunk, sr, descriptions=[prompt]
                )
                duration -= segment_duration - overlap

            if output is None:
                output = next_segment
            else:
                output = torch.cat(
                    [
                        output[:, :, : -overlap * sr],
                        next_segment,
                    ],
                    2,
                )

        output = output.detach().cpu().float()[0]
        clip = self.audio_write_to_bytes(
            output, model.sample_rate, strategy="loudness", format=format
        )
        melody_clip = (
            self.audio_write_to_bytes(
                melody_waveform[0], sr, strategy="loudness", format=format
            )
            if melody_url != ""
            else None
        )

        return clip, melody_clip


# We can trigger MusicGen inference from our local machine by running the code in the local entrypoint below.

@app.local_entrypoint()
def main(prompt: str, duration: int = 10, format: str = "wav", melody_url: str = ""):
    dir = Path("/tmp/audiocraft")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    audiocraft = Audiocraft()
    melody_clip, clip = audiocraft.generate.remote(
        prompt, duration=duration, format=format, melody_url=melody_url
    )

    if melody_clip:
        output_path = dir / f"melody_clip.{format}"
        print(f"Saving to {output_path}")
        with open(output_path, "wb") as f:
            f.write(melody_clip.read())

    output_path = dir / f"output.{format}"
    print(f"Saving to {output_path}")
    with open(output_path, "wb") as f:
        f.write(clip.read())

# You can trigger it with:
# ``` shell
# modal run music_gen.py --prompt="metallica meets sabrina carpenter"
# ```
# and optionally pass in a melody and a format



# ## A hosted Gradio interface

# With the Gradio library, we can create a simple web interface around our class in Python, then use Modal to host it for anyone to try out.
# To deploy your own, run

# ``` shell
# modal deploy music_gen.py
# ```

web_image = image.pip_install(
    "fastapi[standard]==0.115.4",
    "gradio==4.44.1",
)


@app.function(
    image=web_image,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 1000 concurrent inputs
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
@asgi_app()
def ui():
    import uuid

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    # Since this Gradio app is running from its own container,
    # allowing us to run the inference service via .remote() methods.
    model = Audiocraft()

    # Create a temporary directory for audio files
    temp_dir = Path("/tmp/audiocraft")
    temp_dir.mkdir(exist_ok=True, parents=True)

    def generate_music(prompt: str, duration: int = 10, format: str = "wav", melody_url: str = ""):
        clip_audio_bytes, melody_clip_audio_bytes = model.generate.remote(prompt, duration, format, melody_url)
        
        # Create a unique filename for this generation
        clip_file = f"{temp_dir}/generated_music_{uuid.uuid4()}.{format}"
        # Save bytes to temporary file that Gradio can serve
        with open(clip_file, "wb") as f:
            f.write(clip_audio_bytes.read())


        melody_clip_file = None
        if melody_clip_audio_bytes is not None:
            melody_clip_file = f"{temp_dir}/melody_clip{uuid.uuid4()}.{format}"
            with open(melody_clip_file, "wb") as f:
                f.write(melody_clip_audio_bytes.read())
            
        return clip_file, melody_clip_file

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# MusicGen")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                duration = gr.Number(label="Duration (seconds)", value=10, minimum=1, maximum=30)
                format = gr.Radio(["wav", "mp3"], label="Format", value="wav")
                melody_url = gr.Text(label="Optional Melody URL", placeholder="Enter URL to melody audio file (.mp3 or .wav)", value="")
                btn = gr.Button("Generate")
            with gr.Column():
                clip_output = gr.Audio(label="Generated Music", autoplay=True)
                melody_clip_output = gr.Audio(label="Melody Clip", autoplay=True, visible= (melody_url.value != ""))
        
        btn.click(
            generate_music,
            inputs=[prompt, duration, format, melody_url],
            outputs=[clip_output, melody_clip_output]
        )

    return mount_gradio_app(app=web_app, blocks=demo, path="/")
















