# # Create your own music samples with MusicGen

# MusicGen is a popular open-source music-generation model family from Meta.
# In this example, we show you how you can run MusicGen models on Modal GPUs,
# along with a Gradio UI for playing around with the model.

# We use [Audiocraft](https://github.com/facebookresearch/audiocraft),
# the inference library released by Meta
# for MusicGen and its kin, like AudioGen.

# ## Setting up dependencies

from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# We start by defining the environment our generation runs in.
# This takes some explaining since, like most cutting-edge ML environments, it is a bit fiddly.

# This environment is captured by a
# [container image](https://modal.com/docs/guide/custom-container),
# which we build step-by-step by calling methods to add dependencies,
# like `apt_install` to add system packages and `pip_install` to add
# Python packages.

# Note that we don't have to install anything with "CUDA"
# in the name -- the drivers come for free with the Modal environment
# and the rest gets installed `pip`. That makes our life a lot easier!
# If you want to see the details, check out [this guide](https://modal.com/docs/guide/gpu)
# in our docs.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "huggingface_hub[hf_transfer]==0.27.1",  # speed up model downloads
        "torch==2.1.0",  # version pinned by audiocraft
        "numpy<2",  # defensively cap the numpy version
        "git+https://github.com/facebookresearch/audiocraft.git@v1.3.0",  # we can install directly from GitHub!
    )
)

# In addition to source code, we'll also need the model weights.

# Audiocraft integrates with the Hugging Face ecosystem, so setting up the models
# is straightforward -- the same `get_pretrained` method we use to load the weights for execution
# will also download them if they aren't present.


def load_model(and_return=False):
    from audiocraft.models import MusicGen

    model_large = MusicGen.get_pretrained("facebook/musicgen-large")
    if and_return:
        return model_large


# But Modal Functions are serverless: instances spin down when they aren't being used.
# If we want to avoid downloading the weights every time we start a new instance,
# we need to store the weights somewhere besides our local filesystem.

# So we add a Modal [Volume](https://modal.com/docs/guide/volumes)
# to store the weights in the cloud. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

cache_dir = "/cache"
model_cache = modal.Volume.from_name("audiocraft-model-cache", create_if_missing=True)

# We don't need to change any of the model loading code --
# we just need to make sure the model gets stored in the right directory.

# To do that, we set an environment variable that Hugging Face expects
# (and another one that speeds up downloads, for good measure)
# and then run the `load_model` Python function.

image = image.env(
    {"HF_HUB_CACHE": cache_dir, "HF_HUB_ENABLE_HF_TRANSER": "1"}
).run_function(load_model, volumes={cache_dir: model_cache})

# While we're at it, let's also define the environment for our UI.
# We'll stick with Python and so use FastAPI and Gradio.

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.4", "gradio==4.44.1"
)

# This is a totally different environment from the one we run our model in.
# Say goodbye to Python dependency conflict hell!

# ## Running music generation on Modal

# Now, we write our music generation logic.
# This is bit complicated because we want to support generating long samples,
# but the model has a maximum context length of thirty seconds.
# We can get longer clips by feeding the model's output back as input,
# auto-regressively, but we have to write that ourselves.

# There are also a few bits to make this work well with Modal:

# - We make an [App](https://modal.com/docs/guide/apps) to organize our deployment.
# - We load the model at start, instead of during inference, with `modal.enter`,
# which requires that we use a Modal [`Cls`](https://modal.com/docs/guide/lifecycle-functions).
# - In the `app.cls` decorator, we specify the Image we built and attach the Volume.
# We also pick a GPU to run on -- here, an NVIDIA L40S.

app = modal.App("example-musicgen")
MAX_SEGMENT_DURATION = 30  # maximum context window size


@app.cls(gpu="l40s", image=image, volumes={cache_dir: model_cache})
class MusicGen:
    @modal.enter()
    def init(self):
        self.model = load_model(and_return=True)

    @modal.method()
    def generate(
        self,
        prompt: str,
        duration: int = 10,
        overlap: int = 10,
        format: str = "wav",  # or mp3
    ) -> bytes:
        f"""Generate a music clip based on the prompt.

        Clips longer than the MAX_SEGMENT_DURATION of {MAX_SEGMENT_DURATION}s
        are generated by clipping all but `overlap` seconds and running inference again."""
        context = None
        overlap = min(overlap, MAX_SEGMENT_DURATION - 1)
        remaining_duration = duration

        if remaining_duration < 0:
            return bytes()

        while remaining_duration > 0:
            # calculate duration of the next segment
            segment_duration = remaining_duration
            if context is not None:
                segment_duration += overlap

            segment_duration = min(segment_duration, MAX_SEGMENT_DURATION)

            # generate next segment
            generated_duration = (
                segment_duration if context is None else (segment_duration - overlap)
            )
            print(f"🎼 generating {generated_duration} seconds of music")
            self.model.set_generation_params(duration=segment_duration)
            next_segment = self._generate_next_segment(prompt, context, overlap)

            # update remaining duration
            remaining_duration -= generated_duration

            # combine with previous segments
            context = self._combine_segments(context, next_segment, overlap)

        output = context.detach().cpu().float()[0]

        return to_audio_bytes(
            output,
            self.model.sample_rate,
            format=format,
            # for more on audio encoding parameters, see the docs for audiocraft
            strategy="loudness",
            loudness_compressor=True,
        )

    def _generate_next_segment(self, prompt, context, overlap):
        """Generate the next audio segment, either fresh or as continuation of a context."""
        if context is None:
            return self.model.generate(descriptions=[prompt])
        else:
            overlap_samples = overlap * self.model.sample_rate
            last_chunk = context[:, :, -overlap_samples:]  # B, C, T
            return self.model.generate_continuation(
                last_chunk, self.model.sample_rate, descriptions=[prompt]
            )

    def _combine_segments(self, context, next_segment, overlap: int):
        """Combine context with next segment, handling overlap."""
        import torch

        if context is None:
            return next_segment

        # Calculate where to trim the context (removing overlap)
        overlap_samples = overlap * self.model.sample_rate
        context_trimmed = context[:, :, :-overlap_samples]  # B, C, T

        return torch.cat([context_trimmed, next_segment], dim=2)


# We can then generate music from anywhere by running code like what we have in the `local_entrypoint` below.


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    duration: int = 10,
    overlap: int = 15,
    format: str = "wav",  # or mp3
):
    if prompt is None:
        prompt = "Amapiano polka, klezmers, log drum bassline, 112 BPM"
    print(
        f"🎼 generating {duration} seconds of music from prompt '{prompt[:64] + ('...' if len(prompt) > 64 else '')}'"
    )

    audiocraft = MusicGen()
    clip = audiocraft.generate.remote(prompt, duration=duration, format=format)

    dir = Path("/tmp/audiocraft")
    dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / f"{slugify(prompt)[:64]}.{format}"
    print(f"🎼 Saving to {output_path}")
    output_path.write_bytes(clip)


# You can execute it with a command like:

# ``` shell
# modal run musicgen.py --prompt="Baroque boy band, Bachstreet Boys, basso continuo, Top 40 pop music" --duration=60
# ```

# ## Hosting a web UI for the music generator

# With the Gradio library, we can create a simple web UI in Python
# that calls out to our music generator,
# then host it on Modal for anyone to try out.

# To deploy both the music generator and the UI, run

# ``` shell
# modal deploy musicgen.py
# ```

# Share the URL with your friends and they can generate their own songs!


@app.function(
    image=web_image,
    # Gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 1000 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    api = FastAPI()

    # Since this Gradio app is running from its own container,
    # we make a `.remote` call to the music generator
    model = MusicGen()
    generate = model.generate.remote

    temp_dir = Path("/dev/shm")

    async def generate_music(prompt: str, duration: int = 10, format: str = "wav"):
        audio_bytes = await generate.aio(prompt, duration=duration, format=format)

        audio_path = temp_dir / f"{uuid4()}.{format}"
        audio_path.write_bytes(audio_bytes)

        return audio_path

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# MusicGen")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                duration = gr.Number(
                    label="Duration (seconds)", value=10, minimum=1, maximum=300
                )
                format = gr.Radio(["wav", "mp3"], label="Format", value="wav")
                btn = gr.Button("Generate")
            with gr.Column():
                clip_output = gr.Audio(label="Generated Music", autoplay=True)

        btn.click(
            generate_music,
            inputs=[prompt, duration, format],
            outputs=[clip_output],
        )

    return mount_gradio_app(app=api, blocks=demo, path="/")


# ## Addenda

# The remainder of the code here is not directly related to Modal
# or to music generation, but is used in the example above.


def to_audio_bytes(wav, sample_rate: int, **kwargs) -> bytes:
    from audiocraft.data.audio import audio_write

    # audiocraft provides a nice utility for converting waveform tensors to audio,
    # but it saves to a file path. here, we create a file path that is actually
    # just backed by memory, instead of disk, to save on some latency

    shm = Path("/dev/shm")  # /dev/shm is a memory-backed filesystem
    stem_name = shm / str(uuid4())

    output_path = audio_write(stem_name, wav, sample_rate, **kwargs)

    return output_path.read_bytes()


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )
