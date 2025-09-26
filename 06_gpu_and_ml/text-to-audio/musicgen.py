# # Create music with ACE-Step

# In this example, we show you how you can run [Ace Studio](https://acestudio.ai/)'s
# [ACE-Step](https://github.com/ace-step/ACE-Step) music generation model
# on Modal.

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
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "git+https://github.com/ace-step/ACE-Step.git@6ae0852b1388de6dc0cca26b31a86d711f723cb3",  # we can install directly from GitHub!
    )
)

# In addition to source code, we'll also need the model weights.

# ACE-Step integrates with the Hugging Face ecosystem, so setting up the models
# is straightforward. `ACEStepPipeline` internally uses the Hugging Face model hub
# to download the weights if not already present.


def load_model(and_return=False):
    from acestep.pipeline_ace_step import ACEStepPipeline

    model_demo = ACEStepPipeline(dtype="bfloat16", cpu_offload=False, overlapped_decode=True)
    if and_return:
        return model_demo


# But Modal Functions are serverless: instances spin down when they aren't being used.
# If we want to avoid downloading the weights every time we start a new instance,
# we need to store the weights somewhere besides our local filesystem.

# So we add a Modal [Volume](https://modal.com/docs/guide/volumes)
# to store the weights in the cloud. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

cache_dir = "/root/.cache/ace-step/checkpoints"
model_cache = modal.Volume.from_name("ACE-Step-model-cache", create_if_missing=True)

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

web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]==0.115.4", "gradio==4.44.1"
)

# This is a totally different environment from the one we run our model in.
# Say goodbye to Python dependency conflict hell!

# ## Running music generation on Modal

# Now, we write our music generation logic.

# - We make an [App](https://modal.com/docs/guide/apps) to organize our deployment.
# - We load the model at start, instead of during inference, with `modal.enter`,
# which requires that we use a Modal [`Cls`](https://modal.com/docs/guide/lifecycle-functions).
# - In the `app.cls` decorator, we specify the Image we built and attach the Volume.
# We also pick a GPU to run on -- here, an NVIDIA L40S.

app = modal.App("example-musicgen")


@app.cls(gpu="l40s", image=image, volumes={cache_dir: model_cache})
class MusicGen:
    @modal.enter()
    def init(self):
        from acestep.pipeline_ace_step import ACEStepPipeline

        self.model: ACEStepPipeline = load_model(and_return=True)

    @modal.method()
    def generate(
        self,
        prompt: str,
        lyrics: str,
        duration: int = 10,
        format: str = "wav",  # or mp3
        manual_seeds: Optional[int] = 508630535
    ) -> bytes:
        import uuid

        output_path = f"/dev/shm/output_{uuid.uuid4().hex}.{format}"
        print("Generating music...")
        self.model(
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=60,
            guidance_scale=15,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10,
            guidance_interval=0.5,
            guidance_interval_decay=0,
            min_guidance_scale=3,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            save_path=output_path,
            manual_seeds=manual_seeds,
        )
        return Path(output_path).read_bytes()


# We can then generate music from anywhere by running code like what we have in the `local_entrypoint` below.


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    duration: float = 10,
    format: str = "wav",  # or mp3
):
    if prompt is None:
        prompt = "aggressive, Heavy Riffs, Blast Beats, Satanic Black Metal"
    if lyrics is None:
        lyrics = """[verse]
Floating through the galaxy on a midnight ride
Stars are dancing all around in cosmic tides
Feel the pulse of space and time beneath our feet
Every beat a heartbeat in this endless suite

[chorus]
Galactic dreams under neon lights
Sailing through the velvet nights
We are echoes in a cosmic sea
In a universe where we are free

[verse]
Planetary whispers in the sky tonight
Every constellation's got a secret sight
Distant worlds and moons we have yet to see
In the void of space where we can just be

[bridge]
Asteroids and comets in a ballet they spin
Lost in the rhythm of where our dreams begin
Close your eyes and let the synths take flight
We're voyagers on an electric night

[verse]
Let the piano keys unlock the stars above
Every chord a memory every note is love
In this synth symphony we find our grace
Drifting forever in this boundless space

[chorus]
Galactic dreams under neon lights
Sailing through the velvet nights
We are echoes in a cosmic sea
In a universe where we are free"""
    print(
        f"🎼 generating {duration} seconds of music from prompt '{prompt[:32] + ('...' if len(prompt) > 32 else '')}'"
        f"and lyrics '{lyrics[:32] + ('...' if len(lyrics) > 32 else '')}'"
    )

    musicgen = MusicGen()
    clip = musicgen.generate.remote(prompt, lyrics, duration=duration, format=format)

    dir = Path("/tmp/musicgen")
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


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )
