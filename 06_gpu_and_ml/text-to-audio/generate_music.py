# # Make music with ACE-Step 1.5

# In this example, we show you how you can run [ACE Studio](https://acestudio.ai/)'s
# [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) music generation model
# on Modal.

# ACE-Step 1.5 introduces a multi-model architecture:
# a DiT (Diffusion Transformer) handler for audio generation
# and an LM (Language Model) handler for prompt augmentation.
# The LM automatically enhances prompts, detects language,
# and generates metadata like BPM and key.

# We'll set up both a serverless music generation service
# and a web user interface.

# ## Setting up dependencies

from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# We start by defining the environment our generation runs in.
# This takes some explaining since, like most cutting-edge ML environments, it is a bit fiddly.

# This environment is captured by a
# [container image](https://modal.com/docs/guide/images),
# which we build step-by-step by calling methods to add dependencies,
# like `apt_install` to add system packages and `uv_pip_install` to add
# Python packages.

# ACE-Step 1.5 uses a local path dependency (`nano-vllm`) in its
# package configuration, so we clone the repo first and install from
# the local directory. This lets `uv` resolve all dependencies together,
# including the CUDA-enabled PyTorch build and the local `nano-vllm` package.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone --branch v0.1.6 --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ace-step",
    )
    .uv_pip_install(
        "/opt/ace-step", "hf_transfer==0.1.9", "torchcodec==0.10.0", "torch~=2.10.0"
    )
    .entrypoint([])
)

# In addition to source code, we'll also need the model weights.

# ACE-Step 1.5 integrates with the Hugging Face ecosystem, so setting up the models
# is straightforward. The model handlers use Hugging Face
# to download the weights if not already present.

# We use a single `checkpoints/` directory for all model downloads
# (both the DiT and LM models) and persist it with a Modal
# [Volume](https://modal.com/docs/guide/volumes).
# For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

checkpoints_dir = "/opt/ace-step/checkpoints"
model_cache = modal.Volume.from_name("ACE-Step-v15-model-cache", create_if_missing=True)

# We set the `ACESTEP_PROJECT_ROOT` environment variable so that
# the model handlers know where to find the checkpoints directory.

image = image.env(
    {"ACESTEP_PROJECT_ROOT": "/opt/ace-step", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
)

# While we're at it, let's also define the environment for our UI.
# We'll stick with Python and so use FastAPI and Gradio.

web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]==0.115.4",
    "gradio==6.11.0",
    "huggingface-hub==1.9.1",
    "pydantic==2.10.1",
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

app = modal.App("example-generate-music")


@app.cls(gpu="l40s", image=image, volumes={checkpoints_dir: model_cache})
class MusicGenerator:
    @modal.enter()
    def init(self):
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model, ensure_main_model

        # Download models if not already cached in the Volume.
        lm_model_name = "acestep-5Hz-lm-4B"
        ensure_main_model(checkpoints_dir=checkpoints_dir)
        ensure_lm_model(model_name=lm_model_name, checkpoints_dir=checkpoints_dir)

        # Initialize the audio generation model.
        self.dit_handler = AceStepHandler()
        init_status, enable_generate = self.dit_handler.initialize_service(
            project_root="/opt/ace-step",
            config_path="acestep-v15-turbo",
            device="cuda",
        )
        if not enable_generate:
            raise RuntimeError(f"DiT model initialization failed: {init_status}")

        # Initialize the language model for prompt enhancement.
        self.llm_handler = LLMHandler()
        lm_status, lm_success = self.llm_handler.initialize(
            checkpoint_dir=checkpoints_dir,
            lm_model_path=lm_model_name,
            backend="vllm",
            device="cuda",
        )
        if not lm_success:
            raise RuntimeError(f"LM initialization failed: {lm_status}")

    @modal.method()
    def run(
        self,
        prompt: str,
        lyrics: str,
        duration: float = 60.0,
        format: str = "mp3",  # or wav
        manual_seeds: Optional[int] = 1,
    ) -> bytes:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            thinking=True,
        )
        config = GenerationConfig(
            audio_format=format,
            batch_size=1,
            seeds=[manual_seeds] if manual_seeds is not None else None,
            use_random_seed=manual_seeds is None,
        )
        result = generate_music(
            self.dit_handler,
            self.llm_handler,
            params,
            config,
            save_dir="/dev/shm",
        )
        if not result.success:
            raise RuntimeError(f"Music generation failed: {result.error}")
        return Path(result.audios[0]["path"]).read_bytes()


# We can then generate music from anywhere by running code like what we have in the `local_entrypoint` below.


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    duration: Optional[float] = None,
    format: str = "mp3",  # or wav
    manual_seeds: Optional[int] = 1,
):
    if lyrics is None:
        lyrics = "[Instrumental]"
    if prompt is None:
        prompt = "Korean pop music, bright energetic electronic music, catchy melody, female vocals"
        lyrics = """[intro][intro]
            [chorus]
            We're goin' up, up, up, it's our moment
            You know together we're glowing
            Gonna be, gonna be golden
            Oh, up, up, up with our voices
            영원히 깨질 수 없는
            Gonna be, gonna be golden"""
    if duration is None:
        duration = 30.0  # seconds
    print(
        f"🎼 generating {duration} seconds of music from prompt '{prompt[:32] + ('...' if len(prompt) > 32 else '')}'"
        f" and lyrics '{lyrics[:32] + ('...' if len(lyrics) > 32 else '')}'"
    )

    music_generator = MusicGenerator()  # outside of this file, use modal.Cls.from_name
    clip = music_generator.run.remote(
        prompt, lyrics, duration=duration, format=format, manual_seeds=manual_seeds
    )

    dir = Path("/tmp/generate-music")
    dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / f"{slugify(prompt)[:64]}.{format}"
    print(f"🎼 Saving to {output_path}")
    output_path.write_bytes(clip)


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


# You can execute it with a command like:

# ``` shell
# modal run generate_music.py
# ```

# Pass in `--help` to see options and how to use them.

# ## Hosting a web UI for the music generator

# With the Gradio library, we can create a simple web UI in Python
# that calls out to our music generator,
# then host it on Modal for anyone to try out.

# To deploy both the music generator and the UI, run

# ``` shell
# modal deploy generate_music.py
# ```


@app.function(
    image=web_image,
    # Gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    api = FastAPI()

    # Since this Gradio app is running from its own container,
    # we make a `.remote` call to the music generator
    music_generator = MusicGenerator()
    generate = music_generator.run.remote

    temp_dir = Path("/dev/shm")

    async def generate_music(
        prompt: str, lyrics: str, duration: float = 30.0, format: str = "mp3"
    ):
        audio_bytes = await generate.aio(
            prompt, lyrics, duration=duration, format=format
        )

        audio_path = temp_dir / f"{uuid4()}.{format}"
        audio_path.write_bytes(audio_bytes)

        return audio_path

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# Generate Music")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                lyrics = gr.Textbox(label="Lyrics")
                duration = gr.Number(
                    label="Duration (seconds)", value=10.0, minimum=1.0, maximum=300.0
                )
                format = gr.Radio(["wav", "mp3"], label="Format", value="mp3")
                btn = gr.Button("Generate")
            with gr.Column():
                clip_output = gr.Audio(label="Generated Music", autoplay=True)

        btn.click(
            generate_music,
            inputs=[prompt, lyrics, duration, format],
            outputs=[clip_output],
        )

    return mount_gradio_app(app=api, blocks=demo, path="/")
