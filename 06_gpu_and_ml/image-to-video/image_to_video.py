# ---
# output-directory: "/tmp/image_to_video"
# args: ["--prompt", ""]
# ---

# # Run Lightricks LTX-Video Image to Video Model as a CLI, API, and web UI

# This example shows how to run [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) on Modal
# to generate videos from your local command line, via an API, and as a web UI.

# Generating a 5 second video takes ~1 minute from cold start.
# Once the container is warm, a 5 second video takes ~15 seconds.

# Here is an example using the model to animate the
# [Disaster Girl](https://en.wikipedia.org/wiki/Disaster_Girl) meme:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-public-assets.s3.us-east-1.amazonaws.com/example_image_to_video.mp4" type="video/mp4" />
# </video>
# </center>

# ## Basic setup

import io
import random
import time
from pathlib import Path
from typing import Annotated

import fastapi
import modal


# All Modal programs need an [`App`](https://modal.com/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = modal.App("example-image-to-video")

# ## Configuring dependencies

# The model runs remotely inside a [container](https://modal.com/docs/guide/custom-container).
# That means we need to install the necessary dependencies in that container's image.

# Below, we start from a lightweight base Linux image
# and then install our Linux and Python dependencies, like Hugging Face's `diffusers` library and `torch`.

MODEL_PATH = "/models"
OUTPUT_PATH = "/outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv")
    .pip_install(
        "accelerate==1.4.0",
        "diffusers==0.32.2",
        "fastapi[standard]==0.115.8",
        "huggingface-hub[hf_transfer]==0.29.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python==4.11.0.86",
        "pillow==11.1.0",
        "sentencepiece==0.2.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.49.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": MODEL_PATH,
        }
    )
)

with image.imports():
    import diffusers
    import torch
    from PIL import Image


# ## Implementing LTX-Video inference on Modal

# We wrap inference in a Modal [Cls](https://modal.com/docs/guide/lifecycle-methods)
# that ensures models are loaded and then moved to the GPU once when a new container
# starts, before the container picks up any work.

# The `run` function just wraps a `diffusers` pipeline.
# It saves the generated video to a Modal volume, and returns the filename.

# We also include a `web` wrapper that makes it possible
# to trigger inference via an API call.
# See the `/docs` route of the URL ending in `inference-web.modal.run`
# that appears when you deploy the app for details.

MODEL_ID = "Lightricks/LTX-Video"
MODEL_REVISION_ID = "a6d59ee37c13c58261aa79027d3e41cd41960925"

model_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("outputs", create_if_missing=True)

MINUTES = 60


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        self.pipe = diffusers.LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def run(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str = None,
        num_frames: int = None,
        num_inference_steps: int = None,
        seed: int = None,
    ) -> str:
        negative_prompt = (
            negative_prompt
            or "worst quality, inconsistent motion, blurry, jittery, distorted"
        )
        width = 768
        height = 512
        num_frames = num_frames or 25
        num_inference_steps = num_inference_steps or 50
        seed = seed or random.randint(0, 2**32 - 1)
        print(f"Seeding RNG with: {seed}")
        torch.manual_seed(seed)

        image = diffusers.utils.load_image(Image.open(io.BytesIO(image_bytes)))

        video = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        ).frames[0]

        mp4_name = slugify(prompt)
        diffusers.utils.export_to_video(
            video, f"{Path(OUTPUT_PATH) / mp4_name}", fps=24
        )
        output_volume.commit()
        torch.cuda.empty_cache()  # reduce fragmentation
        return mp4_name

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(
        self,
        image_bytes: Annotated[bytes, fastapi.File()],
        prompt: str,
        negative_prompt: str = None,
        num_frames: int = None,
        num_inference_steps: int = None,
        seed: int = None,
    ) -> fastapi.Response:
        mp4_name = self.run.local(  # run in the same container
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return fastapi.responses.FileResponse(
            path=f"{Path(OUTPUT_PATH) / mp4_name}",
            media_type="video/mp4",
            filename=mp4_name,
        )


# ## Generating videos from the command line

# Creating a [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint)
# that calls the Inference().run method will allow us to run inference from the command line.
# The function's parameters define the CLI.

# Run it with

# ```bash
# modal run image_to_video.py --prompt "A cat looking out the window at a snowy mountain" --image-path /path/to/cat.jpg
# ```

# You can also pass `--help` to see the full list of arguments.

# ```bash
# modal run image_to_video.py --help
# ```


@app.local_entrypoint()
def entrypoint(
    image_path: str,
    prompt: str,
    negative_prompt: str = None,
    num_frames: int = None,
    num_inference_steps: int = None,
    seed: int = None,
    twice: bool = False,
):
    import os
    import urllib.request

    print(
        f"Image => {image_path}",
        f"Prompt => {prompt}",
        f"Negative prompt => {negative_prompt}",
        f"Number of frames => {num_frames} (24fps)",
        f"Number of inference steps => {num_inference_steps}",
        f"Seed => {seed}",
        sep="\n",
    )

    if image_path.startswith(("http://", "https://")):
        image_bytes = urllib.request.urlopen(image_path).read()
    elif os.path.isfile(image_path):
        image_bytes = Path(image_path).read_bytes()
    else:
        raise ValueError(f"{image_path} is not a valid file or URL.")

    inference_service = Inference()

    for _ in range(2 if twice else 1):
        start = time.time()
        mp4_name = inference_service.run.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            seed=seed,
        )
        duration = time.time() - start
        print(f"Generated video in {duration:.3f}s")

        output_dir = Path("/tmp/image_to_video")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / mp4_name
        output_path.write_bytes(b"".join(output_volume.read_file(mp4_name)))
        print(f"Video saved to {output_path}")


# ## Generating videos via an API

# The Modal `Cls` above also included a [`fastapi_endpoint`](https://modal.com/docs/examples/basic_web),
# which adds a simple web API to the inference method.

# To try it out, run

# ```bash
# modal deploy image_to_video.py
# ```

# copy the printed URL ending in `inference-web.modal.run`,
# and add `/docs` to the end. This will bring up the interactive
# Swagger/OpenAPI docs for the endpoint.

# ## Generating videos in a web UI

# Lastly, we add a simple front-end web UI (written in Alpine.js) for
# our image to video backend.

# This is also deployed by running

# ```bash
# modal deploy image_to_video.py.
# ```

# The `Inference` class will serve multiple users from its own auto-scaling pool of warm GPU containers automatically.

frontend_path = Path(__file__).parent / "frontend"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jinja2==3.1.5", "fastapi[standard]==0.115.8")
    .add_local_dir(frontend_path, remote_path="/assets")
)


@app.function(
    image=web_image,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    import fastapi.templating

    web_app = fastapi.FastAPI()
    templates = fastapi.templating.Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Inference().web.web_url,
                "model_name": "LTX-Video Image to Video",
                "default_prompt": "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


def slugify(s: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{''.join(c if c.isalnum() else '-' for c in s[:100]).strip('-')}.mp4"
