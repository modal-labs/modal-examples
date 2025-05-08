# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/text-to-video/ltx.py", "--num-inference-steps", "64"]
# ---

# # Text-to-video generation with Lightricks LTX-Video

# This example demonstrates how to run the [LTX](https://github.com/Lightricks/LTX-Video)
# video generation model by [Lightricks](https://www.lightricks.com/) on Modal.

# Generating a 2 second video takes about 3 mins from cold start.
# Once the container is warm, it takes about 30 seconds to generate a video.

# Here's one that we generated:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/blonde-woman-blinking.mp4" type="video/mp4" />
# </video>
# </center>


import string
import time
from pathlib import Path

import modal

app = modal.App()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.0",
        "diffusers==0.33.1",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models",
        }
    )
)

# ## Saving outputs

# On Modal, we save large or expensive-to-compute data to
# [distributed Volumes](https://modal.com/docs/guide/volumes).

# We'll use this for saving our LTX weights, as well as our video outputs.

VOLUME_NAME = "ltx-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # remote path for saving video outputs

MODEL_VOLUME_NAME = "ltx-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")  # remote path for saving model weights

MINUTES = 60
HOURS = 60 * MINUTES

# ## Downloading the model

# We download the model weights into Volume cache to speed up cold starts.

# This download takes about two minutes, depending on traffic
# and network speed.

# If you want to launch the download first,
# before running the rest of the code,
# use the following command from the folder containing this file:

# ```bash
# modal run --detach ltx::download_model
# ```

# The `--detach` flag ensures the download will continue
# even if you close your terminal or shut down your computer
# while it's running.


with image.imports():
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video


@app.function(
    image=image,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES,
)
def download_model():
    # uses HF_HOME to point download to the model volume
    DiffusionPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
    )


# ## Setting up our LTX class


# We'll use the `@cls` decorator to define a [Modal Class](https://modal.com/docs/guide/lifecycle-functions)
# which we use to control the lifecycle of our cloud container.
#
# We configure it to use our image, the distributed volume, and a single H100 GPU.
@app.cls(
    image=image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
        MODEL_PATH: model,
    },
    gpu="H100",
    timeout=1 * HOURS,
)
class LTX:
    @modal.enter()
    def load_model(self):
        # our HF_HOME env var points to the model volume as the cache
        self.pipe = DiffusionPipeline.from_pretrained("Lightricks/LTX-Video")
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=200,
        guidance_scale=4.5,
        num_frames=19,
        width=704,
        height=480,
    ):
        frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width,
            height=height,
        ).frames[0]

        # save to disk using prompt as filename
        mp4_name = slugify(prompt)
        export_to_video(frames, Path(OUTPUTS_PATH) / mp4_name)
        outputs.commit()
        return mp4_name


# ## Running LTX-Video inference

# We can trigger LTX-Video inference from our local machine by running the code in
# the local entrypoint below.

# It ensures the model is downloaded to a remote volume,
# spins up a new replica to generate a video, also saved remotely,
# and then downloads the video to the local machine.

# You can trigger it with:
# ```bash
# modal run --detach ltx
# ```

# Optional command line flags can be viewed with:
# ```bash
# modal run ltx --help
# ```

# Using these flags, you can tweak your generation from the command line:
# ```bash
# modal run --detach ltx --prompt="a cat playing drums in a jazz ensemble" --num-inference-steps=64
# ```


@app.local_entrypoint()
def main(
    prompt="The camera pans over a snow-covered mountain range, revealing a vast expanse of snow-capped peaks and valleys.The mountains are covered in a thick layer of snow, with some areas appearing almost white while others have a slightly darker, almost grayish hue. The peaks are jagged and irregular, with some rising sharply into the sky while others are more rounded. The valleys are deep and narrow, with steep slopes that are also covered in snow. The trees in the foreground are mostly bare, with only a few leaves remaining on their branches. The sky is overcast, with thick clouds obscuring the sun. The overall impression is one of peace and tranquility.",
    negative_prompt="worst quality, blurry, jittery, distorted",
    num_inference_steps=200,
    guidance_scale=4.5,
    num_frames=18,  # produces ~1s of video
    width=704,
    height=480,
):
    print(f"🎥 Generating a video from the prompt {prompt}")
    ltx = LTX()
    start = time.time()
    mp4_name = ltx.generate.remote(
        prompt=str(prompt),
        negative_prompt=str(negative_prompt),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        num_frames=int(num_frames),
        width=width,
        height=height,
    )
    duration = time.time() - start
    print(f"🎥 Generated video in {duration:.3f}s")
    print(f"🎥 LTX video saved to volume at {mp4_name}")

    local_dir = Path("/tmp/ltx")
    local_dir.mkdir(exist_ok=True, parents=True)
    local_path = local_dir / mp4_name
    local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
    print(f"🎥 LTX video saved locally at {local_path}")


# ## Addenda

# The remainder of the code in this file is utility code.


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # since filenames can't be longer than 255 characters
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name
