# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/text-to-video/mochi.py", "--num-inference-steps", "64"]
# ---

# # Text-to-video generation with Mochi

# This example demonstrates how to run the [Mochi 1](https://github.com/genmoai/models)
# video generation model by [Genmo](https://www.genmo.ai/) on Modal.

# Here's one that we generated, inspired by our logo:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/modal-logo-splat.mp4" type="video/mp4" />
# </video>
# </center>

# Note that the Mochi model, at time of writing,
# requires several minutes on one H100 to produce
# a high-quality clip of even a few seconds.
# So a single video generation therefore costs about $0.33
# at our ~$5/hr rate for H100s.

# Keep your eyes peeled for improved efficiency
# as the open source community works on this new model.
# We welcome PRs to improve the performance of this example!

# ## Setting up the environment for Mochi

# At the time of writing, Mochi is supported natively in the [`diffusers`](https://github.com/huggingface/diffusers) library,
# but only in a pre-release version.
# So we'll need to install `diffusers` and `transformers` from GitHub.

import string
import time
from pathlib import Path

import modal

app = modal.App("example-mochi")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "accelerate==1.1.1",
        "hf_transfer==0.1.8",
        "sentencepiece==0.2.0",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "git+https://github.com/huggingface/transformers@30335093276212ce74938bdfd85bfd5df31a668a",
        "git+https://github.com/huggingface/diffusers@99c0483b67427de467f11aa35d54678fd36a7ea2",
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
# [distributed Volumes](https://modal.com/docs/guide/volumes)

# We'll use this for saving our Mochi weights, as well as our video outputs.

VOLUME_NAME = "mochi-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # remote path for saving video outputs

MODEL_VOLUME_NAME = "mochi-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")  # remote path for saving model weights

MINUTES = 60
HOURS = 60 * MINUTES

# ## Downloading the model

# We download the model weights into Volume cache to speed up cold starts. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

# This download takes five minutes or more, depending on traffic
# and network speed.

# If you want to launch the download first,
# before running the rest of the code,
# use the following command from the folder containing this file:

# ```bash
# modal run --detach mochi::download_model
# ```

# The `--detach` flag ensures the download will continue
# even if you close your terminal or shut down your computer
# while it's running.


with image.imports():
    import torch
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video


@app.function(
    image=image,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES,
)
def download_model(revision="83359d26a7e2bbe200ecbfda8ebff850fd03b545"):
    # uses HF_HOME to point download to the model volume
    MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview",
        torch_dtype=torch.bfloat16,
        revision=revision,
    )


# ## Setting up our Mochi class


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
class Mochi:
    @modal.enter()
    def load_model(self):
        # our HF_HOME env var points to the model volume as the cache
        self.pipe = MochiPipeline.from_pretrained(
            "genmo/mochi-1-preview",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=200,
        guidance_scale=4.5,
        num_frames=19,
    ):
        frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
        ).frames[0]

        # save to disk using prompt as filename
        mp4_name = slugify(prompt)
        export_to_video(frames, Path(OUTPUTS_PATH) / mp4_name)
        outputs.commit()
        return mp4_name


# ## Running Mochi inference

# We can trigger Mochi inference from our local machine by running the code in
# the local entrypoint below.

# It ensures the model is downloaded to a remote volume,
# spins up a new replica to generate a video, also saved remotely,
# and then downloads the video to the local machine.

# You can trigger it with:
# ```bash
# modal run --detach mochi
# ```

# Optional command line flags can be viewed with:
# ```bash
# modal run mochi --help
# ```

# Using these flags, you can tweak your generation from the command line:
# ```bash
# modal run --detach mochi --prompt="a cat playing drums in a jazz ensemble" --num-inference-steps=64
# ```


@app.local_entrypoint()
def main(
    prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
    negative_prompt="",
    num_inference_steps=200,
    guidance_scale=4.5,
    num_frames=19,  # produces ~1s of video
):
    mochi = Mochi()
    mp4_name = mochi.generate.remote(
        prompt=str(prompt),
        negative_prompt=str(negative_prompt),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        num_frames=int(num_frames),
    )
    print(f"🍡 video saved to volume at {mp4_name}")

    local_dir = Path("/tmp/mochi")
    local_dir.mkdir(exist_ok=True, parents=True)
    local_path = local_dir / mp4_name
    local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
    print(f"🍡 video saved locally at {local_path}")


# ## Addenda

# The remainder of the code in this file is utility code.


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # since filenames can't be longer than 255 characters
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name
