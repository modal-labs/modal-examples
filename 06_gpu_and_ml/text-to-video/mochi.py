# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/text-to-video/mochi.py", "--num-inference-steps", "64"]
# ---

# # Generate videos from text prompts with Mochi

# This example demonstrates how to run the [Mochi 1](https://github.com/genmoai/models)
# video generation model by [Genmo](https://www.genmo.ai/) on Modal.

# Here's one that we generated, inspired by our logo:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-public-assets.s3.us-east-1.amazonaws.com/modal-logo-splat.mp4" type="video/mp4" />
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

# At the time of writing, Mochi is supported natively in the [Diffusers](https://github.com/huggingface/diffusers) library.
# We'll install install diffusers from source to get these latest features.

import modal

import time
import string
from pathlib import Path

app = modal.App()

image = (modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch", "accelerate", "hf-transfer", "sentencepiece", "imageio[ffmpeg]")
    .pip_install(
        "git+https://github.com/huggingface/diffusers",
        "git+https://github.com/huggingface/transformers"
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ## Saving outputs
# On Modal, we save large or expensive-to-compute data to 
# [distributed Volumes](https://modal.com/docs/guide/volumes)

VOLUME_NAME = "mochi-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = "/outputs"  # remote path for saving video outputs

MINUTES = 60
HOURS = 60 * MINUTES

# ## Setting up our Mochi class
# We'll use the `modal.cls` decorator to define a [Modal Class](https://modal.com/docs/guide/lifecycle-functions)
# which we use to control the lifecycle of our cloud container.
# 
# We configure it to use our image, the distributed volume, and a single H100 GPU.

with image.imports():
    import torch
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video
    
@app.cls(
    image=image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
    },
    gpu=modal.gpu.H100(count=1),
    timeout= 1 * HOURS,
    )
class Mochi:
    # We can stack the `modal.build` and `modal.enter` decorators
    # because we use the diffusers library, which caches the model to disk on its first run.
    # This builds our model weights into the container image, ready for future use.
    @modal.build()
    @modal.enter()
    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = MochiPipeline.from_pretrained(
            "genmo/mochi-1-preview", 
            torch_dtype=torch.bfloat16, 
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

    @modal.method()
    def generate(self, 
        prompt, 
        negative_prompt = "",
        num_inference_steps = 200,
        guidance_scale = 4.5,
        num_frames = 19,
    ):
        frames = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            num_frames=num_frames,
        ).frames[0]
        
        # save to disk using prompt as filename
        for char in string.punctuation:
            prompt = prompt.replace(char, "")
        prompt = prompt.replace(" ", "_")
        prompt = prompt[:230] # since filenames can't be longer than 255 characters
        mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
        
        export_to_video(frames, Path(OUTPUTS_PATH) /  mp4_name)
        outputs.commit()
        return mp4_name

# ## Running Mochi inference

# We can trigger Mochi inference from our local machine by running the code in
# the local entrypoint below.

# It ensures the model is downloaded into the image,
# spins up a replica to generate a video,
# and then downloads that video to the local machine.

# You can trigger it with:

# ```bash
# modal run --detach mochi
# ```
# 
# Optional command line flags are:
# --prompt="your prompt"
# --negative-prompt="your negative prompt"
# --num-inference-steps=200
# --guidance-scale=4.5
# --num-frames=19
# 
# Such as:
# ```bash
# modal run --detach mochi --prompt="a cat playing drums in a jazz ensemble" --num-inference-steps=64
# ```

@app.local_entrypoint()
def main(
    prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
    negative_prompt = "",
    num_inference_steps = 200,
    guidance_scale = 4.5,
    num_frames = 19, # produces ~1s of video
    ): 
    
    mochi = Mochi()
    mp4_name = mochi.generate.remote(
        prompt=str(prompt), 
        negative_prompt=str(negative_prompt),
        num_inference_steps=int(num_inference_steps), 
        guidance_scale=float(guidance_scale),
        num_frames=int(num_frames),
    )
    print("🍡 video saved to volume at "+mp4_name)

    local_dir = Path("/tmp/mochi")
    local_dir.mkdir(exist_ok=True, parents=True)
    local_path = local_dir / mp4_name
    local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
    print("🍡 video saved locally at", local_path)
    