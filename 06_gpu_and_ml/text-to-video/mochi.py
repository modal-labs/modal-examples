# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/text-to-video/mochi.py", "--num_inference_steps", "64"]
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
# requires several minutes on four H100s to produce
# a high-quality clip of even a few seconds.
# It also takes a five to ten minutes to boot up.
# So a single video generation therefore costs about $2
# at our ~$5/hr rate for H100s.

# Keep your eyes peeled for improved efficiency
# as the open source community works on this new model.
# We welcome PRs to improve the performance of this example!

# ## Setting up the environment for Mochi

# We start by defining the environment the model runs in.
# We'll need the [full CUDA toolkit](https://modal.com/docs/guide/cuda),
# [Flash Attention](https://arxiv.org/abs/2205.14135) for fast attention kernels,
# and the Mochi model code.

import json
import os
import tempfile
import time
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES


cuda_version = "12.3.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.4.0", "packaging", "ninja", "wheel", "setuptools")
    .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    .pip_install(
        "git+https://github.com/genmoai/models.git@075b6e36db58f1242921deff83a1066887b9c9e1"
    )
)

app = modal.App("example-mochi", image=image)

with image.imports():
    import numpy as np
    import ray
    import torch
    from einops import rearrange
    from mochi_preview.handler import MochiWrapper
    from PIL import Image
    from tqdm import tqdm

# ## Saving model weights and outputs

# Mochi weighs in at ~80 GB (~20B params, released in full 32bit precision)
# and can take several minutes to generate videos.

# On Modal, we save large or expensive-to-compute data to
# [distributed Volumes](https://modal.com/docs/guide/volumes)
# so that they are accessible from any Modal Function
# or downloadable via the Modal dashboard or CLI.

model = modal.Volume.from_name("mochi-model", create_if_missing=True)
outputs = modal.Volume.from_name("mochi-outputs", create_if_missing=True)

MODEL_CACHE = Path("/root/.cache")  # remote path for saving the model
OUTPUTS_PATH = "/outputs"  # remote path for saving video outputs

# We download the model using the `hf-transfer`
# library from Hugging Face and additionally download
# the text encoder (Google's T5 XXL) using `transformers`.

# This can takes five to thirty minutes, depending on traffic
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

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub",
        "hf-transfer",
        "torch",
        "transformers",
        "sentencepiece",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    volumes={MODEL_CACHE: model}, timeout=2 * HOURS, image=download_image
)
def download_model(
    model_revision: str = "8e9673c5349979457e515fddd38911df6b4ca07f",
):
    from huggingface_hub import snapshot_download
    from transformers import T5EncoderModel, T5Tokenizer

    model.reload()
    print("🍡 downloading Mochi model")

    snapshot_download(
        repo_id="genmo/mochi-1-preview",
        local_dir=MODEL_CACHE / "mochi-1-preview",
        revision=model_revision,
    )
    print("🍡 model downloaded")

    print("🍡 downloading text encoder")

    T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=False)
    T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")

    model.commit()
    print("🍡 text encoder downloaded")


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


@app.local_entrypoint()
def main(
    prompt: str = "A cat playing drums in a jazz ensemble",
    num_inference_steps: int = 200,
):
    from pathlib import Path

    mochi = Mochi()
    local_dir = Path("/tmp/moshi")
    local_dir.mkdir(exist_ok=True, parents=True)
    download_model.remote()
    remote_path = Path(
        mochi.generate_video.remote(
            prompt=prompt, num_inference_steps=num_inference_steps
        )
    )
    local_path = local_dir / remote_path.name
    local_path.write_bytes(b"".join(outputs.read_file(remote_path.name)))
    print("🍡 video saved locally at", local_path)


# To deploy Mochi, run
# ```bash
# modal deploy mochi
# ```

# And then use it from another Python process that has access to your Modal credentials:

# ```python
# import modal
#
# Mochi = modal.Cls.lookup("example-mochi", "Mochi")
# remote_path = Mochi().generate_video.remote(prompt="A cat playing drums in a jazz ensemble")
# ```


# The Mochi inference logic is defined in the Modal [`Cls`](https://modal.com/docs/guide/lifecycle-functions) below.

# See [the Mochi GitHub repo](https://github.com/genmoai/models)
# for more details on running Mochi.


@app.cls(
    gpu=modal.gpu.H100(count=4),
    volumes={
        MODEL_CACHE: model,
        OUTPUTS_PATH: outputs,  # videos are saved to (distributed) disk
    },
    # boot takes a while, so we keep the container warm for 20 minutes after the last call finishes
    timeout=1 * HOURS,
    container_idle_timeout=20 * MINUTES,
)
class Mochi:
    @modal.enter()
    def load_model(self):
        model.reload()
        ray.init()
        model_path = MODEL_CACHE / "mochi-1-preview"
        vae_stats_path = f"{model_path}/vae_stats.json"
        vae_checkpoint_path = f"{model_path}/vae.safetensors"
        model_config_path = f"{model_path}/dit-config.yaml"
        model_checkpoint_path = f"{model_path}/dit.safetensors"
        num_gpus = torch.cuda.device_count()
        if num_gpus < 4:
            print(
                f"🍡 WARNING: Mochi requires at least 4xH100 GPUs, but only {num_gpus} GPU(s) are available."
            )
        print(
            f"🍡 loading model to {num_gpus} GPUs. This can take 5-15 minutes."
        )
        self.model = MochiWrapper(
            num_workers=num_gpus,
            vae_stats_path=vae_stats_path,
            vae_checkpoint_path=vae_checkpoint_path,
            dit_config_path=model_config_path,
            dit_checkpoint_path=model_checkpoint_path,
        )
        print("🍡 model loaded")

    @modal.exit()
    def graceful_exit(self):
        ray.shutdown()

    @modal.method()
    def generate_video(
        self,
        prompt="",
        negative_prompt="",
        width=848,
        height=480,
        num_frames=163,
        seed=12345,
        cfg_scale=4.5,
        num_inference_steps=200,
    ):
        # credit: https://github.com/genmoai/models/blob/7c7d33c49d53bbf939fd6676610e949f3008b5a8/src/mochi_preview/infer.py#L63

        # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
        # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
        sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

        # cfg_schedule should be a list of floats of length num_inference_steps.
        # For simplicity, we just use the same cfg scale at all timesteps,
        # but more optimal schedules may use varying cfg, e.g:
        # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
        cfg_schedule = [cfg_scale] * num_inference_steps

        args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "mochi_args": {
                "sigma_schedule": sigma_schedule,
                "cfg_schedule": cfg_schedule,
                "num_inference_steps": num_inference_steps,
                "batch_cfg": True,
            },
            "prompt": [prompt],
            "negative_prompt": [negative_prompt],
            "seed": seed,
        }

        final_frames = None
        for cur_progress, frames, finished in tqdm(
            self.model(args), total=num_inference_steps + 1
        ):
            final_frames = frames

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        final_frames = rearrange(final_frames, "t b h w c -> b t h w c")
        final_frames = final_frames[0]

        output_path = os.path.join(
            OUTPUTS_PATH, f"output_{int(time.time())}.mp4"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = []
            for i, frame in enumerate(final_frames):
                frame = (frame * 255).astype(np.uint8)
                frame_img = Image.fromarray(frame)
                frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
                frame_img.save(frame_path)
                frame_paths.append(frame_path)

            frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
            ffmpeg_cmd = f"ffmpeg -y -r 30 -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p {output_path}"
            os.system(ffmpeg_cmd)

            json_path = os.path.splitext(output_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(args, f, indent=4)

        outputs.commit()
        print(f"Video saved remotely at: {output_path}")
        return output_path


# ## Addenda

# The remainder of the code in this file is utility code.


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (
        linear_steps * quadratic_steps**2
    )
    linear_coef = (
        threshold_noise / linear_steps
        - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule
