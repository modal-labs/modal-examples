import json
import os
import tempfile
import time

import modal

cuda_version = "12.3.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

def download_model():
    print("Downloading Mochi model")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="genmo/mochi-1-preview",
        local_dir=model_path,
    )
    print("Model downloaded")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/genmoai/models"
    )
    .workdir("/models")
    .run_commands("git checkout 84f7c77ce01ea3e7d3b70a20f22f0038886137ff") # pin to a specific commit
    .pip_install("uv")
    # Mochi repo instructions
    .pip_install(
        "addict>=2.4.0",
        "click>=8.1.7",
        "einops>=0.8.0",
        "omegaconf>=2.3.0",
        "pillow>=11.0.0",
        "pyyaml>=6.0.2",
        "ray>=2.37.0",
        "sentencepiece>=0.2.0",
        "setuptools>=75.2.0",
        "torch==2.4.0",
        "transformers>=4.45.2",
    )
    .run_commands(
        "uv pip install --system --no-build-isolation flash-attn==2.6.3",
        "uv pip install --system .", # installs the models/src/mochi_preview as a package
    )
    .run_function(download_model) # can take 15 minutes or more
)

app = modal.App("example-mochi", image=image)
outputs = modal.Volume.from_name("mochi-outputs", create_if_missing=True)

with image.imports():
    import numpy as np
    import ray
    import torch
    from einops import rearrange
    from mochi_preview.handler import MochiWrapper
    from PIL import Image
    from tqdm import tqdm

# remote path for saving the model
model_path = "/model"

MINUTES = 60

@app.cls(
    gpu=modal.gpu.H100(count=4),
    volumes={"/outputs": outputs}, # videos from mochi save to volume
    timeout=60 * MINUTES,
)
class Mochi:
    @modal.enter()
    def load_model(self):
        ray.init()
        MOCHI_DIR = model_path
        VAE_CHECKPOINT_PATH = f"{MOCHI_DIR}/vae.safetensors"
        MODEL_CONFIG_PATH = f"{MOCHI_DIR}/dit-config.yaml"
        MODEL_CHECKPOINT_PATH = f"{MOCHI_DIR}/dit.safetensors"
        num_gpus = torch.cuda.device_count()
        if num_gpus < 4:
            print(f"WARNING: Mochi requires at least 4xH100 GPUs, but only {num_gpus} GPU(s) are available.")
        print(f"Launching with {num_gpus} GPUs.")

        self.model = MochiWrapper(
            num_workers=num_gpus,
            vae_stats_path=f"{MOCHI_DIR}/vae_stats.json",
            vae_checkpoint_path=VAE_CHECKPOINT_PATH,
            dit_config_path=MODEL_CONFIG_PATH,
            dit_checkpoint_path=MODEL_CHECKPOINT_PATH,
        )
        print("Model loaded")

    @staticmethod
    def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
        if linear_steps is None:
            linear_steps = num_steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
        quadratic_steps = num_steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, num_steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
        return sigma_schedule

    @modal.method()
    def generate_video(
        self,
        prompt = "",
        negative_prompt = "",
        width = 848,
        height = 480,
        num_frames = 163,
        seed = 12345,
        cfg_scale = 4.5,
        num_inference_steps = 64,
    ):
        # credit: https://github.com/genmoai/models/blob/7c7d33c49d53bbf939fd6676610e949f3008b5a8/src/mochi_preview/infer.py#L63

        # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
        # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
        sigma_schedule = self.linear_quadratic_schedule(num_inference_steps, 0.025)

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
        for cur_progress, frames, finished in tqdm(self.model(args), total=num_inference_steps + 1):
            final_frames = frames

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        final_frames = rearrange(final_frames, "t b h w c -> b t h w c")
        final_frames = final_frames[0]

        os.makedirs("/outputs", exist_ok=True)
        output_path = os.path.join("/outputs", f"output_{int(time.time())}.mp4")

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

        print(f"Video generated at: {output_path}")
        return output_path

@app.function(
    volumes={"/outputs": outputs}, # using the same outputs volume syncronizes Mochi instances with our gradio server
    image=modal.Image.debian_slim(python_version="3.11").pip_install("gradio>=3.36.1", "fastapi"),
    concurrency_limit=1, # gradio has sticky sessions, so keep only one container
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    mochi = Mochi()

    with gr.Blocks() as demo:
        gr.Markdown("Video Generator")
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
            )
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            seed = gr.Number(label="Seed", value=1710977262, precision=0)
        with gr.Row():
            width = gr.Number(label="Width", value=848, precision=0)
            height = gr.Number(label="Height", value=480, precision=0)
            num_frames = gr.Number(label="Number of Frames", value=163, precision=0)
        with gr.Row():
            cfg_scale = gr.Number(label="CFG Scale", value=4.5)
            num_inference_steps = gr.Number(
                label="Number of Inference Steps", value=200, precision=0
            )
        btn = gr.Button("Generate Video")
        output = gr.Video()

        btn.click(
            mochi.generate_video.remote, # call our remote Mochi service on click
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                seed,
                cfg_scale,
                num_inference_steps,
            ],
            outputs=output, # output path to the video
        )

    web_app = FastAPI()

    return mount_gradio_app(
        app=web_app,
        blocks=demo,
    )

@app.local_entrypoint()
def main():
    mochi = Mochi()
    mochi.generate_video.remote()


#
## genmoai code, for reference
# from https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py

# import json
# import os
# import tempfile
# import time

# import click
# import numpy as np
# import ray
# from einops import rearrange
# from PIL import Image
# from tqdm import tqdm
# import torch

# from mochi_preview.handler import MochiWrapper

# model = None
# model_path = None


# def set_model_path(path):
#     global model_path
#     model_path = path


# def load_model():
#     global model, model_path
#     if model is None:
#         ray.init()
#         MOCHI_DIR = model_path
#         VAE_CHECKPOINT_PATH = f"{MOCHI_DIR}/vae.safetensors"
#         MODEL_CONFIG_PATH = f"{MOCHI_DIR}/dit-config.yaml"
#         MODEL_CHECKPOINT_PATH = f"{MOCHI_DIR}/dit.safetensors"
#         num_gpus = torch.cuda.device_count()
#         if num_gpus < 4:
#             print(f"WARNING: Mochi requires at least 4xH100 GPUs, but only {num_gpus} GPU(s) are available.")
#         print(f"Launching with {num_gpus} GPUs.")

#         model = MochiWrapper(
#             num_workers=num_gpus,
#             vae_stats_path=f"{MOCHI_DIR}/vae_stats.json",
#             vae_checkpoint_path=VAE_CHECKPOINT_PATH,
#             dit_config_path=MODEL_CONFIG_PATH,
#             dit_checkpoint_path=MODEL_CHECKPOINT_PATH,
#         )

# def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
#     if linear_steps is None:
#         linear_steps = num_steps // 2
#     linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
#     threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
#     quadratic_steps = num_steps - linear_steps
#     quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
#     linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
#     const = quadratic_coef * (linear_steps ** 2)
#     quadratic_sigma_schedule = [
#         quadratic_coef * (i ** 2) + linear_coef * i + const
#         for i in range(linear_steps, num_steps)
#     ]
#     sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
#     sigma_schedule = [1.0 - x for x in sigma_schedule]
#     return sigma_schedule

# def generate_video(
#     prompt,
#     negative_prompt,
#     width,
#     height,
#     num_frames,
#     seed,
#     cfg_scale,
#     num_inference_steps,
# ):
#     load_model()

#     # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
#     # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
#     sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

#     # cfg_schedule should be a list of floats of length num_inference_steps.
#     # For simplicity, we just use the same cfg scale at all timesteps,
#     # but more optimal schedules may use varying cfg, e.g:
#     # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
#     cfg_schedule = [cfg_scale] * num_inference_steps

#     args = {
#         "height": height,
#         "width": width,
#         "num_frames": num_frames,
#         "mochi_args": {
#             "sigma_schedule": sigma_schedule,
#             "cfg_schedule": cfg_schedule,
#             "num_inference_steps": num_inference_steps,
#             "batch_cfg": True,
#         },
#         "prompt": [prompt],
#         "negative_prompt": [negative_prompt],
#         "seed": seed,
#     }

#     final_frames = None
#     for cur_progress, frames, finished in tqdm(model(args), total=num_inference_steps + 1):
#         final_frames = frames

#     assert isinstance(final_frames, np.ndarray)
#     assert final_frames.dtype == np.float32

#     final_frames = rearrange(final_frames, "t b h w c -> b t h w c")
#     final_frames = final_frames[0]

#     os.makedirs("outputs", exist_ok=True)
#     output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")

#     with tempfile.TemporaryDirectory() as tmpdir:
#         frame_paths = []
#         for i, frame in enumerate(final_frames):
#             frame = (frame * 255).astype(np.uint8)
#             frame_img = Image.fromarray(frame)
#             frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
#             frame_img.save(frame_path)
#             frame_paths.append(frame_path)

#         frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
#         ffmpeg_cmd = f"ffmpeg -y -r 30 -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p {output_path}"
#         os.system(ffmpeg_cmd)

#         json_path = os.path.splitext(output_path)[0] + ".json"
#         with open(json_path, "w") as f:
#             json.dump(args, f, indent=4)

#     return output_path


#
# CLI

# @click.command()
# @click.option("--prompt", required=True, help="Prompt for video generation.")
# @click.option(
#     "--negative_prompt", default="", help="Negative prompt for video generation."
# )
# @click.option("--width", default=848, type=int, help="Width of the video.")
# @click.option("--height", default=480, type=int, help="Height of the video.")
# @click.option("--num_frames", default=163, type=int, help="Number of frames.")
# @click.option("--seed", default=12345, type=int, help="Random seed.")
# @click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
# @click.option(
#     "--num_steps", default=64, type=int, help="Number of inference steps."
# )
# @click.option("--model_dir", required=True, help="Path to the model directory.")
# def generate_cli(
#     prompt,
#     negative_prompt,
#     width,
#     height,
#     num_frames,
#     seed,
#     cfg_scale,
#     num_steps,
#     model_dir,
# ):
#     set_model_path(model_dir)
#     output = generate_video(
#         prompt,
#         negative_prompt,
#         width,
#         height,
#         num_frames,
#         seed,
#         cfg_scale,
#         num_steps,
#     )
#     click.echo(f"Video generated at: {output}")

# if __name__ == "__main__":
#     generate_cli()