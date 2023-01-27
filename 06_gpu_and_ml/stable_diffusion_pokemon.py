# Follows https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda to fine-tune stable diffusion for Pokemon

from pathlib import Path

from fastapi import FastAPI

import modal

web_app = FastAPI()
stub = modal.Stub(name="stable-diffusion-fine-tune-pokemon")

image = (
    modal.Image.debian_slim()
    .run_commands(
        [
            "apt-get install -y git wget libgl1 libglib2.0-0",
            "git clone https://github.com/justinpinkney/stable-diffusion.git repo",
            "cd repo && pip install -r requirements.txt",
        ]
    )
    .pip_install("gradio~=3.10")
)

# A persistent shared volume will store model artefacts across Modal app runs.
# This is crucial as finetuning runs are separate from the Gradio app we run as a webhook.

volume = modal.SharedVolume().persist("stable-diffusion-pokemon")
MODEL_DIR = Path("/stable-diffusion")


@stub.function(
    image=image,
    gpu="A100",  # finetuning is VRAM hungry, so this should be an A100
    shared_volumes={
        str(MODEL_DIR): volume,  # fine-tuned model will be stored at `MODEL_DIR`
    },
    timeout=5 * 60 * 60,  # 5 hours
)
def train():
    import subprocess

    print("Copying training repo")
    subprocess.check_call(["cp", "-R", "repo", "stable-diffusion"], cwd=Path("/"))

    print("Downloading model")
    subprocess.check_call(
        [
            "wget",
            "-q",
            "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt",
        ],
        cwd=MODEL_DIR,
    )

    print("Starting training")
    subprocess.run(
        [
            "python",
            "main.py",
            "-t",
            "--base",
            "configs/stable-diffusion/pokemon.yaml",
            "--gpus",
            ",0",
            "--scale_lr",
            "False",
            "--num_nodes",
            "1",
            "--check_val_every_n_epoch",
            "10",
            "--finetune_from",
            "sd-v1-4-full-ema.ckpt",
        ],
        cwd=MODEL_DIR,
    )


# ## Wrap the trained model in Gradio's web UI
#
# Gradio.app makes it super easy to expose a model's functionality
# in an easy-to-use, responsive web interface.
#
# This model is a text-to-image generator,
# so we set up an interface that includes a user-entry text box
# and a frame for displaying images.


@stub.asgi(
    image=image,
    gpu="A100",
    shared_volumes={str(MODEL_DIR): volume},
)
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    import os

    # take the latest one in logs
    logs_dir = max(os.listdir(MODEL_DIR / "logs"))
    checkpoint = "last.ckpt"
    go = prepare_model(
        MODEL_DIR / "configs/stable-diffusion/pokemon.yaml", MODEL_DIR / f"logs/{logs_dir}/checkpoints/{checkpoint}"
    )

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title="Generate Pokemon images",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


def load_model_from_config(config, ckpt, verbose=False):
    import torch
    from ldm.util import instantiate_from_config

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


# from https://github.com/justinpinkney/stable-diffusion/blob/main/scripts/txt2img.py
def prepare_model(config_file, ckpt_file):
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf
    import torch
    from ldm.models.diffusion.ddim import DDIMSampler
    from einops import rearrange
    from PIL import Image
    import numpy as np

    C, H, W, f = 4, 512, 512, 8

    seed_everything(42)
    config = OmegaConf.load(config_file)
    model = load_model_from_config(config, ckpt_file).to("cuda")
    sampler = DDIMSampler(model)

    def go(prompt):
        start_code = None
        with torch.no_grad():
            with model.ema_scope():
                uc = model.get_learned_conditioning([""])
                c = model.get_learned_conditioning([prompt])
                shape = [C, H // f, W // f]
                samples_ddim, _ = sampler.sample(
                    S=50,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=5.0,
                    unconditional_conditioning=uc,
                    x_T=start_code,
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = x_samples_ddim[0]
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                return Image.fromarray(x_sample.astype(np.uint8))

    return go
