# Follows https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda to fine-tune stable diffusion for Pokemon


from pathlib import Path

from fastapi import FastAPI

import modal

web_app = FastAPI()
stub = modal.Stub(name="stable-diffusion-fine-tune-pokemon")

REPO_DIR = Path("/stable-diffusion")
MODEL_DIR = Path("/model")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .run_commands(
        [
            f"git clone https://github.com/justinpinkney/stable-diffusion.git {str(REPO_DIR)}",
            f"cd {str(REPO_DIR)} && pip install -r requirements.txt",
        ]
    )
    .pip_install("gradio~=3.10")
)

# A persistent shared volume will store model artefacts across Modal app runs.
# This is crucial as finetuning runs are separate from the Gradio app we run as a webhook.

volume = modal.SharedVolume().persist("sd-pokemon")


@stub.function(
    image=image,
    gpu=modal.gpu.A10G(count=2),
    # fine-tuned model will be stored at `MODEL_DIR`
    shared_volumes={MODEL_DIR: volume},
    timeout=5 * 60 * 60,  # 5 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import os
    import subprocess

    from huggingface_hub import hf_hub_download

    # Download huggingface model to MODEL_DIR
    print("Downloading model")
    ckpt_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-2-1",
        filename="v2-1_768-ema-pruned.ckpt",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        cache_dir=MODEL_DIR,
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
            "0,1",
            "--scale_lr",
            "False",
            "--check_val_every_n_epoch",
            "10",
            "--finetune_from",
            ckpt_path,
            "--logs_dir",
            MODEL_DIR / "logs",
        ],
        cwd=REPO_DIR,
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
    gpu="A10G",
    shared_volumes={str(MODEL_DIR): volume},
)
def fastapi_app():
    import os

    import gradio as gr
    from gradio.routes import mount_gradio_app

    # take the latest one in logs
    logs_dir = max(os.listdir(MODEL_DIR / "logs"))
    checkpoint = "last.ckpt"
    go = prepare_model(
        MODEL_DIR / "configs/stable-diffusion/pokemon.yaml",
        MODEL_DIR / f"logs/{logs_dir}/checkpoints/{checkpoint}",
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
    import numpy as np
    import torch
    from einops import rearrange
    from ldm.models.diffusion.ddim import DDIMSampler
    from omegaconf import OmegaConf
    from PIL import Image
    from pytorch_lightning import seed_everything

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
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                x_sample = x_samples_ddim[0]
                x_sample = 255.0 * rearrange(
                    x_sample.cpu().numpy(), "c h w -> h w c"
                )
                return Image.fromarray(x_sample.astype(np.uint8))

    return go
