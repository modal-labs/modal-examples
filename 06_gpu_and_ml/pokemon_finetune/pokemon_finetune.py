# Follows https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda to fine-tune stable diffusion for Pokemon


import signal
import subprocess
from pathlib import Path

from fastapi import FastAPI

import modal

web_app = FastAPI()
stub = modal.Stub(name="stable-diffusion-fine-tune-pokemon")

REPO_DIR = Path("/stable-diffusion")
MODEL_DIR = Path("/model")
LOGS_DIR = MODEL_DIR / "logs"
CONFIG_PATH = Path("/config.yaml")

image = (
    modal.Image.debian_slim(python_version="3.10")
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

volume = modal.SharedVolume().persist("stable-diffusion-pokemon")
# volume = modal.SharedVolume().persist("sd-pokemon")


def find_last_checkpoint(dir: Path):
    nonempty_ckpts = [c for c in dir.glob("**/*.ckpt") if c.stat().st_size > 0]
    if not nonempty_ckpts:
        return None
    return max(*nonempty_ckpts, key=lambda c: c.stat().st_mtime)


@stub.function(
    image=image,
    gpu="A100",
    shared_volumes={MODEL_DIR: volume},
    timeout=5 * 60 * 60,  # 5 hours
    secrets=[modal.Secret.from_name("huggingface")],
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "config.yaml", CONFIG_PATH
        ),
    ],
)
def train():
    import os

    from huggingface_hub import hf_hub_download

    # Download huggingface model to MODEL_DIR
    print("Downloading model")
    base_ckpt_path = hf_hub_download(
        repo_id="CompVis/stable-diffusion-v-1-4-original",
        filename="sd-v1-4-full-ema.ckpt",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        cache_dir=MODEL_DIR,
    )

    args = [
        "python",
        "main.py",
        "--train",
        "--base",
        CONFIG_PATH.as_posix(),
        "--gpus",
        "0,",
        "--scale_lr",
        "False",
        "--check_val_every_n_epoch",
        "10",
        "--finetune_from",
        base_ckpt_path,
        "--logdir",
        LOGS_DIR.as_posix(),
    ]

    last_trained_ckpt = find_last_checkpoint(LOGS_DIR)
    if last_trained_ckpt:
        print(f"Resuming from {last_trained_ckpt}")
        args.extend(["--resume", last_trained_ckpt.as_posix()])
    else:
        print("No checkpoint found, starting from scratch")

    try:
        p = subprocess.Popen(args, cwd=REPO_DIR)
        p.wait()
    except KeyboardInterrupt:
        print("Received SIGINT, waiting for training to finish...")
        # The training process receives its own SIGINT already.
        p.send_signal(signal.SIGINT)
        p.wait(timeout=30)
        if p.poll() is None:
            print("Training process did not terminate, killing...")
            p.kill()


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
    import os

    import gradio as gr
    from gradio.routes import mount_gradio_app

    checkpoint = find_last_checkpoint(LOGS_DIR)

    if checkpoint is None:
        raise Exception("No checkpoint found")

    go = prepare_model(
        MODEL_DIR / "configs/stable-diffusion/pokemon.yaml", checkpoint
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
