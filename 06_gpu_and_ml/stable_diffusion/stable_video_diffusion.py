# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/stable_diffusion/stable_video_diffusion.py"]
# ---
# # Run Stable Video Diffusion in a Streamlit app
#
# This example runs the [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) image-to-video model.

import os
import sys

import modal

app = modal.App(name="example-stable-video-diffusion-streamlit")
q = modal.Queue.from_name(
    "stable-video-diffusion-streamlit", create_if_missing=True
)

session_timeout = 15 * 60


def download_model():
    # Needed because all paths are relative :/
    os.chdir("/sgm")
    sys.path.append("/sgm")

    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from scripts.demo.streamlit_helpers import load_model_from_config
    from scripts.demo.video_sampling import VERSION2SPECS

    snapshot_download(
        "stabilityai/stable-video-diffusion-img2vid",
        local_dir="checkpoints/",
        local_dir_use_symlinks=False,
    )

    spec = VERSION2SPECS["svd"]
    config = OmegaConf.load(spec["config"])
    load_model_from_config(config, spec["ckpt"])


svd_image = (
    # The generative-models repo hardcodes `tokenizers==0.12.1`, for which there is no
    # pre-built python 3.11 wheel.
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/Stability-AI/generative-models.git /sgm"
    )
    .workdir("/sgm")
    .pip_install(".")
    .pip_install(
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "torchaudio==2.0.2+cu118",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands("pip install -r requirements/pt2.txt")
    .apt_install("ffmpeg", "libsm6", "libxext6")  # for CV2
    .pip_install("safetensors")
    .run_function(download_model, gpu="any")
)


@app.function(image=svd_image, timeout=session_timeout, gpu="A100")
def run_streamlit(publish_url: bool = False):
    from streamlit.web.bootstrap import load_config_options, run

    # TODO: figure out better way to do this with streamlit.
    os.chdir("/sgm")
    sys.path.append("/sgm")

    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        # Reload Streamlit config with information about Modal tunnel address.
        if publish_url:
            q.put(tunnel.url)
        load_config_options(
            {"browser.serverAddress": tunnel.host, "browser.serverPort": 443}
        )
        run(
            main_script_path="/sgm/scripts/demo/video_sampling.py",
            is_hello=False,
            args=["--timeout", str(session_timeout)],
            flag_options={},
        )


endpoint_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
)


@app.function(image=endpoint_image)
@modal.web_endpoint(method="GET", label="svd")
def share():
    from fastapi.responses import RedirectResponse

    run_streamlit.spawn(publish_url=True)
    url = q.get()
    return RedirectResponse(url, status_code=303)
