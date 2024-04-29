# ---
# lambda-test: false
# ---
#
# # Run ComfyUI
#
# This example shows you how to run a ComfyUI workspace with `modal serve`.
#
# If you're unfamiliar with how ComfyUI works we recommend going through Scott Detweiler's
# [tutorials on Youtube](https://www.youtube.com/watch?v=AbB33AxrcZo).
#
# ![example comfyui workspace](./comfyui-hero.png)

import pathlib
import subprocess

import modal

# ## Define container image
#
# Fun with ComfyUI begins with pre-trained model checkpoints.
# Add downloadable checkpoints to CHECKPOINTS e.g. [huggingface.co/dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0).
# The ComfyUI repository has other recommendations listed in this file:
# [notebooks/comfyui_colab.ipynb](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb).
CHECKPOINTS = [
    "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt"
]


def download_checkpoints():
    import httpx
    from tqdm import tqdm

    for url in CHECKPOINTS:
        checkpoints_directory = "/root/models/checkpoints"
        local_filename = url.split("/")[-1]
        local_filepath = pathlib.Path(checkpoints_directory, local_filename)
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        print(f"downloading {url} ...")
        with httpx.stream("GET", url, follow_redirects=True) as stream:
            total = int(stream.headers["Content-Length"])
            with open(local_filepath, "wb") as f, tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = stream.num_bytes_downloaded
                for data in stream.iter_bytes():
                    f.write(data)
                    progress.update(stream.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = stream.num_bytes_downloaded


# Add plugins to PLUGINS, a list of dictionaries with two keys:
# `url` for the github url and an optional `requirements` for the name of a requirements.txt to pip install (remove this key if there is none for the plugin).
# For recommended plugins, see this list:
# [WASasquatch/comfyui-plugins](https://github.com/WASasquatch/comfyui-plugins).
PLUGINS = [
    {
        "url": "https://github.com/coreyryanhanson/ComfyQR",
        "requirements": "requirements.txt",
    },
    # Kosinkadink/ComfyUI-Advanced-ControlNet
    {
        "url": "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",
        "requirements": "requirements.txt",
    },
    # Nourepide/ComfyUI-Allor
    {
        "url": "https://github.com/Nourepide/ComfyUI-Allor",
        "requirements": "requirements.txt",
    },
    # Add form requirement
    {
        "url": "https://github.com/formanek/ComfyUI-Form",
        "requirements": "requirements.txt",
    },
    # Add ComfyUI-Manager
    {
        "url": "https://github.com/ltdrdata/ComfyUI-Manager",
        "requirements": "requirements.txt",
    },
    # cubiq/ComfyUI_IPAdapter_plus
    # {
    #     "url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
    # },
    # {
    #     "url": "https://github.com/SeargeDP/SeargeSDXL",
    # },
    # jags111/efficiency-nodes-comfyui
    {
        "url": "https://github.com/jags111/efficiency-nodes-comfyui",
        "requirements": "requirements.txt",
    },
    # https://github.com/banodoco/Steerable-Motion
    {
        "url": "https://github.com/banodoco/Steerable-Motion",
        "requirements": "requirements.txt",
    },
    # ADE_AnimateDiffLoRALoader
    {
        "url": "https://github.com/ADE-AI/ADE_AnimateDiffLoRALoader",
        "requirements": "requirements.txt",
    },
    {
        "url": "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved",
    },
    # https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
    {
        "url": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
    },
    # https://github.com/cubiq/ComfyUI_IPAdapter_plus
    {
        "url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
    },
    # https://github.com/cubiq/ComfyUI_essentials
    {
        "url": "https://github.com/cubiq/ComfyUI_essentials",
    },
    # "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    {
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    },
    # https://github.com/M1kep/ComfyLiterals
    {
        "url": "https://github.com/M1kep/ComfyLiterals",
    },
    # https://github.com/M1kep/Comfy_KepMatteAnything
    {
        "url": "https://github.com/M1kep/Comfy_KepMatteAnything",
    },
    # https://github.com/M1kep/KepPromptLang
    {
        "url": "https://github.com/M1kep/KepPromptLang",
    },
]

extra_plugins = [
    {
        "url": "https://github.com/FizzleDorf/ComfyUI_FizzNodes",
    },
    {
        "url": "https://github.com/SeargeDP/SeargeSDXL",
    },
]


def download_plugins(plugin_list):
    import subprocess

    for plugin in plugin_list:
        url = plugin["url"]
        name = url.split("/")[-1]
        command = f"cd /root/custom_nodes && git clone {url}"
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Repository {url} cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")

        if plugin.get("requirements"):
            pip_command = f"cd /root/custom_nodes/{name} && pip install -r {plugin['requirements']}"
            try:
                subprocess.run(pip_command, shell=True, check=True)
                print(f"Requirements for {url} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error installing requirements: {e.stderr}")


# Pin to a specific commit from https://github.com/comfyanonymous/ComfyUI/commits/master/
# for stability. To update to a later ComfyUI version, change this commit identifier.
comfyui_commit_sha = "a38b9b3ac152fb5679dad03813a93c09e0a4d15e"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Here we place the latest ComfyUI repository code into /root.
    # Because /root is almost empty, but not entirely empty
    # as it contains this comfy_ui.py script, `git clone` won't work.
    # As a workaround we `init` inside the non-empty directory, then `checkout`.
    .pip_install("torch")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        # f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && git checkout master; git pull",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
        "cd /root && git clone https://github.com/pydn/ComfyUI-to-Python-Extension.git",
        "cd /root/ComfyUI-to-Python-Extension && pip install -r requirements.txt",
        "cd /root/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
    )
    .pip_install(
        "httpx",
        "requests",
        "tqdm",
    )
    .run_commands(
        "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    )
    .pip_install(
        "lark",
    )
    .run_commands(
        # Ensure wget is installed before attempting to download the file
        "apt-get install -y wget",
    )
    .run_commands(
        "git clone https://github.com/IDEA-Research/GroundingDINO.git",
        "cd GroundingDINO && pip install -e .",
        "mkdir -p GroundingDINO/weights",
        "cd GroundingDINO/weights && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    )
    .run_function(download_checkpoints)
    .run_function(download_plugins, PLUGINS)
    .pip_install("simpleeval")
    .run_function(download_plugins, extra_plugins)
)
app = modal.App(name="example-comfy-ui", image=image)

# ## Start the ComfyUI server
#
# Inside the container, we will run the ComfyUI server and execution queue on port 8188. Then, we
# wrap this function in the `@web_server` decorator to expose the server as a web endpoint.
#
# For ASGI-compatible frameworks, you can also use Modal's `@asgi_app` decorator.


@app.function(
    gpu="any",
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Restrict to 1 container because we want our ComfyUI session state
    # to be on a single container.
    concurrency_limit=1,
    keep_warm=1,
    timeout=1800,
)
@modal.web_server(8188, startup_timeout=600)
def web():
    cmd = "python main.py --dont-print-server --multi-user --listen --port 8188"
    subprocess.Popen(cmd, shell=True)
