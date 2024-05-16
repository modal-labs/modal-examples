# ---
# lambda-test: false
# ---
#
# # Run a ComfyUI workflow in Python
#
# This example serves a ComfyUI [inpainting workflow](https://github.com/comfyanonymous/ComfyUI_examples/tree/master/inpaint) as an endpoint.
# ![example comfyui workspace](./comfyui-hero.png)
import json
import pathlib
import random
from typing import Any, Dict, Mapping, Sequence, Union

from comfy_ui import comfyui_image
from fastapi.responses import HTMLResponse
from modal import App, Mount, Volume, web_endpoint

with comfyui_image.imports():
    from helpers import download_to_comfyui

app = App(
    name="example-comfy-python-api"
)  # Note: prior to April 2024, "app" was called "stub"
vol_name = "comfyui-images"
vol = Volume.from_name(vol_name, create_if_missing=True)


@app.function(
    image=comfyui_image,
    gpu="any",
    mounts=[
        Mount.from_local_file(
            pathlib.Path(__file__).parent / "workflow_api.json",
            "/root/workflow_api.json",
        )
    ],
)
def convert_workflow_to_python(workflow: str):
    import subprocess

    root_path = pathlib.Path("/root")
    comfyui_to_python_path = root_path / "ComfyUI-to-Python-Extension"

    # Install the extension at runtime since we don't want to clutter up the base ComfyUI image
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/pydn/ComfyUI-to-Python-Extension.git",
        ],
        cwd=root_path,
    )
    subprocess.run(
        ["pip", "install", "-r", "requirements.txt"], cwd=comfyui_to_python_path
    )
    (comfyui_to_python_path / "workflow_api.json").write_text(workflow)
    result = subprocess.run(
        ["python", "comfyui_to_python.py"], cwd=comfyui_to_python_path
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"comfy_api.py exited unexpectedly with code {result.returncode}"
        )
    else:
        try:
            return (comfyui_to_python_path / "workflow_api.py").read_text()
        except FileNotFoundError:
            print("Error: File workflow_api.py not found.")


@app.local_entrypoint()
def get_python_workflow():
    workflow_json = (
        pathlib.Path(__file__).parent / "workflow_api.json"
    ).read_text()
    filename = "_generated_workflow_api.py"
    generated_path = pathlib.Path(__file__).parent / filename
    generated_path.write_text(convert_workflow_to_python.remote(workflow_json))
    print("Wrote python file to {filename}")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# ComfyUI images expect input images to be saved in the /input directory
def download_image(url, save_path="/root/input/"):
    import requests

    try:
        response = requests.get(url)
        response.raise_for_status()
        pathlib.Path(save_path + url.split("/")[-1]).write_bytes(
            response.content
        )
        print(f"{url} image successfully downloaded")

    except Exception as e:
        print(f"Error downloading {url} image: {e}")


# Adapted from main() in `_generated_workflow_api.py` after running modal run comfyui.comfy_api::get_python_workflow
def run_python_workflow(item: Dict):
    # In the generated version, these are in the global scope, but for Modal we move into the function scope
    import torch
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        KSampler,
        LoadImage,
        SaveImage,
        VAEDecode,
        VAEEncodeForInpaint,
    )

    download_image(item["image"])
    models = json.loads(
        (pathlib.Path(__file__).parent / "model.json").read_text()
    )
    for m in models:
        download_to_comfyui(m["url"], m["path"])

    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=item["image"].split("/")[-1])

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name="512-inpainting-ema.ckpt"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_3 = cliptextencode.encode(
            text=f"closeup photograph of a {item['prompt']} in the yosemite national park mountains nature",
            clip=get_value_at_index(checkpointloadersimple_2, 1),
        )

        cliptextencode_5 = cliptextencode.encode(
            text="watermark, text",
            clip=get_value_at_index(checkpointloadersimple_2, 1),
        )

        vaeencodeforinpaint = VAEEncodeForInpaint()
        vaeencodeforinpaint_9 = vaeencodeforinpaint.encode(
            grow_mask_by=6,
            pixels=get_value_at_index(loadimage_1, 0),
            vae=get_value_at_index(checkpointloadersimple_2, 2),
            mask=get_value_at_index(loadimage_1, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(10):
            ksampler_6 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="uni_pc_bh2",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_2, 0),
                positive=get_value_at_index(cliptextencode_3, 0),
                negative=get_value_at_index(cliptextencode_5, 0),
                latent_image=get_value_at_index(vaeencodeforinpaint_9, 0),
            )

            vaedecode_7 = vaedecode.decode(
                samples=get_value_at_index(ksampler_6, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
            )

            saveimage_8 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(vaedecode_7, 0),
            )

        return saveimage_8


# Serves the python workflow behind a web endpoint
# Generated images are written to a Volume
@app.function(
    image=comfyui_image,
    gpu="any",
    volumes={"/data": vol},
    mounts=[
        Mount.from_local_file(
            pathlib.Path(__file__).parent / "model.json",
            "/root/model.json",
        ),
        Mount.from_local_file(
            pathlib.Path(__file__).parent / "helpers.py",
            "/root/helpers.py",
        ),
    ],
)
@web_endpoint(method="POST")
def serve_workflow(item: Dict):
    saved_image = run_python_workflow(item)
    images = saved_image["ui"]["images"]

    for i in images:
        filename = "output/" + i["filename"]
        with open(f'/data/{i["filename"]}', "wb") as f:
            f.write(pathlib.Path(filename).read_bytes())
        vol.commit()

    return HTMLResponse(f"<html>Image saved at volume {vol_name}! </html>")


# Run the workflow as a function rather than an endpoint (for easier local testing)
@app.function(image=comfyui_image, gpu="any")
def run_workflow(item: Dict):
    saved_image = run_python_workflow(item)
    images = saved_image["ui"]["images"]
    image_list = []

    for i in images:
        filename = "output/" + i["filename"]
        image_list.append(pathlib.Path(filename).read_bytes())
    return image_list


@app.local_entrypoint()
def main() -> None:
    values = {
        "prompt": "white heron",
        "image": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png",
    }
    image_list = run_workflow.remote(values)
    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")
