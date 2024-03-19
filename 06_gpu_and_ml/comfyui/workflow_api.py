# ---
# lambda-test: false
# ---
#
# # Run a ComfyUI workflow in Python
#
# This example serves a ComfyUI [inpainting workflow](https://github.com/comfyanonymous/ComfyUI_examples/tree/master/inpaint) as an endpoint.
# ![example comfyui workspace](./comfyui-hero.png)
import pathlib
import random
from typing import Any, Dict, Mapping, Sequence, Union

from fastapi.responses import HTMLResponse
from modal import Stub, Volume, web_endpoint

from .comfy_ui import image

stub = Stub(name="example-comfy-python-api")
vol_name = "comfyui-images"
vol = Volume.from_name(vol_name, create_if_missing=True)


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
def download_image(url, image_name, save_path='/root/input/'):
    import requests
    try:
        response = requests.get(url)
        response.raise_for_status()
        pathlib.Path(save_path + image_name).write_bytes(response.content)
        print(f"{url} image successfully downloaded")

    except Exception as e:
        print(f"Error downloading {url} image: {e}")


# Adapted from main() in `_generated_workflow_api.py` after running modal run comfyui.comfy_api::get_python_workflow
def run_python_workflow(item: Dict):
    # In the generated version, these are in the global scope, but for Modal we move into the function scope
    import torch
    from nodes import (
        LoadImage,
        VAEEncodeForInpaint,
        KSampler,
        CheckpointLoaderSimple,
        CLIPTextEncode,
        SaveImage,
        VAEDecode,
        NODE_CLASS_MAPPINGS,
    )

    image_name = 'yosemite.png'
    download_image(item['image'], image_name)
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=image_name)

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
            text="watermark, text", clip=get_value_at_index(checkpointloadersimple_2, 1)
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
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_7, 0)
            )

        return saveimage_8


# Serves the python workflow behind a web endpoint
# Generated images are written to a Volume
@stub.function(image=image, gpu="any", volumes={"/data": vol})
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
@stub.function(image=image, gpu="any")
def run_workflow(item: Dict):
    saved_image = run_python_workflow(item)
    images = saved_image["ui"]["images"]
    image_list = []

    for i in images:
        filename = "output/" + i["filename"]
        image_list.append(pathlib.Path(filename).read_bytes())
    return image_list


@stub.local_entrypoint()
def main() -> None:
    values = {
        "prompt": "white heron",
        "image": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png"
    }
    image_list = run_workflow.remote(values)
    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")
