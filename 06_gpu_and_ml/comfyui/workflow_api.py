# ---
# lambda-test: false
# ---
#
# # Run a ComfyUI workflow in Python
#
# This example shows you how to run a ComfyUI workflow as a Modal web endpoint or function.
# This is based on the output file generated from `get_python_workflow` in `comfy_api.py`.
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


# Adapt this from main() in `_generated_workflow_api.py`
def run_python_workflow(item: Dict):
    # In the generated version, these are in the global scope, but for Modal we move into the function scope
    import torch
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        EmptyLatentImage,
        KSampler,
        KSamplerAdvanced,
        LatentUpscaleBy,
        SaveImage,
        VAEDecode,
    )

    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_1 = checkpointloadersimple.load_checkpoint(
            ckpt_name="dreamlike-photoreal-2.0.safetensors"
        )

        cliptextencode = CLIPTextEncode()
        # parameterize the prompt based on user input
        cliptextencode_2 = cliptextencode.encode(
            text=item["pos_prompt"],
            clip=get_value_at_index(checkpointloadersimple_1, 1),
        )

        cliptextencode_3 = cliptextencode.encode(
            text=item["neg_prompt"],
            clip=get_value_at_index(checkpointloadersimple_1, 1),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        ksampler = KSampler()
        latentupscaleby = LatentUpscaleBy()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        ksampler_4 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=12,
            cfg=8,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(checkpointloadersimple_1, 0),
            positive=get_value_at_index(cliptextencode_2, 0),
            negative=get_value_at_index(cliptextencode_3, 0),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        latentupscaleby_10 = latentupscaleby.upscale(
            upscale_method="nearest-exact",
            scale_by=2,
            samples=get_value_at_index(ksampler_4, 0),
        )

        ksampleradvanced_8 = ksampleradvanced.sample(
            add_noise="enable",
            noise_seed=random.randint(1, 2**64),
            steps=30,
            cfg=8,
            sampler_name="euler",
            scheduler="karras",
            start_at_step=12,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            model=get_value_at_index(checkpointloadersimple_1, 0),
            positive=get_value_at_index(cliptextencode_2, 0),
            negative=get_value_at_index(cliptextencode_3, 0),
            latent_image=get_value_at_index(latentupscaleby_10, 0),
        )

        vaedecode_6 = vaedecode.decode(
            samples=get_value_at_index(ksampleradvanced_8, 0),
            vae=get_value_at_index(checkpointloadersimple_1, 2),
        )

        saveimage_19 = saveimage.save_images(
            filename_prefix="ComfyUI", images=vaedecode_6[0]
        )

        return saveimage_19


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
        "pos_prompt": "astronaut riding a unicorn in space",
        "neg_prompt": "blurry, low quality",
    }
    image_list = run_workflow.remote(values)
    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")
