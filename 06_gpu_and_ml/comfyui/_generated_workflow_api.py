import os
import random
import sys
from typing import Sequence, Mapping, Any, Union


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


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

from nodes import (
    CheckpointLoaderSimple,
    EmptyLatentImage,
    KSampler,
    KSamplerAdvanced,
    LatentUpscaleBy,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
    VAEDecode,
)


def main():
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_1 = checkpointloadersimple.load_checkpoint(
            ckpt_name="dreamlike-photoreal-2.0.safetensors"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_2 = cliptextencode.encode(
            text="a bag of wooden blocks",
            clip=get_value_at_index(checkpointloadersimple_1, 1),
        )

        cliptextencode_3 = cliptextencode.encode(
            text="bag of noodles", clip=get_value_at_index(checkpointloadersimple_1, 1)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        ksampler = KSampler()
        latentupscaleby = LatentUpscaleBy()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()

        for q in range(10):
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

            vaedecode_11 = vaedecode.decode(
                samples=get_value_at_index(ksampler_4, 0),
                vae=get_value_at_index(checkpointloadersimple_1, 2),
            )


if __name__ == "__main__":
    main()
