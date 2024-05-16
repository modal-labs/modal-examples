# ---
# cmd: ["python", "06_gpu_and_ml/comfyui/client.py", "--modal-workspace", "modal-labs", "--prompt", "Spider-Man visits Yosemite, rendered by Blender, trending on artstation"]
# ---

import argparse
import pathlib
import sys
import time

import requests


def main(args: argparse.Namespace):
    url = f"https://{args.modal_workspace}--example-comfyui-comfyui-api{'-dev' if args.dev else ''}.modal.run/"
    data = {
        "prompt": args.prompt,
        "input_image_url": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png",
    }
    print("Waiting for response...")
    start_time = time.time()
    res = requests.post(url, json=data)
    if res.status_code == 200:
        end_time = time.time()
        print(
            f"Image finished generating in {round(end_time - start_time, 1)} seconds!"
        )
        filename = "comfyui_gen_image.png"
        (pathlib.Path(__file__).parent / filename).write_bytes(res.content)
        print(f"saved '{filename}'")
    else:
        if res.status_code == 404:
            print(f"Workflow API not found at {url}")
        res.raise_for_status()


def parse_args(arglist: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modal-workspace",
        type=str,
        required=True,
        help="Name of the Modal workspace with the deployed app. Run `modal profile current` to check.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="what to draw in the blank part of the image",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="use this flag when running the ComfyUI server in development mode with `modal serve`",
    )
    parser.add_argument(
        "--input_image-url",
        default="https://github.com/comfyanonymous/ComfyUI_examples/blob/abcc12912ca11f2f7a36b3a36a4b7651db907459/inpaint/yosemite_inpaint_example.png",
        type=str,
        help="URL of the image to inpaint",
    )
    return parser.parse_args(arglist[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
