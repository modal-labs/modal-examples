# ---
# lambda-test: false
# ---

import argparse
import pathlib
import sys
import time

import requests


def main(args: argparse.Namespace):
    url = "https://modal-labs--example-comfyui-comfyui-api-dev.modal.run/"
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
            f"Image finished generating in {round(end_time - start_time)} seconds!"
        )
        filename = "comfyui_gen_image.png"
        (pathlib.Path.home() / filename).write_bytes(res.content)
        print(f"saved '{filename}'")
    else:
        print("Request failed!")


def parse_args(arglist: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        default="white heron",
        type=str,
        help="object to draw into the image",
    )
    return parser.parse_args(arglist[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
