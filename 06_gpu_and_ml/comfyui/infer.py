# ---
# lambda-test: false
# ---

import argparse
import pathlib
import sys

import requests


def main(args: argparse.Namespace):
    comfyui_workflow_data_path = (
        pathlib.Path(__file__).parent / "workflow-examples" / "inpainting"
    )

    url = "https://modal-labs--example-comfyui-backend-dev.modal.run"
    workflow = pathlib.Path(
        comfyui_workflow_data_path / "workflow_api.json"
    ).read_text()
    models = pathlib.Path(comfyui_workflow_data_path / "model.json").read_text()
    data = {
        "workflow_data": workflow,
        "models": models,
        "text_prompt": args.prompt,
        "input_image_url": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png",
    }

    res = requests.post(url, json=data)
    if res.status_code == 200:
        print("Image finished generating!")
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
