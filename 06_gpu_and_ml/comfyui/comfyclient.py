# ---
# cmd: ["python", "06_gpu_and_ml/comfyui/comfyclient.py", "--modal-workspace", "modal-labs-examples", "--prompt", "Spider-Man visits Yosemite, rendered by Blender, trending on artstation"]
# output-directory: "/tmp/comfyui"
# ---

import argparse
import json
import pathlib
import sys
import time
import urllib.request

OUTPUT_DIR = pathlib.Path("/tmp/comfyui")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main(args: argparse.Namespace):
    url = f"https://{args.modal_workspace}--example-comfyapp-comfyui-api{'-dev' if args.dev else ''}.modal.run/"
    data = json.dumps({"prompt": args.prompt}).encode("utf-8")
    print(f"Sending request to {url} with prompt: {args.prompt}")
    print("Waiting for response...")
    start_time = time.time()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            assert response.status == 200, response.status
            elapsed = round(time.time() - start_time, 1)
            print(f"Image finished generating in {elapsed} seconds!")
            filename = OUTPUT_DIR / f"{slugify(args.prompt)}.png"
            filename.write_bytes(response.read())
            print(f"Saved to '{filename}'")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Workflow API not found at {url}")


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
        help="Prompt for the image generation model.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="use this flag when running the ComfyUI server in development mode with `modal serve`",
    )

    return parser.parse_args(arglist[1:])


def slugify(s: str) -> str:
    return s.lower().replace(" ", "-").replace(".", "-").replace("/", "-")[:32]


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
