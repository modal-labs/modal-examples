# ---
# cmd: ["python", "06_gpu_and_ml/comfyui/comfyclient.py", "--modal-workspace", "modal-labs", "--prompt", "Spider-Man visits Yosemite, rendered by Blender, trending on artstation"]
# output-directory: "/tmp/comfyui"
# ---

import argparse
import pathlib
import sys
import time

import requests

OUTPUT_DIR = pathlib.Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main(args: argparse.Namespace):
    data = {
        "prompt": args.prompt,
    }
    print(f"Sending request to {args.url} with prompt: {data['prompt']}")
    print("Waiting for response...")
    start_time = time.time()
    res = requests.post(args.url, json=data)
    if res.status_code == 200:
        end_time = time.time()
        print(
            f"Image finished generating in {round(end_time - start_time, 1)} seconds!"
        )
        filename = OUTPUT_DIR / f"{slugify(args.prompt)}.png"
        filename.write_bytes(res.content)
        print(f"saved to '{filename}'")
    else:
        if res.status_code == 404:
            print(f"Workflow API not found at {args.url}")
        res.raise_for_status()


def parse_args(arglist: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the deployed ComfyUI app.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for the image generation model.",
    )

    return parser.parse_args(arglist[1:])


def slugify(s: str) -> str:
    return s.lower().replace(" ", "-").replace(".", "-").replace("/", "-")[:32]


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
