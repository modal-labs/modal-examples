import subprocess
import os
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Iterator, Sequence, Tuple, List

import modal


# ────────────────────────────── Constants ──────────────────────────────
HF_SECRET = modal.Secret.from_name("huggingface-secret")
VOL_NAME = "example-embedding-data"
VOL_MNT = Path("/data")
data_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
MODEL_REPO = VOL_MNT / "dynamo_repo"  # will hold model.plan + config

# image with dynamo + torch + dynamoclient (tiny helper)
dynamo_IMAGE = modal.Image.from_dockerfile(
    "/home/ec2-user/dynamo/container/Dockerfile.tensorrt_llm"
)


app = modal.App(
    "clip-dynamo-embed22",
    image=dynamo_IMAGE,
    volumes={VOL_MNT: data_volume},
    secrets=[HF_SECRET],
)

with dynamo_IMAGE.imports():
    import torch, torchvision  # noqa: F401   – for torchscript
    from transformers import CLIPVisionModel, CLIPImageProcessorFast
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image
    import dynamoclient.http as httpclient


@app.cls(
    image=dynamo_IMAGE,
    volumes={VOL_MNT: data_volume},
    timeout=24 * 60 * 60,  # if using a large HF dataset, this may need to be longer
    cpu=4,  # HuggingFace will use multi-process parallelism to download
    gpu="H100:2",  # HuggingFace will use multi-process parallelism to download
)
class Server:
    @modal.enter()
    def startup(self):
        self._proc = subprocess.Popen(
            "cd $DYNAMO_HOME/examples/multimodal && "
            "dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml"
        )

    @modal.method()
    def infer(self, in_idx: int):
        import subprocess, textwrap, sys

        # Entire cURL command as **one** shell string.
        curl_cmd = textwrap.dedent("""\
            curl http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"llava-hf/llava-1.5-7b-hf","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"http://images.cocodataset.org/test2017/000000155781.jpg"}}]}],"max_tokens":300,"stream":false}'
        """)

        # Launch the command; `shell=True` is required because we pass a single string.
        proc = subprocess.Popen(
            curl_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # return strings instead of bytes
        )
        return proc


@app.local_entrypoint()
def main():
    x = Server()
    for status in x.infer.map([1, 2]):
        print(status)
