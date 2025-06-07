import os
import subprocess
import time
from pathlib import Path

import modal

# ────────────────────────────── Constants ──────────────────────────────
HF_SECRET = modal.Secret.from_name("huggingface-secret")
VOL_NAME = "example-embedding-data"
VOL_MNT = Path("/data")
data_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# .run_commands(
#     "uv pip install --system --no-build-isolation "
#     "flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6",
# )
flash_attn_release = (  # --force-reinstall --no-deps
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
# flash_attn_release = (
#     "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
#     "flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# )
# image with dynamo + torch + dynamoclient (tiny helper)
dynamo_IMAGE = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3", add_python="3.12")
    ########################################################################################
    # Basics  ---> note, fucked up ...arch=amd64. recheck.
    .run_commands("apt-get update")
    .apt_install(
        # For execution
        "curl",
        "git",
        "pip",
        "nats-server",
        "etcd-server",
        # For building
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "clang",
        "libclang-dev",
        "llvm-dev",
        "pkg-config",
        "cmake",
    )
    ########################################################################################
    # Build Rust
    # Remove any old cargo versions
    .run_commands("apt-get purge -y cargo rustc || true")
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y "
        "--profile minimal --default-toolchain 1.87.0"
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "PATH": "/root/.cargo/bin:$PATH",  # make rustup’s cargo first
            "RUSTUP_HOME": "/root/.rustup",
            "CARGO_HOME": "/root/.cargo",
            "TORCH_CUDA_ARCH_LIST": "80;90",
        }
    )
    ########################################################################################
    # Start to set up Python env pre-dynamo
    .run_commands(
        "pip install --upgrade pip 'hatchling>=1.24' 'hatch-fancy-pypi-readme>=22.5'"
    )
    # .pip_install("uv")
    # .run_commands("uv pip install --system ")
    ########################################################################################
    # Build dynamo
    .run_commands("git clone https://github.com/ai-dynamo/dynamo.git")
    .run_commands(
        "source $HOME/.cargo/env && cd /dynamo && cargo build --release --locked"
    )
    .run_commands("cd /dynamo/lib/bindings/python && uv pip install --system .")
    .run_commands(
        "cd /dynamo && uv pip install --system .[all] nixl hf_transfer accelerate transformers"
    )  # "--features cuda"?
    ########################################################################################
    # Done
    # .run_commands(
    #     f"uv pip install --system --force-reinstall --no-deps {flash_attn_release}"
    # )
    .env(
        {
            "HF_HOME": VOL_MNT.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "DYNAMO_HOME": "/dynamo",
            "RUST_BACKTRACE": "full",
            "NIXL_NETLINK_STRATEGY": "proc",
            "NIXL_DISABLE_NETLINK": "1",
        }
    )
    .entrypoint([])
)

app = modal.App(
    "clip-dynamo-embed",
    image=dynamo_IMAGE,
    volumes={VOL_MNT: data_volume},
    secrets=[HF_SECRET],
)

with dynamo_IMAGE.imports():
    import torch  # noqa: F401   – for torchscript


def _env(key: str, default: str):
    return os.environ.get(key, default)


DYNAMO_PORT = int(_env("DYNAMO_PORT", "8000"))
NATS_PORT = int(_env("NATS_PORT", "4222"))
ETCD_PORT = int(_env("ETCD_PORT", "2379"))


@app.cls(
    image=dynamo_IMAGE,
    volumes={VOL_MNT: data_volume},
    secrets=[HF_SECRET],
    timeout=24 * 60 * 60,
    cpu=4,
    gpu="H100:4",
)
class Server:
    @modal.enter()
    def startup(self):
        # 1. Launch infra
        self.nats = subprocess.Popen(
            ["nats-server", "-js", "--trace", f"--port={NATS_PORT}"],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )
        self.wait_for_port("localhost", NATS_PORT, "nats")

        self.etcd = subprocess.Popen(
            [
                "etcd",
                f"--advertise-client-urls=http://0.0.0.0:{ETCD_PORT}",
                f"--listen-client-urls=http://0.0.0.0:{ETCD_PORT}",
            ],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )
        self.wait_for_port("localhost", ETCD_PORT, "etcd")

        # 2. Launch Dynamo
        workdir = Path(os.environ["DYNAMO_HOME"]) / "examples" / "multimodal"
        self.dynamo = subprocess.Popen(
            [
                "dynamo",
                "serve",
                "graphs.agg:Frontend",
                "-f",
                "configs/agg.yaml",
                # "--http-port",
                # str(DYNAMO_PORT),
            ],
            cwd=str(workdir),
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )

        # 3. Wait for each service
        self.wait_for_port("localhost", DYNAMO_PORT, "dynamo")

    def wait_for_port(
        self, host: str, port: int, service_name: str, timeout_min: float = 10
    ):
        import socket

        from tqdm import tqdm

        checkrate_hz = 2
        t = int(timeout_min * 60 * checkrate_hz)
        bar = tqdm(range(t), total=t, desc=f"Waiting for {service_name} heartbeat")
        for tick in bar:
            try:
                with socket.create_connection((host, port), timeout=1):
                    bar.close()
                    tqdm.write(
                        f"\n\t{service_name} is ready on {host}:{port} after {tick / checkrate_hz}s.\n"
                    )
                    return
            except OSError:
                time.sleep(1 / checkrate_hz)

        raise RuntimeError(f"{service_name} failed to become ready on {host}:{port}.")

    @modal.exit()
    def shutdown(self):
        self.nats.terminate()
        self.etcd.terminate()
        self.dynamo.terminate()

    def _dump_proc_logs(self):
        for name in ["dynamo", "nats", "etcd"]:
            proc = getattr(self, name)
            print(f"\n⚠️  {name} exited with {proc.returncode}. Last 40 lines:")
            print(f"OUT\n\t{proc.stdout}")
            print(f"ERR\n\t{proc.stderr}")

    @modal.method()
    def infer(
        self,
        in_idx: int,
        image_url: str = "http://images.cocodataset.org/test2017/000000155781.jpg",
    ):
        import subprocess, textwrap, sys

        # Entire cURL command as **one** shell string.
        url = f"http://localhost:{DYNAMO_PORT}/v1/chat/completions"
        payload = {
            "model": "llava-hf/llava-1.5-7b-hf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 300,
            "stream": False,
        }
        import requests

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()  # ✅ pickle-able result
        except Exception as e:
            # On failure, dump crashed service logs to help debugging
            self._dump_proc_logs()
            raise RuntimeError(f"Request to {url} failed: {e}") from e
        # # Launch the command; `shell=True` is required because we pass a single string.
        # result = subprocess.run(
        #     curl_cmd,
        #     shell=True,
        #     capture_output=True,  # grabs both stdout & stderr
        #     text=True,
        #     check=True,  # raises if curl exits non-zero
        # )
        # return {
        #     "stdout": result.stdout,
        #     "stderr": result.stderr,
        #     "returncode": result.returncode,
        # }


@app.local_entrypoint()
def main():
    x = Server()
    print(x.infer.remote(1))
