import os
import subprocess
import textwrap
import time
from pathlib import Path

import modal
import modal.experimental

# # Config
# MODAL
HF_SECRET = modal.Secret.from_name("huggingface-secret")
VOL_NAME = "example-embedding-data"
VOL_MNT = Path("/data")
data_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# CLUSTER
N_NODES = 2
GPU = "H100"
GPUS_PER_NODE = 8  # one worker will use all local GPUs
IMAGE_TAG = "dynamo-trtllm:modal"  # pushed automatically to Modal’s registry

# APP
DYNAMO_PORT = 8000
NATS_PORT = 4222
ETCD_PORT = 2379
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

# NIXL
NIXL_COMMIT = "f531404be4866d85ed618b3baf4008c636798d63"
NIXL_REPO = "ai-dynamo/nixl.git"
NIXL_UCX_EFA_REF = "7ec95b95e524a87e81cac92f5ca8523e3966b16b"

# PATHS
ARCH = "amd64"
ARCH_ALT = "x86_64"
DYNAMO_HOME = "/dynamo"

# TensorRT
# Apr 22 2025 commit : https://github.com/NVIDIA/TensorRT-LLM/pull/3707/commits
TRTLLM_COMMIT = "main"  # "992e6776714736ce68b3df48596188b6a5f3e92e"  # "8cb6163a57226e69d8a85788eff542a440ed9c89"  # "main"  #
TENSORRTLLM_INDEX_URL = "https://pypi.python.org/simple"
TENSORRTLLM_PIP_WHEEL = ""


# # Image
dynamo_img = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3", add_python="3.12")
    ########################################################################
    # utils / deps
    .run_commands(
        "apt-get update -y && "
        "apt-get install -y --no-install-recommends "
        "git wget curl tmux vim meson ninja-build gdb nvtop "
        "protobuf-compiler cmake libssl-dev pkg-config libclang-dev "
        "git-lfs liburing-dev "
    )
    ########################################################################
    # UCX EFA
    .run_commands("rm -rf /opt/hpcx/ucx && rm -rf /usr/local/ucx")
    .run_commands(
        "cd /usr/local/src && "
        "git clone https://github.com/openucx/ucx.git && "
        "cd ucx &&                   "
        "git checkout v1.19.x &&     "
        "./autogen.sh && ./configure "
        "--prefix=/usr/local/ucx     "
        "--enable-shared             "
        "--disable-static            "
        "--disable-doxygen-doc       "
        "--enable-optimizations      "
        "--enable-cma                "
        "--enable-devel-headers      "
        "--with-cuda=/usr/local/cuda "
        "--with-verbs                "
        "--with-efa                  "
        "--with-dm                   "
        "--with-gdrcopy=/usr/local   "
        "--enable-mt &&              "
        "make -j &&                  "
        "make -j install-strip &&    "
        "ldconfig"
    )
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib:/usr/local/ucx/lib:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH",
            "CPATH": "/usr/include",  # NO CPATH YET# :$CPATH",
            "PATH": "/usr/bin:$PATH",
            "PKG_CONFIG_PATH": "/usr/lib/pkgconfig",  # NO PKG_CONFIG_PATH YET # :$PKG_CONFIG_PATH",
        }
    )
    # SHELL ["/bin/bash", "-c"]
    ########################################################################
    # NIXL
    .run_commands(
        "git init /opt/nixl && "
        "cd /opt/nixl && "
        "git remote add origin https://github.com/ai-dynamo/nixl.git && "
        f'git fetch --depth 1 origin "{NIXL_COMMIT}" && '
        f'git checkout "{NIXL_COMMIT}" '
    )
    .run_commands(
        "cd /opt/nixl && "
        "mkdir build && "
        "meson setup build/ --prefix=/usr/local/nixl && "
        "cd build/ && "
        "ninja && "
        "ninja install "
    )
    .env(
        {
            "NIXL_PREFIX": "/usr/local/nixl",
            "NIXL_ROOT": "/usr/local/nixl",
            "LD_LIBRARY_PATH": "/usr/local/nixl/lib:$LD_LIBRARY_PATH",
        }
    )
    # .pip_install("nixl")
    ########################################################################
    # helper servers (nats & etcd)
    # nats
    .run_commands(
        f"wget --tries=3 --waitretry=5 https://github.com/nats-io/nats-server/releases/download/v2.10.24/nats-server-v2.10.24-{ARCH}.deb && "
        f"dpkg -i nats-server-v2.10.24-{ARCH}.deb && rm nats-server-v2.10.24-{ARCH}.deb"
    )
    # etcd
    .env({"ETCD_VERSION": "v3.5.18"})
    .run_commands(
        f"wget https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-{ARCH}.tar.gz -O /tmp/etcd.tar.gz && "
        "mkdir -p /usr/local/bin/etcd && "
        "tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && "
        "rm /tmp/etcd.tar.gz"
    )
    .env({"PATH": "/usr/local/bin/etcd/:$PATH"})
    .run_commands("pip install --upgrade pip uv")
    ########################################################################
    # TensorRT: FRESH install
    .run_commands("git lfs install")
    .run_commands(
        "truncate -s0 /etc/pip/constraint.txt || true  && "
        "pip uninstall -y tensorrt || true  "
    )
    # # Setup TRT repo
    # # ---- TensorRT-LLM full clone -----------------------------------------
    # .run_commands("cd /tmp/trtllm && "
    #     "git clone --depth 1 --branch main https://github.com/NVIDIA/TensorRT-LLM.git /tmp/trtllm && "
    #     "cd /tmp/trtllm && git submodule update --init --recursive && "
    #     "git submodule foreach --recursive 'git lfs pull || true'"
    # )
    # # Setup TRT wheel (pt 1)
    # .run_commands(
    #     # neutralize constraints file
    #     "cd /tmp/trtllm && truncate -s0 /etc/pip/constraint.txt || true  && "
    #     # setup wheel
    #     "pip install -r requirements.txt"
    # )
    # .run_commands(
    #     "pip install --no-cache-dir --upgrade build wheel scikit-build-core cmake ninja"
    # )
    # # 2️⃣ Produce the wheel
    # .run_commands(
    #     "cd /tmp/trtllm && "
    #     "/usr/local/bin/python3.12 scripts/build_wheel.py --build_dir dist "
    # )
    # .run_commands(
    #     "cd /tmp/trtllm && /usr/local/bin/python3.12 -m build --wheel --outdir dist ."dsfs
    # # Setup TRT wheel (pt 2)
    # # .run_commands("cd /tmp/trtllm && python python/build.py bdist_wheel")
    # # .run_commands("python -m pip wheel /tmp/trtllm -w /tmp/trtllm/dist")
    # # # install the freshly built wheel
    # .run_commands(
    #     'uv pip install --system /tmp/trtllm/dist/tensorrt_llm-*.whl "pynvml<12"'
    # )
    .run_commands(
        "apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && uv pip install --system tensorrt_llm"
    )
    # .pip_install(
    #     "tensorrt-llm==0.18.0",
    #     "pynvml<12",  # avoid breaking change to pynvml version API
    #     pre=True,
    #     extra_index_url="https://pypi.nvidia.com",
    # )
    # # ########################################################################
    # # Dynamo (trying to cheat around building from scratch)
    .run_commands(f"git clone https://github.com/ai-dynamo/dynamo.git {DYNAMO_HOME}")
    .run_commands(
        f'uv pip install --system --requirement "{DYNAMO_HOME}/container/deps/requirements.test.txt"'
    )
    .run_commands("uv pip install --system ai-dynamo grpclib")
    .env(
        {
            "TRTLLM_USE_UCX_KVCACHE": "1",
            "DYNAMO_HOME": DYNAMO_HOME,
            "NIXL_PREFIX": "/usr/local/nixl",
            "LD_LIBRARY_PATH": "/usr/local/nixl/lib:$LD_LIBRARY_PATH",
        }
    )
    # .pip_install("nixl", "msgspec")
    .entrypoint([])
)

app = modal.App(
    "trtllm-dynamo",
    image=dynamo_img,
    secrets=[HF_SECRET],
)


@app.function(
    image=dynamo_img,
    volumes={VOL_MNT: data_volume},
    secrets=[HF_SECRET],
    timeout=24 * 60 * 60,
    cloud="oci",
    gpu=f"{GPU}:{GPUS_PER_NODE}",
)
@modal.experimental.clustered(N_NODES, rdma=True)
@modal.web_server(port=DYNAMO_PORT, startup_timeout=60 * 60)
def dynamo_cluster():
    # params....
    example: str = "tensorrt_llm"
    aggregated: bool = True
    print("Cluster is launched")

    def wait_for_port(host: str, port: int, service_name: str, timeout_min: float = 10):
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

    # Get Modal cluster info for inter-container communication
    cluster_info = modal.experimental.get_cluster_info()
    rank: int = cluster_info.rank
    container_id = os.environ["MODAL_TASK_ID"]
    print(f"Node {rank} fired up in container {container_id}!")
    head: str = cluster_info.container_ips[0]
    os.environ["HEAD_NODE_IP"] = head
    os.environ["NATS_SERVER"] = f"nats://{head}:{NATS_PORT}"
    os.environ["ETCD_ENDPOINTS"] = f"{head}:{ETCD_PORT}"
    # Start sidecars only on rank-0
    if rank == 0:
        nats = subprocess.Popen(
            ["nats-server", "-js", "--trace", f"--port={NATS_PORT}"],
        )
        wait_for_port("localhost", NATS_PORT, "nats")

        etcd = subprocess.Popen(
            [
                "etcd",
                f"--advertise-client-urls=http://0.0.0.0:{ETCD_PORT}",
                f"--listen-client-urls=http://0.0.0.0:{ETCD_PORT}",
            ],
        )
        wait_for_port("localhost", ETCD_PORT, "etcd")

        # Front-end
        file = "agg" if aggregated else "disagg"
        workdir = Path(os.environ["DYNAMO_HOME"]) / "examples" / example
        frontend = subprocess.Popen(
            [
                "dynamo",
                "serve",
                f"graphs.{file}:Frontend",  # DYNAMO_PORT is in there methinks
                "-f",
                f"configs/{file}.yaml",
                # "--http-port",
                # str(DYNAMO_PORT),
                # "--bind",
                # "0.0.0.0",
            ],
            cwd=str(workdir),
        )
        wait_for_port("localhost", DYNAMO_PORT, "dynamo")

        print(f"[rank-0] Frontend → http://{head}:{DYNAMO_HOME}")

    print("Helper node : automagically found by Dynamo? Seems unlikely :(")

    # # Decode worker on every node #---> i think invoking the frontend does this automatically?...
    # decoder = subprocess.Popen(
    #     [
    #         "dynamo",
    #         "serve",
    #         "graphs.agg:VllmDecodeWorker",
    #         "-f",
    #         "examples/multimodal/configs/agg.yaml",
    #         "--model",
    #         MODEL_NAME,
    #         "--served-model-name",
    #         "llava",
    #         "--master-addr",
    #         head,  # let worker talk to rank-0 via NATS
    #     ]
    # )

    # # Keep container alive; sidecars stay in foreground
    # subprocess.Popen(["tail", "-f", "/dev/null"]).wait()


@app.local_entrypoint()
def infer(example: str = "tensorrt_llm", test_timeout=10 * 60):
    import json
    import time
    import urllib

    ############################################################################
    # Health check
    print(f"Running health check for server at {dynamo_cluster.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(
                dynamo_cluster.get_web_url() + "/health"
            ) as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {dynamo_cluster.get_web_url()}"

    print(f"Successful health check for server at {dynamo_cluster.get_web_url()}")

    ############################################################################
    # Beam me up scottie
    # cluster = DynamoCluster()  # start background cluster
    # cluster.infer.remote("Hello, who are you?")

    # Entire cURL command as **one** shell string.
    image_url = "http://images.cocodataset.org/test2017/000000155781.jpg"
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

    # headers = {
    #     "Authorization": f"Bearer {API_KEY}",
    #     "Content-Type": "application/json",
    # }
    #     # headers=headers,
    req = urllib.request.Request(
        dynamo_cluster.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))
