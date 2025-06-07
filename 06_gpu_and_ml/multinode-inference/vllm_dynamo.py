import os
import subprocess
import time
from pathlib import Path

import modal
import modal.experimental


# TODO:
#  1. convert a bunch of these const into env vars since we wont be twiddling them all
#  2. try to remove all the hardcoded paths e.g. `/workspace`
# TODO: Maybe
#  1. combine the vllm / trt / other image builds into one big file???
#

# # Config
# MODAL
HF_SECRET = modal.Secret.from_name("huggingface-secret")
NGC_SECRET = modal.Secret.from_name("ngc-secret")
VOL_NAME = "example-embedding-data"
VOL_MNT = Path("/data")
data_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# CLUSTER
N_NODES = 4
GPU = "H100"
GPUS_PER_NODE = 8  # one worker will use all local GPUs
MODEL_NAME = "nvidia/Llama-3.1-Minitron-4B-Width-Base"

# APP
DYNAMO_PORT = 8000
NATS_PORT = 4222
ETCD_PORT = 2379
PROMETHEUS_PORT = 9090  # Not currently implemented
GRAFANA_PORT = 3001  # Dynamo's docker-compose overrides Grafana default

# DYNAMO
VIRTUAL_ENV = "/opt/dynamo/venv"
DYNAMO_HOME = "/workspace"

# NIXL
NIXL_COMMIT = "f531404be4866d85ed618b3baf4008c636798d63"
NIXL_REPO = "ai-dynamo/nixl.git"
NIXL_UCX_EFA_REF = "7ec95b95e524a87e81cac92f5ca8523e3966b16b"
NIXL_UCX_REF = "v1.19.x"
NIXL_PREFIX = "/usr/local/nixl"

# PATHS
ARCH = "amd64"
ARCH_ALT = "x86_64"
DYNAMO_SRC = "/src/dynamo"

# vLLM
# Install patched vllm - keep this early in Dockerfile to avoid
# rebuilds from unrelated source code changes
VLLM_REF = "0.8.4"
VLLM_PATCH = f"vllm_v{VLLM_REF}-dynamo-kv-disagg-patch.patch"
VLLM_PATCHED_PACKAGE_NAME = "ai_dynamo_vllm"
VLLM_PATCHED_PACKAGE_VERSION = "0.8.4.post2"
VLLM_MAX_JOBS = 4
VLLM_BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base"
VLLM_BASE_IMAGE_TAG = "25.01-cuda12.8-devel-ubuntu24.04"
# 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


# # Image
dynamo_img = (
    # cant get nvidia credentials to work.
    # modal.Image.from_registry(
    #     f"{VLLM_BASE_IMAGE}/{VLLM_BASE_IMAGE_TAG}",
    #     add_python="3.12",
    #     secret=NGC_SECRET,
    # )
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3", add_python="3.12")
    ########################################################################
    # utils / deps
    .run_commands(
        "apt-get update -y && apt-get install -y "
        # NIXL build dependencies
        "cmake meson ninja-build pybind11-dev "
        # Rust build dependencies
        "clang libclang-dev git "
        # Install utilities
        "nvtop tmux vim autoconf libtool "
        # Bonus
        "liburing-dev"
    )
    ########################################################################
    # UCX EFA
    # These headers are missing with the hpcx installer, required
    # by UCX to find RDMA devices
    .run_commands(
        "apt-get update -y && "
        "apt-get install -y --no-install-recommends "
        "--reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev "
        "libnuma-dev librdmacm-dev ibverbs-providers"
    )
    .run_commands("rm -rf /opt/hpcx/ucx && rm -rf /usr/local/ucx")
    .run_commands(
        "cd /usr/local/src && "
        "git clone https://github.com/openucx/ucx.git && "
        "cd ucx && "
        "git checkout $NIXL_UCX_REF && "
        "./autogen.sh && ./configure "
        "--prefix=/usr/local/ucx "
        "--enable-shared "
        "--disable-static "
        "--disable-doxygen-doc "
        "--enable-optimizations "
        "--enable-cma "
        "--enable-devel-headers "
        "--with-cuda=/usr/local/cuda "
        "--with-verbs "
        "--with-efa "
        "--with-dm "
        "--with-gdrcopy=/usr/local "
        "--enable-mt && "
        "make -j && "
        "make -j install-strip && "
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
        f"meson setup build/ --prefix={NIXL_PREFIX} && "
        "cd build/ && "
        "ninja && "
        "ninja install "
    )
    .env(
        {  # certainly dont need both of these but seeming them in the logs
            "NIXL_PREFIX": f"{NIXL_PREFIX}",
            "NIXL_ROOT": f"{NIXL_PREFIX}",
            "LD_LIBRARY_PATH": f"{NIXL_PREFIX}/lib:$LD_LIBRARY_PATH",
        }
    )
    ########################################################################
    # NATS & ETCD SETUP
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
    ########################################################################
    # Virtual Environment Setup
    .run_commands(
        "pip install --upgrade pip uv && "
        "mkdir /opt/dynamo && "
        "uv venv /opt/dynamo/venv --python 3.12"
    )
    .env({"VIRTUAL_ENV": "/opt/dynamo/venv", "PATH": "/opt/dynamo/venv/bin:${PATH}"})
    # Install NIXL Python mod
    .run_commands("cd /opt/nixl && uv build . --out-dir /workspace/wheels/nixl")
    .run_commands("uv pip install /workspace/wheels/nixl/*.whl")
    # We are going to need bits of dynamo going forward
    .run_commands(f"git clone https://github.com/ai-dynamo/dynamo.git {DYNAMO_SRC}")
    ########################################################################
    # vLLM
    .run_commands(
        "mkdir -p /root/.cache/uv && "  ##<<<---- not sure if this is necessary but it might be doing something in the background?
        "mkdir /tmp/vllm && "
        "uv pip install pip wheel && "
        f"python -m pip download --only-binary=:all: --no-deps --dest /tmp/vllm vllm==v{VLLM_REF} && "
        # Patch vLLM pre-built download with dynamo additions
        f"cd /tmp/vllm && "
        f"wheel unpack *.whl && "
        f"cd vllm-{VLLM_REF}/ && "
        f"patch -p1 < {DYNAMO_SRC}/container/deps/vllm/{VLLM_PATCH} && "  ##<<<---- GET PATCH FROM SRC!
        # Rename the package from vllm to ai_dynamo_vllm
        f"mv vllm-{VLLM_REF}.dist-info {VLLM_PATCHED_PACKAGE_NAME}-{VLLM_PATCHED_PACKAGE_VERSION}.dist-info && "
        f'sed -i "s/^Name: vllm/Name: {VLLM_PATCHED_PACKAGE_NAME}/g" {VLLM_PATCHED_PACKAGE_NAME}-{VLLM_PATCHED_PACKAGE_VERSION}.dist-info/METADATA && '
        f'sed -i "s/^Version: {VLLM_REF}/Version: {VLLM_PATCHED_PACKAGE_VERSION}/g" {VLLM_PATCHED_PACKAGE_NAME}-{VLLM_PATCHED_PACKAGE_VERSION}.dist-info/METADATA && '
        # Update wheel tag from linux_${ARCH_ALT} to manylinux1_${ARCH_ALT} in WHEEL file
        f'sed -i "s/Tag: cp38-abi3-linux_{ARCH}/Tag: cp38-abi3-manylinux1_{ARCH}/g" {VLLM_PATCHED_PACKAGE_NAME}-{VLLM_PATCHED_PACKAGE_VERSION}.dist-info/WHEEL && '
        # Also update the tag in RECORD file to match
        f'sed -i "s/-cp38-abi3-linux_{ARCH}.whl/-cp38-abi3-manylinux1_{ARCH}.whl/g" {VLLM_PATCHED_PACKAGE_NAME}-{VLLM_PATCHED_PACKAGE_VERSION}.dist-info/RECORD && '
        f"mkdir -p /workspace/dist && "
        f"wheel pack . --dest-dir /workspace/dist && "
        f"uv pip install /workspace/dist/{VLLM_PATCHED_PACKAGE_NAME}-*.whl ;"
    )
    # Common dependencies
    .run_commands(
        f"uv pip install -r {DYNAMO_SRC}/container/deps/requirements.txt && "
        f"uv pip install -r {DYNAMO_SRC}/container/deps/requirements.test.txt"
    )
    # Finish pyright install, enable Git operations in the /workspace directory
    .run_commands(
        "pyright --help > /dev/null 2>&1 && "
        r'printf "[safe]\n      directory=/workspace\n" > /root/.gitconfig && '
        "ln -sf /bin/bash /bin/sh"
    )
    # ########################################################################
    # Rust build/dev dependencies
    .run_commands(
        "apt update -y && "
        "apt install --no-install-recommends -y "
        "build-essential protobuf-compiler cmake libssl-dev pkg-config"
    )
    .env(
        {
            "RUSTUP_HOME": "/usr/local/rustup",
            "CARGO_HOME": "/usr/local/cargo",
            "PATH": "/usr/local/cargo/bin:$PATH",
            "RUST_VERSION": "1.87.0",
            "RUSTARCH": f"{ARCH_ALT}-unknown-linux-gnu",  # Define Rust target based on ARCH_ALT ARG
            "CARGO_BUILD_JOBS": "16",
        }
    )
    # Install Rust using RUSTARCH derived from ARCH_ALT
    .run_commands(
        "cd /workspace && "
        'wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && '
        # TODO: Add SHA check back based on RUSTARCH
        "chmod +x rustup-init && "
        "./rustup-init -y --no-modify-path --profile default --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && "
        "rm rustup-init && "
        "chmod -R a+w $RUSTUP_HOME $CARGO_HOME"
    )
    # ########################################################################
    # Build Wheels
    .env(
        {  # "CARGO_TARGET_DIR": "/workspace/target",
            "DEBIAN_FRONTEND": "noninteractive"
        }
    )
    .run_commands(
        # 1. compile the Rust workspace
        f"cd {DYNAMO_SRC} && "
        "cargo build --release --locked "
        "--features dynamo-llm/block-manager --workspace && "
        # 2. build the pure-Python wheel(s)
        "uv build --wheel --out-dir /tmp/dist && "
        # 3. build the Python bindings (maturin)
        "cd lib/bindings/python && "
        "uv pip install maturin[patchelf] && "
        "maturin build --release --features block-manager --out /tmp/dist && "
        # 4. install the freshly-built wheels into the active venv
        "uv pip install /tmp/dist/ai_dynamo_runtime*.whl && "
        "uv pip install /tmp/dist/ai_dynamo-*.whl"
    )
    # ########################################################################
    # Final runtime cleanup
    .env(
        {
            "DYNAMO_HOME": DYNAMO_HOME,
            "NIXL_PLUGIN_DIR": f"{NIXL_PREFIX}/lib/{ARCH_ALT}-linux-gnu/plugins",
            # need this again? /usr/local/ucx/lib:
            "LD_LIBRARY_PATH": f"{NIXL_PREFIX}/lib/{ARCH_ALT}-linux-gnu:{NIXL_PREFIX}/lib/{ARCH_ALT}-linux-gnu/plugins:$LD_LIBRARY_PATH",
            # Tell vllm to use the Dynamo LLM C API for KV Cache Routing:
            "VLLM_KV_CAPI_PATH": "/opt/dynamo/bindings/lib/libdynamo_llm_capi.so",
        }
    )
    # No need to rebuild, just make sure our images are using the right python
    .run_commands("ln -sf /opt/dynamo/venv/bin/* /usr/local/bin")
    # ########################################################################
    # Cleanup
    # Need this or Modal is thrown off by protobuf>5:
    .run_commands('uv pip install "protobuf==4.25.3" "grpclib~=0.4.7"')
    .env({"NCCL_DEBUG": "INFO"})
    .entrypoint([])
)
import yaml


app = modal.App(
    "vllm-dynamo",
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
    os.environ["VLLM_USE_NVML"] = "0"

    def create_dynamo_yaml(model: str, workdir: Path) -> Path:
        """
        Create a throw-away YAML config that is identical to the stock
        multinode-405b.yaml except for `.model` fields.

        Returns: Path to the new file (placed under workdir/generated/…).
        """
        # --- template stub (only the parts that reference the model) -----------

        out_dir = workdir / "generated"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{model.replace('/', '_')}.yaml"
        if not out_path.is_file():
            template = {
                "Frontend": {
                    "served_model_name": model,
                    "endpoint": "dynamo.Processor.chat/completions",
                    "port": 8000,
                },
                "Processor": {
                    "model": model,
                    "block-size": 64,
                    "max-model-len": 8192,
                    "router": "kv",
                },
                "Router": {
                    "model": model,
                    "min-workers": 1,
                },
                "VllmWorker": {
                    "model": model,
                    "kv-transfer-config": '{"kv_connector":"DynamoNixlConnector"}',
                    "block-size": 64,
                    "max-model-len": 8192,
                    "max-num-seqs": 16,
                    "remote-prefill": True,
                    "conditional-disagg": True,
                    "max-local-prefill-length": 10,
                    "max-prefill-queue-size": 2,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    "router": "kv",
                    # "quantization": "modelopt",
                    "enable-prefix-caching": True,
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                },
                "PrefillWorker": {
                    "model": model,
                    "kv-transfer-config": '{"kv_connector":"DynamoNixlConnector"}',
                    "block-size": 64,
                    "max-model-len": 8192,
                    "max-num-seqs": 16,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    # "quantization": "modelopt",
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                },
            }

            with out_path.open("w") as f:
                yaml.safe_dump(template, f, sort_keys=False)

        return out_path

    print("Cluster is launched/launching...")

    # Get Modal cluster info for inter-container communication
    cluster_info = modal.experimental.get_cluster_info()
    rank: int = cluster_info.rank
    container_id = os.environ["MODAL_TASK_ID"]
    print(f"Node {rank} fired up in container {container_id}!")

    # Set etcd + nats addresses
    head: str = cluster_info.container_ips[0]
    ETCD_URL = f"http://[{head}]:{ETCD_PORT}"
    os.environ.update(
        {
            "NATS_SERVER": f"nats://[{head}]:{NATS_PORT}",
            "ETCD_ENDPOINTS": ETCD_URL,
        }
    )
    me = "127.0.0.1" if rank == 0 else head  # where WE should dial

    # Example configs (to be parameterized...)
    workdir = Path(DYNAMO_SRC) / "examples" / "llm"
    ROUTER = "graphs.agg_router"

    # Jank
    CONFIG_FILE = create_dynamo_yaml(MODEL_NAME, workdir)
    config_exists = CONFIG_FILE.is_file()
    CONFIG_FILE = CONFIG_FILE.as_posix()

    print(
        f"Using auto generated config at: {CONFIG_FILE} which does " + ""
        if config_exists
        else "NOT " + "exist..."
    )

    # = "./configs/multinode-405b.yaml"
    data = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    print(f"\t\t\tRANK{rank}:\n\t{data}")

    if rank == 0:
        # Start sidecars on node 0 only
        subprocess.Popen(
            ["nats-server", "-js", "--trace", f"--port={NATS_PORT}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.Popen(
            [
                "etcd",
                f"--listen-client-urls=http://[::]:{ETCD_PORT}",
                f"--advertise-client-urls={ETCD_URL}",
                # "--data-dir",
                # "/tmp/etcd-data",
                # "--logger=zap",
                # "--log-level=error",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    # Wait for sidecars to spin up before invoking Dynamo
    wait_for_port(me, NATS_PORT, "nats", rank)
    wait_for_port(me, ETCD_PORT, "etcd", rank)

    # Start Dynamo on each node
    if rank == 0:
        subprocess.Popen(
            ["dynamo", "serve", f"{ROUTER}:Frontend", "-f", CONFIG_FILE],
            cwd=workdir,
        )
        wait_for_port("127.0.0.1", DYNAMO_PORT, "dynamo-frontend", rank)

    elif rank == 1:
        subprocess.Popen(
            ["dynamo", "serve", "components.worker:VllmWorker", "-f", CONFIG_FILE],
            cwd=workdir,
            env=os.environ,
        )

    elif rank > 1:
        subprocess.Popen(
            [
                "dynamo",
                "serve",
                "components.prefill_worker:PrefillWorker",
                "-f",
                CONFIG_FILE,
            ],
            cwd=workdir,
            env=os.environ,
        )


def wait_for_port(
    host: str, port: int, service: str, rank: int, timeout_min: float = 10
):
    import socket

    from tqdm import tqdm

    hz = 2
    max_iters = int(timeout_min * 60 * hz)
    bar = tqdm(
        range(max_iters),
        position=rank,  # each rank gets its own row
        desc=f"r{rank}:{service}",
        leave=True,
    )
    start = time.perf_counter()

    for _ in bar:
        try:
            with socket.create_connection((host, port), timeout=1):
                bar.close()
                tqdm.write(
                    f"Rank{rank} found {service} @ {host}:{port} "
                    f"in {time.perf_counter() - start:.2f}s"
                )
                return
        except OSError:
            time.sleep(1 / hz)

    raise RuntimeError(f"Rank{rank} timeout: {service} @ {host}:{port}")


@app.local_entrypoint()
def infer():
    import json
    import time
    import urllib

    MULTI_MODAL = False

    ############################################################################
    # health check
    BASE = dynamo_cluster.get_web_url()  # e.g. "https://…modal.run"
    ENDPOINT = "/v1/chat/completions"

    ############################################################################
    # Send a test request
    if MULTI_MODAL:
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
    else:
        PAYLOAD = {
            "model": "nvidia/Llama-3.1-405B-Instruct-FP8",
            "messages": [
                {
                    "role": "user",
                    # -- text-only prompt instead of image-+-text
                    "content": (
                        "In the heart of Eldoria, an ancient land of boundless magic and "
                        "mysterious creatures, lies the long-forgotten city of Aeloria. "
                        "Once a beacon of knowledge and power, Aeloria was buried beneath "
                        "the shifting sands of time … (full prompt here) …"
                    ),
                }
            ],
            "max_tokens": 300,
            "stream": True,  # leave True if you want server-sent events
        }

    # req = urllib.request.Request(
    #     dynamo_cluster.get_web_url() + endpoint,
    #     data=json.dumps(payload).encode("utf-8"),  # ← fix
    #     method="POST",
    #     headers={"Content-Type": "application/json"},  # ← add header
    # )

    # with urllib.request.urlopen(req, timeout=300) as response:
    #     print(json.loads(response.read().decode()))

    def post_until_ok(
        base: str,
        endpoint: str,
        body: dict,
        give_up_after_s: int = 600,
        pause_s: int = 5,
    ):
        """Keep POSTing until we get a 2XX or the deadline hits."""
        url = f"{base.rstrip('/')}{endpoint}"
        deadline = time.time() + give_up_after_s
        attempt = 0

        while True:
            attempt += 1
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if 200 <= resp.status < 300:  # finally worked 🎉
                        print(f"✓ success on attempt {attempt}")
                        return json.loads(resp.read().decode())

                    # Non-2XX but not an exception (rare)
                    print(f"attempt {attempt}: HTTP {resp.status}; retrying…")

            except urllib.error.HTTPError as e:
                # 500/503/404 → service up but back-end not ready; keep hammering
                if e.code in (500, 503, 404):
                    print(f"attempt {attempt}: HTTP {e.code}; retrying…")
                else:
                    raise  # unexpected

            except Exception as e:
                print(f"attempt {attempt}: connection error ({e}); retrying…")

            if time.time() > deadline:
                raise TimeoutError(
                    f"Gave up after {give_up_after_s}s ({attempt} tries)"
                )

            time.sleep(pause_s)

    # ---------------------------------------------------------------------------
    # fire it up
    reply = post_until_ok(BASE, ENDPOINT, PAYLOAD)
    print("Model answered:\n", reply)
