import os
import time

import modal

GPU = "h100"
CPU = 8
MOUNT_ROOT_DIR = "/workspace_sgl"
app = modal.App("sglang-server")

model_volume = modal.Volume.from_name("sglang", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/test/cu128",
    )
    .pip_install(
        "transformers",
        extra_options="-U",
    )
    .apt_install("git", "git-lfs", "wget", "curl", "libnuma-dev")
    .run_commands(
        "git clone https://github.com/sgl-project/sglang && cd sglang && pip3 install pip --upgrade && pip3 install -e 'python[all]'",
        gpu=GPU,
    )
    .pip_install(
        "sgl-kernel==0.3.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/workspace_sgl/hf_home"})
)
with image.imports():
    import torch
    from sglang.srt.entrypoints.engine import _launch_subprocesses
    from sglang.srt.entrypoints.http_server import (
        _GlobalState,
        app as sglang_app,
        set_global_state,
    )
    from sglang.srt.metrics.func_timer import enable_func_timer
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import add_api_key_middleware, add_prometheus_middleware


@app.cls(
    image=image,
    gpu=GPU,
    cpu=CPU,
    volumes={MOUNT_ROOT_DIR: model_volume},
    allow_concurrent_inputs=100,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SGLangServer:
    model_path = "lmsys/gpt-oss-20b-bf16"

    @modal.enter(snap=True)
    def load_model(self):
        """Load model into CPU memory for memory snapshot."""
        start_time = time.time()

        self.server_args = ServerArgs(
            model_path=self.model_path,
            host="0.0.0.0",
            port=8000,
        )
        print(
            f"SGLang model loaded into CPU memory in {time.time() - start_time} seconds"
        )

        # Verify CUDA availability and device
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")

        print(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )

        try:
            self.tokenizer_manager, self.template_manager, self.scheduler_info = (
                _launch_subprocesses(server_args=self.server_args)
            )

            set_global_state(
                _GlobalState(
                    tokenizer_manager=self.tokenizer_manager,
                    template_manager=self.template_manager,
                    scheduler_info=self.scheduler_info,
                )
            )

            self.sglang_app = sglang_app

            if self.server_args.api_key:
                add_api_key_middleware(self.sglang_app, self.server_args.api_key)

            if self.server_args.enable_metrics:
                add_prometheus_middleware(self.sglang_app)
                enable_func_timer()

            self.sglang_app.server_args = self.server_args

            print(
                f"SGLang server initialized on GPU in {time.time() - start_time} seconds"
            )
            print(f"SGLang server ready with model: {self.server_args.model_path}")

        except Exception as e:
            print(f"Error initializing SGLang server: {e}")
            raise

    @modal.asgi_app()
    def asgi_app(self):
        return self.sglang_app
