# ---
# output-directory: "/tmp/world_model"
# args: ["--prompt", "A serene mountain lake at sunrise, mist rising off the water"]
# ---

# # Generate 3D worlds from text with LTX-2.3 and InSpatio-World

# This example shows how to run inference on [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) and [InSpatio-World](https://inspatio.github.io/inspatio-world/) on Modal
# to generate explorable 3D worlds (videos rendered along a camera trajectory) from your local command line and in a web UI.

# The pipeline runs in two stages:
# 1. LTX-2.3 generates a short reference video from the prompt, writes it to a Volume, then spawns stage 2.
# 2. InSpatio-World lifts that video into a 3D scene and re-renders it along a camera trajectory.

# Here is a sample we generated:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/world.mp4" type="video/mp4" />
# </video>
# </center>

# ## Setup

# The LTX text encoder uses Gemma 3 weights that require accepting a license.
# Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized,
# click "Agree and access repository", then create a token at
# https://huggingface.co/settings/tokens and add it as a
# [Modal Secret](https://modal.com/secrets) named `huggingface-secret`
# with key `HF_TOKEN`.

import asyncio
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import modal

app = modal.App("example-world-model")

MINUTES = 60  # seconds

# ## Paths and Volumes

# Model weights and generated videos both live on Volumes, so weights download
# only once and outputs persist across containers and runs.

ARTIFACTS_PATH = "/artifacts"  # where the output Volume is mounted
INSPATIO_WEIGHTS = "/models/inspatio"  # InSpatio weights, on the weights Volume
INSPATIO_REPO = "/opt/inspatio-world"  # InSpatio source, cloned into the image

model_volume = modal.Volume.from_name("world-model-weights", create_if_missing=True)
output_volume = modal.Volume.from_name("world-model-outputs", create_if_missing=True)

frontend_path = Path(__file__).parent / "frontend"

# We pin the LTX-2 commit that supports the LTX-2.3 model architecture.
LTX2_COMMIT = "d6053703e001"
INSPATIO_COMMIT = "fef970664e33f519a31f0ee19d58689e41752c0e"

# Pin model revisions to avoid surprises when upstream repos update.
LTX_REVISION = "76730e634e70a28f4e8d51f5e29c08e40e2d8e74"
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"
INSPATIO_MODEL_REVISION = "f8d1abe227d486be8593825f0611974aa6207e4d"
WAN_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
DA3_REVISION = "8615eefb62f2db4f8d6ebaa59160086981672829"
FLORENCE_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"

# InSpatio renders one camera pose per source frame along this trajectory. The
# three whitespace-separated lines are keyframes for pitch (deg), yaw (deg), and
# displacement, interpolated across the whole clip into one slow look-around.
TRAJECTORY_TXT = (
    "0 0 0\n"  # pitch: stay level
    "0 -12 0 12 0\n"  # yaw: pan right, back to center, left, back to center
    "1 1 1\n"  # displacement: constant orbit radius (no zoom)
)

# ## Tracking progress through the Volume

# The two stages run asynchronously, so we don't return a result directly.
# Instead we track a session's progress by which files exist on the output
# Volume: first `source.mp4` (LTX), then `world.mp4` (InSpatio). Both the web UI
# and the CLI poll this same state.


def session_dir(session_id: str) -> Path:
    return Path(ARTIFACTS_PATH) / session_id


def status_from_files(present: set[str]) -> str:
    if "world.mp4" in present:
        return "done"
    if "source.mp4" in present:
        return "running_inspatio"
    return "running_ltx"


def session_status(session_id: str) -> dict:
    """Read a session's progress and asset URLs from the mounted Volume."""
    base = session_dir(session_id)
    present = {name for name in ("source.mp4", "world.mp4") if (base / name).exists()}
    assets = {
        f"{name.removesuffix('.mp4')}_video": f"/api/assets/{session_id}/{name}"
        for name in present
    }
    return {"status": status_from_files(present), "assets": assets}


def wait_for_file(path: Path, timeout_s: float = 120.0) -> bool:
    """Reload the output Volume until `path` appears (or we time out)."""
    deadline = time.time() + timeout_s
    delay = 1.0
    while True:
        output_volume.reload()
        if path.exists():
            return True
        if time.time() >= deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)


# We also run two small `ffmpeg` helpers on the GPU workers (both images include
# `ffmpeg`). `transcode_to_web_mp4` makes a clip streamable in the browser, and
# `expand_to_frames` stretches the short LTX clip so InSpatio has enough frames
# to render a single, continuous camera move.


def transcode_to_web_mp4(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode {src.name}: {result.stderr[-2000:]}")


def expand_to_frames(src: Path, dst: Path, num_frames: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "setpts=PTS*10,fps=24",
        "-frames:v",
        str(num_frames),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg expand {src.name}: {result.stderr[-2000:]}")


# ## Stage 1: LTX-2.3 reference video

# We build the LTX image from CUDA, installing LTX-2's three subpackages at the
# pinned commit along with a Flash-Attention 3 wheel for faster inference on
# Hopper GPUs.

ltx_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "transformers==4.57.6",
        "huggingface-hub==0.36.2",
        "hf_transfer==0.1.8",
        "fastapi[standard]==0.115.8",
        f"git+https://github.com/Lightricks/LTX-2.git@{LTX2_COMMIT}#subdirectory=packages/ltx-core",
        f"git+https://github.com/Lightricks/LTX-2.git@{LTX2_COMMIT}#subdirectory=packages/ltx-pipelines",
        f"git+https://github.com/Lightricks/LTX-2.git@{LTX2_COMMIT}#subdirectory=packages/ltx-trainer",
        "https://huggingface.co/alexnasa/flash-attn-3/resolve/main/128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",  # reduce fragmentation
        }
    )
    .entrypoint([])
)

with ltx_image.imports():
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
    from ltx_pipelines.utils.media_io import encode_video


# We wrap inference in a [Cls](https://modal.com/docs/guide/lifecycle-functions)
# so the weights download and pipeline build happen once
# per container in `@modal.enter`.


@app.cls(
    image=ltx_image,
    gpu="H200",
    timeout=30 * MINUTES,
    scaledown_window=15 * MINUTES,
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
    volumes={
        "/root/.cache/huggingface": model_volume,
        ARTIFACTS_PATH: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTXInference:
    @modal.enter()
    def setup(self):
        torch.set_float32_matmul_precision("high")

        ltx_repo = "Lightricks/LTX-2.3"
        checkpoint_path = hf_hub_download(
            ltx_repo, "ltx-2.3-22b-dev.safetensors", revision=LTX_REVISION
        )
        upsampler_path = hf_hub_download(
            ltx_repo,
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            revision=LTX_REVISION,
        )
        distilled_lora_path = hf_hub_download(
            ltx_repo,
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            revision=LTX_REVISION,
        )
        gemma_dir = snapshot_download(
            "google/gemma-3-12b-it-qat-q4_0-unquantized", revision=GEMMA_REVISION
        )
        model_volume.commit()

        self.params = detect_params(checkpoint_path)
        self.tiling_config = TilingConfig.default()
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP
                )
            ],
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_dir,
            loras=[],
        )

    @modal.method()
    def run(self, session_id: str, prompt: str) -> str:
        seed = random.randint(0, 2**32 - 1)
        print(f"LTX-2.3 session {session_id}: seed={seed}, 832x512, 25 frames @ 24fps")

        out_dir = session_dir(session_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "source.mp4"

        with torch.no_grad():
            video, audio = self.pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=seed,
                height=512,
                width=832,
                num_frames=25,
                frame_rate=24,
                num_inference_steps=self.params.num_inference_steps,
                video_guider_params=self.params.video_guider_params,
                audio_guider_params=self.params.audio_guider_params,
                images=[],
                tiling_config=self.tiling_config,
                enhance_prompt=True,
            )
            raw_path = out_path.with_suffix(".raw.mp4")
            encode_video(
                video=video,
                fps=24,
                audio=audio,
                output_path=str(raw_path),
                video_chunks_number=get_video_chunks_number(25, self.tiling_config),
            )
            transcode_to_web_mp4(raw_path, out_path)
            raw_path.unlink(missing_ok=True)

        output_volume.commit()
        torch.cuda.empty_cache()

        # Spawn the InSpatio worker to generate the world video without blocking
        InSpatioInference().run.spawn(session_id=session_id, source_path=str(out_path))
        return str(out_path)


# ## Stage 2: InSpatio world generation

# InSpatio-World needs its own pinned dependency stack (Torch 2.5 + CUDA 12.1).
# We clone the upstream repo into the image and point its `checkpoints` directory
# at the weights Volume.

inspatio_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "accelerate==1.13.0",
        "av==13.1.0",
        "decord==0.6.0",
        "depth-anything-3==0.1.1",
        "diffusers==0.37.0",
        "easydict==1.13",
        "einops==0.8.2",
        "ftfy==6.3.1",
        "huggingface-hub==0.36.2",
        "imageio==2.37.3",
        "imageio-ffmpeg==0.6.0",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "open3d==0.19.0",
        "opencv-python==4.11.0.86",
        "pillow==12.0.0",
        "plyfile==1.1",
        "safetensors==0.7.0",
        "scipy==1.15.3",
        "timm==1.0.25",
        "tokenizers==0.22.2",
        "transformers==4.57.6",
        "trimesh==4.11.3",
        "xformers==0.0.29.post1",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        extra_index_url="https://download.pytorch.org/whl/cu121",
        extra_options="--index-strategy unsafe-best-match",
    )
    .run_commands(
        # Clone InSpatio, rename a deprecated `torch_dtype` kwarg for newer
        # transformers, and symlink its checkpoints dir to the weights Volume.
        f"git clone https://github.com/inspatio/inspatio-world.git {INSPATIO_REPO}"
        f" && git -C {INSPATIO_REPO} checkout {INSPATIO_COMMIT}"
        f" && find {INSPATIO_REPO} -name '*.py'"
        " | xargs grep -l 'torch_dtype'"
        " | xargs sed -i 's/torch_dtype=/dtype=/g'"
        f" && rm -rf {INSPATIO_REPO}/checkpoints"
        f" && ln -s {INSPATIO_WEIGHTS} {INSPATIO_REPO}/checkpoints"
    )
    .entrypoint([])
)


@app.cls(
    image=inspatio_image,
    gpu="H200",
    timeout=90 * MINUTES,
    scaledown_window=10 * MINUTES,
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
    volumes={
        INSPATIO_WEIGHTS: model_volume,
        ARTIFACTS_PATH: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class InSpatioInference:
    @modal.enter()
    def setup(self):
        """Download InSpatio checkpoints to the weights Volume on first container start."""
        from huggingface_hub import snapshot_download

        sentinel = Path(INSPATIO_WEIGHTS) / ".ready"
        if sentinel.exists():
            return

        print("Downloading InSpatio-World checkpoints (first run only)...")
        weights = Path(INSPATIO_WEIGHTS)
        weights.mkdir(parents=True, exist_ok=True)

        # taehv lives only on GitHub; the rest are Hugging Face repos.
        taehv = weights / "taehv"
        shutil.rmtree(taehv, ignore_errors=True)
        repo = "https://github.com/madebyollin/taehv.git"
        subprocess.run(["git", "clone", "--depth", "1", repo, str(taehv)], check=True)
        for repo_id, dest, rev in [
            ("inspatio/world", "InSpatio-World-1.3B", INSPATIO_MODEL_REVISION),
            ("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B", WAN_REVISION),
            ("depth-anything/DA3NESTED-GIANT-LARGE", "DA3", DA3_REVISION),
            ("microsoft/Florence-2-large", "Florence-2-large", FLORENCE_REVISION),
        ]:
            snapshot_download(repo_id, local_dir=str(weights / dest), revision=rev)

        sentinel.write_text("ok")
        model_volume.commit()

    @modal.method()
    def warmup(self) -> str:
        """Boot the container (and download weights on first run) ahead of time."""
        return "ok"

    @modal.method()
    def run(self, session_id: str, source_path: str) -> None:
        work = session_dir(session_id) / "_work"
        input_dir = work / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Wait for LTX's video to propagate to this container, then hold it out to
        # 240 frames so InSpatio renders one continuous 10s pan (at 24fps).
        source = Path(source_path)
        if not wait_for_file(source):
            raise FileNotFoundError(f"source video missing: {source}")
        expand_to_frames(source, input_dir / "source.mp4", 240)

        traj_path = work / "trajectory.txt"
        traj_path.write_text(TRAJECTORY_TXT)

        output_folder = work / "output" / "world"
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"InSpatio session {session_id}: gentle look-around")
        cmd = [
            "bash",
            f"{INSPATIO_REPO}/run_test_pipeline.sh",
            "--input_dir",
            str(input_dir),
            "--traj_txt_path",
            str(traj_path),
            "--checkpoint_path",
            f"{INSPATIO_WEIGHTS}/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors",
            "--config_path",
            f"{INSPATIO_REPO}/configs/inference_1.3b.yaml",
            "--da3_model_path",
            f"{INSPATIO_WEIGHTS}/DA3",
            "--florence_model_path",
            f"{INSPATIO_WEIGHTS}/Florence-2-large",
            "--output_folder",
            str(output_folder),
            "--disable_adaptive_frame",  # one pose per frame, no bounce/subsample
        ]
        result = subprocess.run(cmd, cwd=INSPATIO_REPO)
        if result.returncode != 0:
            raise RuntimeError(f"InSpatio pipeline exit code {result.returncode}")

        world_src = next(iter(sorted(output_folder.rglob("*pred_video*.mp4"))), None)
        if not world_src:
            raise RuntimeError("InSpatio produced no world video")
        transcode_to_web_mp4(world_src, session_dir(session_id) / "world.mp4")
        shutil.rmtree(work, ignore_errors=True)
        output_volume.commit()


# ## Web UI

# A small [ASGI app](https://modal.com/docs/guide/webhooks) serves the frontend
# and exposes the session lifecycle: start a session (spawn the LTX worker and
# warm InSpatio), poll its status by reading the Volume, and serve the videos the
# workers write there.

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "jinja2==3.1.5", "fastapi[standard]==0.115.8", "python-multipart==0.0.20"
    )
    .add_local_dir(frontend_path, remote_path="/assets")
)

ASSET_NAMES = frozenset({"source.mp4", "world.mp4"})


@app.function(image=web_image, volumes={ARTIFACTS_PATH: output_volume})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    import fastapi.templating

    web_app = fastapi.FastAPI()
    templates = fastapi.templating.Jinja2Templates(directory="/assets")

    # `Volume.reload()` fails while any file on the Volume is open in this
    # container, so this lock serializes reloads (and the asset copy-out below)
    # across the many concurrent requests this single container handles.
    volume_lock = asyncio.Lock()

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_name": "World Model (LTX-2.3 + InSpatio)",
                "default_prompt": "A serene mountain lake at sunrise, mist rising off the water",
            },
        )

    @web_app.post("/api/sessions")
    async def start_session(prompt: str = fastapi.Form(...)):
        session_id = uuid.uuid4().hex[:12]
        # Warm InSpatio in parallel with LTX generation; the LTX worker spawns
        # the InSpatio run itself once its video is on the Volume.
        await InSpatioInference().warmup.spawn.aio()
        await LTXInference().run.spawn.aio(session_id=session_id, prompt=prompt)
        return {"session_id": session_id}

    @web_app.get("/api/sessions/{session_id}")
    async def session_state(session_id: str):
        async with volume_lock:
            await output_volume.reload.aio()
        if not session_dir(session_id).exists():
            return fastapi.responses.JSONResponse(
                {"error": "not found"}, status_code=404
            )
        state = session_status(session_id)
        return fastapi.responses.JSONResponse(
            state, status_code=200 if state["status"] == "done" else 202
        )

    @web_app.get("/api/assets/{session_id}/{filename}")
    async def serve_asset(session_id: str, filename: str):
        if filename not in ASSET_NAMES:
            return fastapi.responses.JSONResponse({"error": "unknown"}, status_code=404)

        vol_path = session_dir(session_id) / filename
        if not vol_path.exists():
            async with volume_lock:
                await output_volume.reload.aio()
        if not vol_path.exists():
            return fastapi.responses.JSONResponse(
                {"error": "not ready"}, status_code=404
            )

        stat = vol_path.stat()
        local_path = (
            Path("/tmp/asset_cache")
            / session_id
            / f"{stat.st_mtime_ns}_{stat.st_size}_{filename}"
        )
        if not local_path.exists():
            async with volume_lock:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vol_path, local_path)

        return fastapi.responses.FileResponse(
            local_path, media_type="video/mp4", content_disposition_type="inline"
        )

    web_app.mount(
        "/static", fastapi.staticfiles.StaticFiles(directory="/assets"), name="static"
    )
    return web_app


# ## Command line

# Run the pipeline from your terminal with

# ```bash
# modal run image_to_world.py --prompt "A serene mountain lake at sunrise, mist rising off the water"
# ```

# This starts a session, watches the output Volume until the world video shows
# up, and prints a link to the Volume dashboard where both videos can be viewed.


@app.local_entrypoint()
def entrypoint(prompt: str):
    session_id = uuid.uuid4().hex[:12]
    print(f"Starting world session {session_id}")
    print(f"  prompt: {prompt}")

    InSpatioInference().warmup.spawn()
    LTXInference().run.spawn(session_id=session_id, prompt=prompt)

    def list_session_files() -> set[str]:
        try:
            return {Path(e.path).name for e in output_volume.listdir(session_id)}
        except Exception:
            return set()

    start, last_status = time.time(), None
    while last_status != "done":
        status = status_from_files(list_session_files())
        if status != last_status:
            print(f"  [{time.time() - start:6.1f}s] {status}")
            last_status = status
        if status != "done":
            time.sleep(10)

    output_volume.hydrate()
    print("\nWorld ready. View the videos on the Modal Volume dashboard:")
    print(f"  https://modal.com/id/{output_volume.object_id}")
    print(f"This run's files live under {session_id}/ :")
    print(f"  {session_id}/source.mp4   (LTX video)")
    print(f"  {session_id}/world.mp4    (InSpatio world video)")
