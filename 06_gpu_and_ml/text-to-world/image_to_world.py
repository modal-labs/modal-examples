# ---
# output-directory: "/tmp/world_model"
# args: ["--prompt", "A serene mountain lake at sunrise, mist rising off the water"]
# ---

# # Text-to-world video generation with LTX-2.3 and InSpatio

# This example chains two diffusion models on Modal to turn a text prompt
# into an explorable 3D "world": a video rendered along a camera trajectory,
# plus the point cloud it was built from.

# The pipeline runs in two stages:
# 1. [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) generates a short
#    reference video from the prompt, writes it to a Volume, then spawns stage 2.
# 2. [InSpatio-World](https://huggingface.co/inspatio/world) lifts that video
#    into a 3D scene and re-renders it along a specified camera trajectory.
#
# The client (CLI or web UI) starts a session by warming the InSpatio worker and
# spawning the LTX worker, then watches the Volume for the resulting files.

# ## Setup

# The LTX text encoder uses Gemma 3 weights that require accepting a license.
# Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized,
# click "Agree and access repository", then create a token at
# https://huggingface.co/settings/tokens and add it as a
# [Modal Secret](https://modal.com/secrets) named `huggingface-secret`
# with key `HF_TOKEN`.

# Deploy the web UI with:
# ```bash
# modal deploy image_to_world.py
# ```

# Or generate a single world from the CLI:
# ```bash
# modal run image_to_world.py --prompt "..." --trajectory x_y_circle_cycle
# ```

# Retrieve CLI outputs from the Modal Volume:
# ```bash
# modal volume ls world-model-outputs
# modal volume get world-model-outputs <session_id>/world.mp4
# ```

# ## Environment setup

import asyncio
import random
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import modal

app = modal.App("world-model")

# ## Paths and volumes

# Model weights are cached to Volumes so they download only once. 
# Generated videos are written to a separate output Volume.

ARTIFACTS_PATH = "/artifacts"
INSPATIO_WEIGHTS = "/models/inspatio"
INSPATIO_REPO = "/opt/inspatio-world"

model_volume = modal.Volume.from_name("world-model-weights", create_if_missing=True)
output_volume = modal.Volume.from_name("world-model-outputs", create_if_missing=True)

# We pin an LTX-2 commit that supports the LTX-2.3 model architecture.
LTX2_COMMIT = "d6053703e001"

DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 25
DEFAULT_FPS = 24

def align_num_frames(n: int) -> int:
    """LTX-2.3 requires frame count = 8n + 1."""
    return max(9, ((n - 1) // 8) * 8 + 1)


def align_dimension(n: int) -> int:
    """LTX-2.3 two-stage pipeline requires width/height divisible by 64."""
    return max(64, (n // 64) * 64)


TRAJECTORY_PRESETS = [
    {"id": "x_y_circle_cycle", "label": "Orbit (pitch + yaw circle)"},
    {"id": "zoom_out_in", "label": "Dolly zoom out + in"},
]

frontend_path = Path(__file__).parent / "frontend"

# Each session owns a directory on the output Volume.

def session_dir(session_id: str) -> Path:
    return Path(ARTIFACTS_PATH) / session_id

# Output filenames the workers write to the session root, mapped to the asset
# keys the frontend consumes.
SESSION_ASSETS = {
    "source_video": "source.mp4",
    "world_video": "world.mp4",
    "conditioning_video": "conditioning.mp4",
    "pointcloud_ply": "pointcloud.ply",
}


def status_from_files(present: set[str]) -> str:
    """Map the set of filenames present in a session dir to a progress label."""
    if "error.txt" in present:
        return "error"
    if "world.mp4" in present:
        return "done"
    if "source.mp4" in present:
        return "running_inspatio"
    return "running_ltx"


def session_status(session_id: str) -> dict:
    """Infer a session's progress from which files exist on the Volume.

    For use inside containers, where the output Volume is mounted at
    ``ARTIFACTS_PATH``. The CLI runs locally and instead lists the Volume over
    the client API (see ``entrypoint``).
    """
    base = session_dir(session_id)
    present = {
        name
        for name in (*SESSION_ASSETS.values(), "error.txt")
        if (base / name).exists()
    }
    assets = {
        key: f"/api/assets/{session_id}/{name}"
        for key, name in SESSION_ASSETS.items()
        if name in present
    }
    status = status_from_files(present)
    error = (base / "error.txt").read_text() if status == "error" else None
    return {"status": status, "error": error, "assets": assets}


def wait_for_file(path: Path, timeout_s: float = 120.0) -> bool:
    """Reload the output Volume until ``path`` shows up"""
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


def transcode_to_web_mp4(src: Path, dst: Path) -> None:
    """Re-encode to H.264 + AAC, yuv420p, with the MOOV atom at the front so
    browsers can stream and seek the video."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            "-c:a", "aac",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode {src.name}: {result.stderr[-2000:]}")


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
        "transformers>=4.52,<5",
        "huggingface-hub>=0.36",
        "hf_transfer>=0.1.8",
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
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
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

# The worker downloads weights once into the cache Volume.

@app.cls(
    image=ltx_image,
    gpu="H200",
    timeout=30 * 60,
    scaledown_window=15 * 60,
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
        """Download model weights to the HF cache Volume and init the pipeline."""
        torch.set_float32_matmul_precision("high")

        ltx_repo = "Lightricks/LTX-2.3"
        checkpoint_path = hf_hub_download(ltx_repo, "ltx-2.3-22b-dev.safetensors")
        upsampler_path = hf_hub_download(
            ltx_repo, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = hf_hub_download(
            ltx_repo, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        gemma_dir = snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized")
        model_volume.commit()

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ]
        self.params = detect_params(checkpoint_path)
        self.tiling_config = TilingConfig.default()
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_dir,
            loras=[],
        )

    @modal.method()
    def run(
        self,
        session_id: str,
        prompt: str,
        trajectory: str = "x_y_circle_cycle",
        negative_prompt: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        fps: int = DEFAULT_FPS,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        try:
            width = align_dimension(width)
            height = align_dimension(height)
            num_frames = align_num_frames(num_frames)
            seed = seed if seed is not None else random.randint(0, 2**32 - 1)
            steps = num_inference_steps or self.params.num_inference_steps
            negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

            print(
                f"LTX-2.3 session {session_id}: seed={seed}, {width}x{height}, "
                f"{num_frames} frames @ {fps}fps"
            )

            out_dir = session_dir(session_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "source.mp4"

            with torch.no_grad():
                video, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    num_inference_steps=steps,
                    video_guider_params=self.params.video_guider_params,
                    audio_guider_params=self.params.audio_guider_params,
                    images=[],
                    tiling_config=self.tiling_config,
                    enhance_prompt=True,
                )
                raw_path = out_path.with_suffix(".raw.mp4")
                encode_video(
                    video=video,
                    fps=fps,
                    audio=audio,
                    output_path=str(raw_path),
                    video_chunks_number=get_video_chunks_number(
                        num_frames, self.tiling_config
                    ),
                )
                transcode_to_web_mp4(raw_path, out_path)
                raw_path.unlink(missing_ok=True)

            output_volume.commit()
            torch.cuda.empty_cache()
        except Exception as e:
            raise

        # Hand off to InSpatio without blocking: the LTX worker (an H200) returns
        # immediately rather than idling while InSpatio runs on its own container.
        InSpatioInference().run.spawn(
            session_id=session_id, trajectory=trajectory, source_path=str(out_path)
        )
        return str(out_path)


# ## Stage 2: InSpatio world generation

# InSpatio-World needs its own pinned dependency stack (Torch 2.5 + CUDA 12.1)
# and clones the upstream repo into the image.

inspatio_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
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
        f"git clone --depth 1 https://github.com/inspatio/inspatio-world.git {INSPATIO_REPO}"
        f" && find {INSPATIO_REPO} -name '*.py'"
        " | xargs grep -l 'torch_dtype'"
        " | xargs sed -i 's/torch_dtype=/dtype=/g'"
        f" && rm -rf {INSPATIO_REPO}/checkpoints"
        f" && ln -s {INSPATIO_WEIGHTS} {INSPATIO_REPO}/checkpoints"
    )
)

# The worker downloads several checkpoints (InSpatio, Wan2.1, DA3, Florence-2,
# taehv) to the weights Volume on first start. We access weights on HuggingFace.
# We git clone taehv, which lives only on GitHub.


@app.cls(
    image=inspatio_image,
    gpu="H200",
    timeout=90 * 60,
    scaledown_window=10 * 60,
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

        def clone_taehv():
            dest = weights / "taehv"
            if dest.exists():
                shutil.rmtree(dest)
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/madebyollin/taehv.git", str(dest)],
                check=True,
            )

        hf_repos = [
            ("inspatio/world", "InSpatio-World-1.3B"),
            ("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B"),
            ("depth-anything/DA3NESTED-GIANT-LARGE", "DA3"),
            ("microsoft/Florence-2-large", "Florence-2-large"),
        ]
        # Clone taehv and download HF repos.
        with ThreadPoolExecutor(max_workers=1) as pool:
            taehv_future = pool.submit(clone_taehv)
            for repo_id, dest in hf_repos:
                snapshot_download(repo_id, local_dir=str(weights / dest))
            taehv_future.result()

        sentinel.write_text("ok")
        model_volume.commit()
        print("InSpatio checkpoints cached.")

    @modal.method()
    def warmup(self) -> str:
        """Boot the container (and download weights on first run) ahead of time."""
        return "ok"

    @modal.method()
    def run(
        self,
        session_id: str,
        trajectory: str,
        source_path: Optional[str] = None,
    ) -> None:
        try:
            work = session_dir(session_id) / "_work"
            input_dir = work / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            dest = input_dir / "source.mp4"

            if not source_path:
                raise ValueError("source_path required")
            source = Path(source_path)
            if not wait_for_file(source):
                raise FileNotFoundError(f"source video missing: {source}")
            shutil.copy2(source, dest)

            traj_path = Path(INSPATIO_REPO) / "traj" / f"{trajectory}.txt"
            if not traj_path.exists():
                raise FileNotFoundError(f"Trajectory not found: {traj_path}")

            output_folder = work / "output" / trajectory
            output_folder.mkdir(parents=True, exist_ok=True)

            checkpoint = (
                Path(INSPATIO_WEIGHTS)
                / "InSpatio-World-1.3B"
                / "InSpatio-World-1.3B.safetensors"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(f"InSpatio checkpoint missing at {checkpoint}")

            cmd = [
                "bash",
                f"{INSPATIO_REPO}/run_test_pipeline.sh",
                "--input_dir",
                str(input_dir),
                "--traj_txt_path",
                str(traj_path),
                "--checkpoint_path",
                str(checkpoint),
                "--config_path",
                f"{INSPATIO_REPO}/configs/inference_1.3b.yaml",
                "--da3_model_path",
                f"{INSPATIO_WEIGHTS}/DA3",
                "--florence_model_path",
                f"{INSPATIO_WEIGHTS}/Florence-2-large",
                "--output_folder",
                str(output_folder),
                "--disable_adaptive_frame",
            ]

            print(f"InSpatio session {session_id}: trajectory={trajectory}")
            result = subprocess.run(
                cmd,
                cwd=INSPATIO_REPO,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"InSpatio pipeline failed with exit code {result.returncode}"
                )

            # Transcode the chosen outputs to web-ready files at the session
            # root, then drop the scratch dir.
            package_outputs(session_id, output_folder)
            shutil.rmtree(work, ignore_errors=True)
            output_volume.commit()
        except Exception as e:
            raise


# After the InSpatio pipeline runs, `package_outputs` picks the relevant videos
# and point cloud out of its output directory and transcodes the videos for the
# browser. It runs inside the InSpatio worker, which already has ffmpeg.


def _pick_videos(output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
    if not mp4s:
        return None, None

    conditioning = None
    world = None
    for p in mp4s:
        name = p.name.lower()
        if any(k in name for k in ("render", "condition", "point", "mask", "depth")):
            conditioning = conditioning or p
        elif any(k in name for k in ("output", "pred", "novel", "result", "gen")):
            world = world or p

    if world is None:
        world = mp4s[0]
    if conditioning is None and len(mp4s) > 1:
        conditioning = mp4s[1]
    return world, conditioning


def _pick_pointcloud(output_dir: Path) -> Optional[Path]:
    plys = sorted(output_dir.rglob("*.ply"), key=lambda p: p.stat().st_size, reverse=True)
    return plys[0] if plys else None


def package_outputs(session_id: str, output_dir: Path) -> None:
    """Transcode the InSpatio outputs into web-ready files at the session root."""
    base = session_dir(session_id)
    world_src, cond_src = (
        _pick_videos(output_dir) if output_dir.exists() else (None, None)
    )

    # Encode InSpatio outputs to H.264 + faststart.
    if world_src:
        transcode_to_web_mp4(world_src, base / "world.mp4")
    if cond_src:
        transcode_to_web_mp4(cond_src, base / "conditioning.mp4")

    ply_src = _pick_pointcloud(output_dir) if output_dir.exists() else None
    if ply_src:
        shutil.copy2(ply_src, base / "pointcloud.ply")


# ## Web UI

# A small ASGI app serves the frontend and exposes the session lifecycle: start
# a session (which spawns the LTX worker and warms InSpatio), poll its status by
# reading the Volume, and serve the files the workers write there.

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("jinja2==3.1.5", "fastapi[standard]==0.115.8", "python-multipart==0.0.20")
    .add_local_dir(frontend_path, remote_path="/assets")
)

ASSET_NAMES = frozenset(
    {"source.mp4", "world.mp4", "conditioning.mp4", "pointcloud.ply"}
)


@app.function(
    image=web_image,
    volumes={ARTIFACTS_PATH: output_volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import fastapi.responses
    import fastapi.staticfiles
    import fastapi.templating

    web_app = fastapi.FastAPI()
    templates = fastapi.templating.Jinja2Templates(directory="/assets")

    # `Volume.reload()` fails if any file on the Volume is open in this container.
    # This lock serializes reloads (and the asset copy-out below) so a reload can
    # never overlap an open Volume file while many requests run concurrently.
    volume_lock = asyncio.Lock()

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_name": "World Model (LTX-2.3 + InSpatio)",
                "default_prompt": (
                    "A serene mountain lake at sunrise, mist rising off the water"
                ),
                "trajectories": TRAJECTORY_PRESETS,
                "default_width": DEFAULT_WIDTH,
                "default_height": DEFAULT_HEIGHT,
                "default_num_frames": DEFAULT_NUM_FRAMES,
                "default_fps": DEFAULT_FPS,
            },
        )

    @web_app.post("/api/sessions")
    async def start_session(
        prompt: str = fastapi.Form(...),
        trajectory: str = fastapi.Form("x_y_circle_cycle"),
        seed: Optional[int] = fastapi.Form(None),
        width: int = fastapi.Form(DEFAULT_WIDTH),
        height: int = fastapi.Form(DEFAULT_HEIGHT),
        num_frames: int = fastapi.Form(DEFAULT_NUM_FRAMES),
        fps: int = fastapi.Form(DEFAULT_FPS),
    ):
        session_id = uuid.uuid4().hex[:12]

        # Warm the InSpatio worker (container boot + weight verify + checkpoint
        # linking) in parallel with LTX generation. The LTX worker spawns the
        # InSpatio run itself once its video is on the Volume.
        await InSpatioInference().warmup.spawn.aio()
        await LTXInference().run.spawn.aio(
            session_id=session_id,
            prompt=prompt,
            trajectory=trajectory,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
        )
        return {"session_id": session_id}

    @web_app.get("/api/sessions/{session_id}")
    async def session_state(session_id: str):
        async with volume_lock:
            await output_volume.reload.aio()
        if not session_dir(session_id).exists():
            return fastapi.responses.JSONResponse(
                {"error": "session not found"},
                status_code=404,
            )
        state = session_status(session_id)
        code = 200 if state["status"] in ("done", "error") else 202
        return fastapi.responses.JSONResponse(state, status_code=code)

    @web_app.get("/api/assets/{session_id}/{filename}")
    async def serve_asset(session_id: str, filename: str):
        if filename not in ASSET_NAMES:
            return fastapi.responses.JSONResponse(
                {"error": "unknown asset"},
                status_code=404,
            )

        vol_path = session_dir(session_id) / filename
        if not vol_path.exists():  # cold container: reload once to see the asset
            async with volume_lock:
                if not vol_path.exists():
                    await output_volume.reload.aio()
            if not vol_path.exists():
                return fastapi.responses.JSONResponse(
                    {"error": "asset not ready"},
                    status_code=404,
                )

        # Copy to local disk and stream from there, so we never hold a file open
        # on the Volume (which would break concurrent reloads). Key the cache by
        # (mtime, size) so an overwritten file is served fresh.
        stat = vol_path.stat()
        local_path = (
            Path("/tmp/asset_cache")
            / session_id
            / f"{stat.st_mtime_ns}_{stat.st_size}_{filename}"
        )
        if not local_path.exists():
            async with volume_lock:
                if not local_path.exists():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = local_path.with_name(local_path.name + ".part")
                    shutil.copy2(vol_path, tmp)
                    tmp.rename(local_path)

        media = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
        return fastapi.responses.FileResponse(
            local_path, media_type=media, content_disposition_type="inline"
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


# ## CLI

# `modal run image_to_world.py --prompt "..."` starts a session, warming the
# InSpatio worker and spawning the LTX worker (which in turn spawns InSpatio).
# The client then watches the output Volume by file presence until the world
# video shows up, and prints a link to the Volume dashboard where the LTX and
# InSpatio videos can be viewed.


@app.local_entrypoint()
def entrypoint(
    prompt: str,
    trajectory: str = "x_y_circle_cycle",
    seed: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
):
    session_id = uuid.uuid4().hex[:12]

    print(f"Starting world session {session_id}")
    print(f"  prompt: {prompt}")
    print(f"  trajectory: {trajectory}")

    InSpatioInference().warmup.spawn()
    LTXInference().run.spawn(
        session_id=session_id,
        prompt=prompt,
        trajectory=trajectory,
        seed=seed,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
    )

    def list_session_files() -> set[str]:
        # `reload()` only works inside a container, so the local CLI lists the
        # Volume over the client API instead.
        try:
            return {Path(e.path).name for e in output_volume.listdir(session_id)}
        except Exception:
            return set()

    start = time.time()
    last_status = None
    while True:
        status = status_from_files(list_session_files())
        if status != last_status:
            print(f"  [{time.time() - start:6.1f}s] {status}")
            last_status = status
        if status in ("done", "error"):
            break
        time.sleep(10)

    if last_status == "error":
        try:
            message = b"".join(
                output_volume.read_file(f"{session_id}/error.txt")
            ).decode()
        except Exception:
            message = "unknown error"
        raise RuntimeError(message)

    # The videos stay on the output Volume. Print a link to its dashboard so the
    # LTX source and InSpatio world videos can be viewed without downloading.
    output_volume.hydrate()
    dashboard_url = f"https://modal.com/id/{output_volume.object_id}"
    print("\nWorld ready. View the videos on the Modal Volume dashboard:")
    print(f"  {dashboard_url}")
    print(f"This run's files live under {session_id}/ :")
    print(f"  {session_id}/source.mp4   (LTX video)")
    print(f"  {session_id}/world.mp4    (InSpatio world video)")
