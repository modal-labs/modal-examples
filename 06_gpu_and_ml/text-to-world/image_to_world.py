# ---
# output-directory: "/tmp/world_model"
# args: ["--prompt", "A serene mountain lake at sunrise, mist rising off the water"]
# ---

# # Image-to-world video generation with LTX-2.3 and InSpatio

# This example chains two diffusion models on Modal to turn a text prompt
# (and an optional reference image) into an explorable 3D "world": a video
# rendered along a camera trajectory, plus the point cloud it was built from.

# The pipeline runs in two stages:
# 1. [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) generates a short
#    reference video from the prompt/image.
# 2. [InSpatio-World](https://huggingface.co/inspatio/world) lifts that video
#    into a 3D scene and re-renders it along a specified camera trajectory.

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
# modal run image_to_world.py --prompt "..." --image-path /path/to/image.jpg
# ```

# Retrieve CLI outputs from the Modal Volume:
# ```bash
# modal volume ls world-model-outputs
# modal volume get world-model-outputs <job_id>/packaged/world.mp4
# ```

# ## Environment setup

import json
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

import modal

app = modal.App("world-model")

MINUTES = 60

# ## Paths and volumes

# Model weights are cached to Volumes so they download only once. Generated
# artifacts for every job are written to a separate output Volume.

HF_CACHE_PATH = "/root/.cache/huggingface"
ARTIFACTS_PATH = "/artifacts"
INSPATIO_WEIGHTS = "/models/inspatio"
INSPATIO_REPO = "/opt/inspatio-world"

ltx_model_volume = modal.Volume.from_name("world-model-ltx-models", create_if_missing=True)
output_volume = modal.Volume.from_name("world-model-outputs", create_if_missing=True)
inspatio_model_volume = modal.Volume.from_name(
    "world-model-inspatio-weights", create_if_missing=True
)

# We pin the LTX-2.3 model repo and an LTX-2 commit that supports its architecture.

LTX23_REPO = "Lightricks/LTX-2.3"
LTX23_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"
LTX23_UPSAMPLER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX23_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
LTX2_COMMIT = "d6053703e001"  # latest main as of 2026-05-28

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


# Each job owns a directory on the output Volume, with a `manifest.json` that
# tracks its status and asset URLs. The web UI polls this manifest, so every
# stage updates it as it progresses.


def job_dir(job_id: str) -> Path:
    return Path(ARTIFACTS_PATH) / job_id


def _apply_manifest(job_id: str, **fields: Any) -> dict:
    """Merge fields into the on-disk manifest."""
    path = job_dir(job_id) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(fields)
    data.setdefault("job_id", job_id)
    path.write_text(json.dumps(data, indent=2))
    return data


def write_manifest(job_id: str, **fields: Any) -> dict:
    """Synchronous manifest write + commit, for use from sync (remote) functions."""
    data = _apply_manifest(job_id, **fields)
    output_volume.commit()
    return data


async def write_manifest_aio(job_id: str, **fields: Any) -> dict:
    """Async manifest write + commit, for use from async (web) endpoints."""
    data = _apply_manifest(job_id, **fields)
    await output_volume.commit.aio()
    return data


def read_manifest(job_id: str) -> dict:
    path = job_dir(job_id) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
    from ltx_pipelines.utils.media_io import encode_video

# The worker downloads weights once into the cache Volume.

@app.cls(
    image=ltx_image,
    gpu="H200",
    timeout=30 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={
        HF_CACHE_PATH: ltx_model_volume,
        ARTIFACTS_PATH: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTXInference:
    @modal.enter()
    def setup(self):
        """Download model weights to the HF cache Volume and init the pipeline."""
        torch.set_float32_matmul_precision("high")

        checkpoint_path = hf_hub_download(LTX23_REPO, LTX23_CHECKPOINT)
        upsampler_path = hf_hub_download(LTX23_REPO, LTX23_UPSAMPLER)
        distilled_lora_path = hf_hub_download(LTX23_REPO, LTX23_DISTILLED_LORA)
        gemma_dir = snapshot_download(GEMMA_REPO)
        ltx_model_volume.commit()

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
        job_id: str,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        negative_prompt: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        fps: int = DEFAULT_FPS,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        width = align_dimension(width)
        height = align_dimension(height)
        num_frames = align_num_frames(num_frames)
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        steps = num_inference_steps or self.params.num_inference_steps
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        print(
            f"LTX-2.3 job {job_id}: seed={seed}, {width}x{height}, "
            f"{num_frames} frames @ {fps}fps"
        )

        images: list[ImageConditioningInput] = []
        tmp_image_path: Optional[str] = None
        if image_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(image_bytes)
            tmp.close()
            tmp_image_path = tmp.name
            images = [ImageConditioningInput(path=tmp_image_path, frame_idx=0, strength=0.8)]

        out_dir = job_dir(job_id) / "ltx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "source.mp4"

        try:
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
                    images=images,
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
        finally:
            if tmp_image_path:
                Path(tmp_image_path).unlink(missing_ok=True)

        output_volume.commit()
        torch.cuda.empty_cache()
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
        "git-lfs",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .run_commands("git lfs install")
    .run_commands(
        "pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .uv_pip_install(
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
    )
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/inspatio/inspatio-world.git /opt/inspatio-world"
        " && find /opt/inspatio-world -name '*.py'"
        " | xargs grep -l 'torch_dtype'"
        " | xargs sed -i 's/torch_dtype=/dtype=/g'"
    )
)

# The worker downloads several checkpoints (InSpatio, Wan2.1, DA3, Florence-2,
# taehv) to a persistent Volume on first start, then symlinks them into the repo
# layout it expects on every container start.


@app.cls(
    image=inspatio_image,
    gpu="H100",
    timeout=90 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={
        INSPATIO_WEIGHTS: inspatio_model_volume,
        ARTIFACTS_PATH: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class InSpatioInference:
    @modal.enter()
    def setup(self):
        """Download InSpatio checkpoints to the weights Volume on first container start."""
        import os

        sentinel = Path(INSPATIO_WEIGHTS) / ".ready"
        if sentinel.exists():
            return

        print("Downloading InSpatio-World checkpoints (first run only)...")
        cwd = Path(INSPATIO_WEIGHTS)
        cwd.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}

        def git_lfs_clone(url: str, dest: str):
            dest_path = cwd / dest
            if dest_path.exists():
                shutil.rmtree(dest_path)
            subprocess.run(["git", "clone", url, dest], cwd=cwd, env=env, check=True)
            subprocess.run(["git", "lfs", "pull"], cwd=dest_path, check=True)

        git_lfs_clone("https://huggingface.co/inspatio/world", "world")
        (cwd / "InSpatio-World-1.3B").mkdir(exist_ok=True)
        shutil.move(
            str(cwd / "world" / "InSpatio-World-1.3B.safetensors"),
            str(cwd / "InSpatio-World-1.3B" / "InSpatio-World-1.3B.safetensors"),
        )
        shutil.rmtree(cwd / "world")

        git_lfs_clone(
            "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
        )
        git_lfs_clone(
            "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE",
            "DA3",
        )
        git_lfs_clone(
            "https://huggingface.co/microsoft/Florence-2-large",
            "Florence-2-large",
        )
        subprocess.run(
            ["git", "clone", "https://github.com/madebyollin/taehv.git"],
            cwd=cwd,
            check=True,
        )

        sentinel.write_text("ok")
        inspatio_model_volume.commit()
        print("InSpatio checkpoints cached.")

    def _link_checkpoints(self):
        """The repo expects weights under <repo>/checkpoints/,
        so symlink them in on every container start."""
        ckpt_dir = Path(INSPATIO_REPO) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weights = Path(INSPATIO_WEIGHTS)
        for name in (
            "Wan2.1-T2V-1.3B",
            "taehv",
            "InSpatio-World-1.3B",
            "DA3",
            "Florence-2-large",
        ):
            target = weights / name
            link = ckpt_dir / name
            if not target.exists() or link.is_symlink() or link.exists():
                continue
            link.symlink_to(target)

    @modal.method()
    def run(
        self,
        job_id: str,
        trajectory: str,
        skip_preprocess: bool = False,
        source_bytes: Optional[bytes] = None,
    ) -> str:
        self._link_checkpoints()

        input_dir = job_dir(job_id) / "inspatio" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        dest = input_dir / "source.mp4"

        if source_bytes is not None:
            dest.write_bytes(source_bytes)
        elif not skip_preprocess:
            source = job_dir(job_id) / "packaged" / "source.mp4"
            if not wait_for_file(source):
                raise FileNotFoundError(f"source video missing: {source}")
            shutil.copy2(source, dest)

        traj_path = Path(INSPATIO_REPO) / "traj" / f"{trajectory}.txt"
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")

        output_folder = job_dir(job_id) / "inspatio" / "output" / trajectory
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
        if skip_preprocess:
            cmd.extend(["--skip_step1", "--skip_step2"])

        print(f"InSpatio job {job_id}: trajectory={trajectory}, skip={skip_preprocess}")
        result = subprocess.run(
            cmd,
            cwd=INSPATIO_REPO,
        )
        if result.returncode != 0:
            raise RuntimeError(f"InSpatio pipeline failed with exit code {result.returncode}")

        output_volume.commit()
        return str(output_folder)


# `package_job` picks the relevant videos and point cloud out of the InSpatio
# output, transcodes the videos for the browser, and records their URLs in the
# manifest. It runs inside the (CPU) orchestrator container that invokes it.

# The orchestrator image needs ffmpeg for transcoding; fastapi is present
# because importing this module pulls in the top-level `import fastapi`.

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("fastapi[standard]==0.115.8")
)


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


def _clear_scratch(job_id: str) -> None:
    """Delete intermediate stage dirs once their artifacts are in ``packaged/``.
    """
    base = job_dir(job_id)
    shutil.rmtree(base / "ltx", ignore_errors=True)
    shutil.rmtree(base / "inspatio" / "output", ignore_errors=True)


def package_job(job_id: str, trajectory: str) -> dict:
    """Package the job's artifacts for the web UI."""
    output_dir = job_dir(job_id) / "inspatio" / "output" / trajectory
    wait_for_file(output_dir)

    packaged = job_dir(job_id) / "packaged"
    packaged.mkdir(parents=True, exist_ok=True)

    packaged_source = packaged / "source.mp4"
    ltx_source = job_dir(job_id) / "ltx" / "source.mp4"
    if ltx_source.exists() and not packaged_source.exists():
        shutil.copy2(ltx_source, packaged_source)

    world_src, cond_src = _pick_videos(output_dir) if output_dir.exists() else (None, None)

    assets: dict[str, Optional[str]] = {
        "source_video": f"/api/assets/{job_id}/source.mp4" if packaged_source.exists() else None,
        "world_video": None,
        "conditioning_video": None,
        "pointcloud_ply": None,
    }

    # Encode InSpatio outputs to H.264 + faststart.
    if world_src:
        transcode_to_web_mp4(world_src, packaged / "world.mp4")
        assets["world_video"] = f"/api/assets/{job_id}/world.mp4"
    if cond_src:
        transcode_to_web_mp4(cond_src, packaged / "conditioning.mp4")
        assets["conditioning_video"] = f"/api/assets/{job_id}/conditioning.mp4"

    ply_src = _pick_pointcloud(output_dir) if output_dir.exists() else None
    if ply_src:
        shutil.copy2(ply_src, packaged / "pointcloud.ply")
        assets["pointcloud_ply"] = f"/api/assets/{job_id}/pointcloud.ply"

    manifest = read_manifest(job_id)
    cached = manifest.get("cached_trajectories", {})
    if trajectory not in cached and assets.get("world_video"):
        cached[trajectory] = assets.copy()

    updated = write_manifest(
        job_id,
        status="done",
        trajectory=trajectory,
        assets=assets,
        cached_trajectories=cached,
        error=None,
    )

    _clear_scratch(job_id)
    output_volume.commit()
    return updated


# ## Orchestration

# `run_world_job` drives a full request through all three stages; the UI spawns
# it in the background and polls the manifest. `run_trajectory_job` re-renders
# an existing world along a new trajectory, skipping the LTX and preprocessing
# stages.


@app.function(
    image=orchestrator_image,
    volumes={ARTIFACTS_PATH: output_volume},
    timeout=90 * MINUTES,
)
def run_world_job(job_id: str, params: dict):
    try:
        write_manifest(
            job_id,
            status="running_ltx",
            **{k: params.get(k) for k in ("prompt", "seed", "trajectory", "width", "height", "num_frames", "fps")},
        )
        LTXInference().run.remote(
            job_id=job_id,
            prompt=params["prompt"],
            image_bytes=params.get("image_bytes"),
            seed=params.get("seed"),
            width=params.get("width", DEFAULT_WIDTH),
            height=params.get("height", DEFAULT_HEIGHT),
            num_frames=params.get("num_frames", DEFAULT_NUM_FRAMES),
            fps=params.get("fps", DEFAULT_FPS),
        )

        trajectory = params.get("trajectory", "x_y_circle_cycle")
        src_ltx = job_dir(job_id) / "ltx" / "source.mp4"
        if not wait_for_file(src_ltx):
            raise FileNotFoundError(f"LTX output missing after commit: {src_ltx}")

        packaged_dir = job_dir(job_id) / "packaged"
        packaged_dir.mkdir(parents=True, exist_ok=True)
        source_bytes = src_ltx.read_bytes()
        (packaged_dir / "source.mp4").write_bytes(source_bytes)
        write_manifest(
            job_id,
            status="running_inspatio",
            trajectory=trajectory,
            assets={"source_video": f"/api/assets/{job_id}/source.mp4"},
        )

        InSpatioInference().run.remote(
            job_id=job_id, trajectory=trajectory, source_bytes=source_bytes
        )

        write_manifest(job_id, status="packaging")
        return package_job(job_id, trajectory)
    except Exception as e:
        write_manifest(job_id, status="error", error=str(e))
        raise


@app.function(
    image=orchestrator_image,
    volumes={ARTIFACTS_PATH: output_volume},
    timeout=45 * MINUTES,
)
def run_trajectory_job(job_id: str, trajectory: str):
    try:
        write_manifest(
            job_id,
            status="running_inspatio",
            trajectory=trajectory,
        )
        InSpatioInference().run.remote(
            job_id=job_id,
            trajectory=trajectory,
            skip_preprocess=True,
        )
        write_manifest(job_id, status="packaging")
        return package_job(job_id, trajectory)
    except Exception as e:
        write_manifest(job_id, status="error", error=str(e))
        raise


# ## Web UI

# A small ASGI app serves the frontend and exposes the job lifecycle: submit a
# job, poll its status, re-render a trajectory, and serve packaged assets.

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

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_name": "World Model (LTX-2.3 + InSpatio)",
                "default_prompt": (
                    "A young girl stands calmly in the foreground, looking directly "
                    "at the camera, as a house fire rages in the background."
                ),
                "trajectories": TRAJECTORY_PRESETS,
                "default_width": DEFAULT_WIDTH,
                "default_height": DEFAULT_HEIGHT,
                "default_num_frames": DEFAULT_NUM_FRAMES,
                "default_fps": DEFAULT_FPS,
            },
        )

    @web_app.post("/api/jobs")
    async def submit_job(
        prompt: str = fastapi.Form(...),
        trajectory: str = fastapi.Form("x_y_circle_cycle"),
        seed: Optional[int] = fastapi.Form(None),
        width: int = fastapi.Form(DEFAULT_WIDTH),
        height: int = fastapi.Form(DEFAULT_HEIGHT),
        num_frames: int = fastapi.Form(DEFAULT_NUM_FRAMES),
        fps: int = fastapi.Form(DEFAULT_FPS),
        image: Annotated[Optional[fastapi.UploadFile], fastapi.File()] = None,
    ):
        job_id = uuid.uuid4().hex[:12]
        image_bytes = await image.read() if image and image.filename else None
        params = {
            "prompt": prompt,
            "trajectory": trajectory,
            "seed": seed,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "image_bytes": image_bytes,
        }
        await write_manifest_aio(
            job_id,
            status="queued",
            prompt=prompt,
            trajectory=trajectory,
            seed=seed,
            cached_trajectories={},
        )
        call = await run_world_job.spawn.aio(job_id, params)
        await write_manifest_aio(job_id, call_id=call.object_id)
        return {"job_id": job_id, "call_id": call.object_id}

    @web_app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        await output_volume.reload.aio()
        manifest = read_manifest(job_id)
        if not manifest:
            return fastapi.responses.JSONResponse(
                {"error": "job not found"},
                status_code=404,
            )

        call_id = manifest.get("call_id")
        if call_id and manifest.get("status") not in ("done", "error"):
            try:
                fc = modal.FunctionCall.from_id(call_id)
                await fc.get.aio(timeout=0)
            except TimeoutError:
                pass
            except modal.exception.OutputExpiredError:
                manifest = read_manifest(job_id)
            except Exception as e:
                await write_manifest_aio(
                    job_id, status="error", error=str(e)
                )
                manifest = read_manifest(job_id)

        await output_volume.reload.aio()
        manifest = read_manifest(job_id)
        code = 200 if manifest.get("status") in ("done", "error") else 202
        return fastapi.responses.JSONResponse(manifest, status_code=code)

    @web_app.post("/api/jobs/{job_id}/trajectory")
    async def rerun_trajectory(job_id: str, trajectory: str = fastapi.Form(...)):
        await output_volume.reload.aio()
        manifest = read_manifest(job_id)
        if not manifest:
            return fastapi.responses.JSONResponse(
                {"error": "job not found"},
                status_code=404,
            )

        cached = manifest.get("cached_trajectories", {})
        if trajectory in cached:
            assets = cached[trajectory]
            await write_manifest_aio(
                job_id,
                status="done",
                trajectory=trajectory,
                assets=assets,
            )
            return read_manifest(job_id)

        call = await run_trajectory_job.spawn.aio(job_id, trajectory)
        await write_manifest_aio(
            job_id, call_id=call.object_id, trajectory=trajectory
        )
        return {"job_id": job_id, "call_id": call.object_id, "status": "running_inspatio"}

    @web_app.get("/api/assets/{job_id}/{filename}")
    async def serve_asset(job_id: str, filename: str):
        if filename not in ASSET_NAMES:
            return fastapi.responses.JSONResponse(
                {"error": "unknown asset"},
                status_code=404,
            )
        await output_volume.reload.aio()
        path = job_dir(job_id) / "packaged" / filename
        if not path.exists():
            return fastapi.responses.JSONResponse(
                {"error": "asset not ready"},
                status_code=404,
            )
        media = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
        return fastapi.responses.FileResponse(
            path, media_type=media, content_disposition_type="inline"
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


# ## CLI

# `modal run image_to_world.py --prompt "..."` runs one job end-to-end and
# downloads the packaged artifacts to the local output directory.


@app.local_entrypoint()
def entrypoint(
    prompt: str,
    image_path: Optional[str] = None,
    trajectory: str = "x_y_circle_cycle",
    seed: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
):
    import os
    import urllib.request

    job_id = uuid.uuid4().hex[:12]
    image_bytes = None
    if image_path:
        if image_path.startswith(("http://", "https://")):
            image_bytes = urllib.request.urlopen(image_path).read()
        elif os.path.isfile(image_path):
            image_bytes = Path(image_path).read_bytes()
        else:
            raise ValueError(f"{image_path} is not a valid file or URL.")

    params = {
        "prompt": prompt,
        "trajectory": trajectory,
        "seed": seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "image_bytes": image_bytes,
    }

    print(f"Starting world job {job_id}")
    print(f"  prompt: {prompt}")
    print(f"  trajectory: {trajectory}")

    start = time.time()
    manifest = run_world_job.remote(job_id, params)
    duration = time.time() - start
    print(f"Job finished in {duration:.1f}s — status={manifest.get('status')}")

    if manifest.get("status") == "error":
        raise RuntimeError(manifest.get("error", "unknown error"))

    out_dir = Path("/tmp/world_model") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("source.mp4", "world.mp4", "conditioning.mp4", "pointcloud.ply"):
        vol_path = f"{job_id}/packaged/{name}"
        try:
            data = b"".join(output_volume.read_file(vol_path))
        except Exception:
            continue
        local = out_dir / name
        local.write_bytes(data)
        print(f"  saved {local}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
