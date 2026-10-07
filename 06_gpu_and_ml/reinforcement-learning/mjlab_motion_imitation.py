# ---
# args: ["--train-iters", "1", "--num-envs", "1"]
# ---

# # Teach a humanoid to imitate motion with mjlab
#
# Motion imitation teaches a robot to reproduce a reference movement, such as a dance. Using the robot's
# observations, we can train a policy for joint movement. Through repeated trials in a physics simulation, the
# policy learns to match the reference movement.
#
# This example uses [mjlab](https://github.com/mujocolab/mjlab) to train a simulated
# [Unitree G1 humanoid](https://www.unitree.com/g1) on a dance from the
# [retargeted LAFAN1 dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), which adapts
# human motion to a robot's joints and proportions. We use a [GPU Function](https://modal.com/docs/guide/gpu) to train and
# record the policy, and store motions, checkpoints, and videos in a [Volume](https://modal.com/docs/guide/volumes).
#
# Watch the robot before and after training:
#
# <video controls autoplay loop muted playsinline>
# <source src="https://modal-cdn.com/cdnbot/modal-mjlab-rl-comparison-videohf63yslv_de25e542.mp4" type="video/mp4">
# </video>
#
# Train the policy and record a video comparing the first and final checkpoints:
#
# ```bash
# modal run mjlab_motion_imitation.py
# ```
#
# Checkpoints are saved in the `mjlab-demo` Volume. To replay a specific checkpoint:
#
# ```bash
# modal run --write-result replay.mp4 mjlab_motion_imitation.py::rollout \
#   --checkpoint runs/Mjlab-Tracking-Flat-Unitree-G1/20260915T120000.000000Z/model_499.pt
# ```

# ## Set up

from datetime import datetime, timezone
from pathlib import Path

import modal

# The task selects the simulated G1 robot and its motion-tracking rewards.

TASK = "Mjlab-Tracking-Flat-Unitree-G1"
SEED = 42

# We select one dance and pin the dataset revision so repeated runs use the same reference motion.

DATASET_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
DATASET_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
MOTION_NAME = "dance1_subject2"

vol = modal.Volume.from_name("mjlab-demo", create_if_missing=True)

VOL_PATH = "/data"
MOTION_FILE = f"{VOL_PATH}/motions/{DATASET_REVISION}/{MOTION_NAME}.npz"
RUNS_ROOT = f"{VOL_PATH}/runs"

# Each GPU runs `NUM_ENVS` parallel robot simulations. More iterations give the policy more tries to match
# the dance, at the cost of a longer run.

GPU_TYPE, N_GPUS = "B200", 2
NUM_ENVS = 8192
TRAIN_ITERS = 500
MINUTES = 60  # seconds

# In the comparison video, the untrained and trained policies are recorded for the same duration.

ROLLOUT_SECONDS = 15.0

# ## Install dependencies
#
# We use a [Modal Image](https://modal.com/docs/guide/images) based on NVIDIA's CUDA runtime image to install the necessary dependencies.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install(
        "git",  # RSL-RL's logger imports GitPython, which requires git.
        "libegl-dev",
    )
    .uv_pip_install(
        "mjlab==1.6.0",
        "torch==2.14.0",
        "torchrunx==0.4.0",
        "tensorboard==2.21.0",
        "warp-lang==1.17.0",
        "mujoco==3.11.0",
        "mujoco-warp==3.11.0",
        "rsl-rl-lib==5.4.2",
        "mediapy==1.2.7",
        "huggingface-hub==0.36.0",
        "wandb==0.30.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "WANDB_MODE": "disabled",
        }
    )
)

app = modal.App("example-mjlab-motion-imitation", image=image)

# ## Download and convert the motion
#
# Each frame of the reference movement lists the robot's base
# position, orientation, and joint angles. mjlab also needs body poses and velocities, so the converter computes these
# and saves the motion in an NPZ file.


@app.function(gpu="A10", volumes={VOL_PATH: vol}, timeout=60 * MINUTES)
def convert_motion():
    import shutil

    import huggingface_hub
    from mjlab.scripts import csv_to_npz

    out = Path(MOTION_FILE)
    if out.exists():
        print(f"[INFO] {out} already present")
        return

    csv = huggingface_hub.hf_hub_download(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        filename=f"g1/{MOTION_NAME}.csv",
        repo_type="dataset",
    )
    converted = Path("/tmp/motion.npz")
    converted.unlink(missing_ok=True)
    csv_to_npz.main(
        input_file=csv, output_name=MOTION_NAME, input_fps=30.0, output_fps=50.0
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(converted, out)
    vol.commit()
    print(f"[INFO] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ## Configure mjlab
#
# In mjlab, `env_cfg` defines the robot simulation, observations, actions, and rewards while `agent_cfg` defines the policy and value
# networks and the Proximal Policy Optimization (PPO) training settings. We load both from mjlab's task registry and override a few parameters for our use case.


def _load_cfgs(play: bool, num_envs: int):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(TASK, play=play)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.scene.num_envs = num_envs
    agent_cfg.logger = "tensorboard"
    agent_cfg.seed = SEED
    env_cfg.seed = SEED
    env_cfg.commands["motion"].motion_file = MOTION_FILE
    return env_cfg, agent_cfg


# ## Train the policy
#
# For each GPU, a worker is created and simulates `NUM_ENVS` robots in parallel.


@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    env={
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(N_GPUS)))
    },  # mjlab's run_train selects CPU when CUDA_VISIBLE_DEVICES is unset or empty.
    volumes={VOL_PATH: vol},
    timeout=60 * MINUTES,
)
def train(run: str, train_iters: int = TRAIN_ITERS, num_envs: int = NUM_ENVS) -> float:
    import time

    import torchrunx
    from mjlab.scripts.train import TrainConfig, run_train

    log_dir = Path(RUNS_ROOT) / run
    log_dir.mkdir(parents=True)
    started = time.monotonic()

    env_cfg, agent_cfg = _load_cfgs(play=False, num_envs=num_envs)
    agent_cfg.max_iterations = train_iters
    agent_cfg.save_interval = max(train_iters // 4, 1)
    cfg = TrainConfig(env=env_cfg, agent=agent_cfg)
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=N_GPUS,
        backend=None,
    ).run(run_train, TASK, cfg, log_dir)
    vol.commit()

    train_s = round(time.monotonic() - started, 1)
    print(f"[TIME] {run}: {train_iters} iters on {N_GPUS}x{GPU_TYPE} in {train_s}s")
    return train_s


# ## Record a saved policy
#
# We load a checkpoint from the Volume and create a playback simulation to see how
# the trained policy behaves.


@app.function(gpu="A10", volumes={VOL_PATH: vol}, timeout=60 * MINUTES)
def rollout(checkpoint: str) -> bytes:
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    configure_torch_backends()
    device = "cuda:0"

    ckpt = (Path(VOL_PATH) / checkpoint).resolve()
    if not ckpt.is_relative_to(Path(RUNS_ROOT).resolve()):
        raise ValueError(f"checkpoint must be inside {RUNS_ROOT}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    env_cfg, agent_cfg = _load_cfgs(play=True, num_envs=8)

    env_cfg.viewer.height, env_cfg.viewer.width = 480, 640
    env_cfg.viewer.max_extra_envs = 0
    env_cfg.viewer.distance = 3.5
    env_cfg.viewer.azimuth = 90.0

    video_dir = ckpt.parent / "videos"
    prefix = f"{ckpt.parent.name}-{ckpt.stem}"

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    steps = int(ROLLOUT_SECONDS / env.step_dt)
    env = VideoRecorder(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=prefix,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), None, device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    wrapped.reset()
    with torch.inference_mode():
        for _ in range(steps):
            wrapped.step(policy(wrapped.get_observations()))

    env.close()
    video = video_dir / f"{prefix}-step-0.mp4"
    vol.commit()
    print(f"[VIDEO] {video}")
    return video.read_bytes()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg"),
    timeout=5 * MINUTES,
)
def combine_videos(before: bytes, after: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        before_path = Path(tmp) / "before.mp4"
        after_path = Path(tmp) / "after.mp4"
        output = Path(tmp) / "comparison.mp4"
        before_path.write_bytes(before)
        after_path.write_bytes(after)

        command = ["ffmpeg", "-i", str(before_path), "-i", str(after_path)]
        command += ["-filter_complex", "hstack=inputs=2:shortest=1"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True)
        return output.read_bytes()


# ## Put it all together


@app.local_entrypoint()
def main(train_iters: int = TRAIN_ITERS, num_envs: int = NUM_ENVS):
    if train_iters < 1:
        raise ValueError("train_iters must be positive")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    run = f"{TASK}/{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}"

    print(f"run: {run}", flush=True)

    convert_motion.remote()
    train_s = train.remote(run=run, train_iters=train_iters, num_envs=num_envs)

    before = rollout.remote(checkpoint=f"{RUNS_ROOT}/{run}/model_0.pt")
    after = rollout.remote(checkpoint=f"{RUNS_ROOT}/{run}/model_{train_iters - 1}.pt")
    comparison = combine_videos.remote(before, after)
    output = Path(f"motion-comparison-{Path(run).name}.mp4").resolve()
    output.write_bytes(comparison)

    print(f"\nclip: {MOTION_NAME}")
    print(
        f"trained {train_iters} iters in {train_s / 60:.1f} min "
        f"({train_s / train_iters:.2f} s/iter, including setup)"
    )
    print(
        f"\nComparison saved to {output} (1 iteration left, {train_iters} iterations right)"
    )
    print(
        "Replay with specific checkpoint: modal run --write-result replay.mp4 "
        "mjlab_motion_imitation.py::rollout "
        f"--checkpoint {RUNS_ROOT}/{run}/model_{train_iters - 1}.pt"
    )
