"""
---
env: {"MODAL_FUNCTION_RUNTIME": "runc"}
---

Train a robot in a simulated environment with Isaac Lab and Modal.

In this example, we'll use Modal to quickly and easily train a robot in a simluated environment using 4 L40S GPUs. 
Specifically, we'll run a headless instance of Isaac Lab to train a policy that teaches Anymal-C,
a quadruped robot, to obey a velocity command and walk over rough terrain.

Isaac Lab is NVIDIA's open source python framework for robot learning with GPUs. It's built on top of Isaac Sim, 
NVIDIA's open source robotics simulation platform. Isaac Sim utilizes Omniverse (simulation and rendering) 
and PhysX (physics engine), which both take advantage of GPUs for acceleration.

Isaac Lab integrates with a variety of RL frameworks. Today, we'll use rl-games, an open source
reinforcement learning library for robotics training, with PPO as the training algorithm. All of
these details are transparent to our use, as Isaac Lab ships a pre-made `task` for training a quadruped
to follow a velocity command.
see: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity

NOTE: this example relies on features that are currently in alpha. Please reach out to support@modal.com to request access.

modal run isaac_lab_rl.py
Output mp4 is written to the `isaac-demo-output` Volume (and is downloadable).
"""

import modal
import os
import subprocess

from dataclasses import dataclass

# NVIDIA's official container bundles Isaac Lab and all the necessary dependencies for this example.
image = modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1", add_python="3.11")
# IMPORTANT: the isaac-lab image's ENTRYPOINT runs `/isaac-sim/runheadless.sh`, and never
# execs the arguments passed to it, which is a requirement for Modal, so we must override it.
# see https://modal.com/docs/guide/existing-images#entrypoint 
image = image.entrypoint([])
# Install ffmpeg for image stitching, and accept the EULA is a requirement.
image = image.apt_install("ffmpeg").env({"ACCEPT_EULA": "Y", "HYDRA_FULL_ERROR": "1"})
# Install Isaac Lab's optional rl-games workflow dependencies.
image = image.run_commands("/workspace/isaaclab/isaaclab.sh -i rl_games")

app = modal.App("isaac-sim-headless-demo")

# Persisted shader cache at /root/.cache/ov, should reduce cold start times after first run.
ov_cache = modal.Volume.from_name("isaac-ov-cache", create_if_missing=True)
# Outputs (rendered mp4s) so you can grab them after the run.
output_vol = modal.Volume.from_name("isaac-demo-output", create_if_missing=True)

@dataclass
class Config:
    train_task: str = "Isaac-Velocity-Rough-Anymal-C-v0"
    play_task: str = "Isaac-Velocity-Rough-Anymal-C-Play-v0"
    video_length: int = 200
    num_gpus: int = 4
    num_envs_per_gpu: int = 4096
    iterations: int = 100
    # the robot is trained to move with a "commanded velocity", which we'll keep
    # consistent at demo time when recording the progress videos at each checkpoint.
    play_command_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # keep the sampled terrain patch/challenge consistent across checkpoint demos.
    play_seed: int = 3

config = Config()

# Training here is PPO via rl-games: thousands (`NUM_ENVS_PER_GPU` * `NUM_GPUS`) of
# robots run in parallel at each iteration, each gets a random commanded velocity, 
# and a reward (track velocity, keep back horizontal, etc) shapes the
# policy. Rough terrain adds a curriculum that ramps up difficulty. 
# We train once, then render multiple checkpoints from that same run 
# so the video shows one policy improving over time:
#   Phase 1 (50 iterations):    midpoint checkpoint from the run -> stumbling
#   Phase 2 (100 iterations):       final checkpoint from the run -> competent
#   Phase 3 (pretrained checkpoint):    NVIDIA's pretrained checkpoint -> robust
@app.function(
    image=image,
    gpu=f"L40S:{config.num_gpus}",
    volumes={
        "/root/.cache/ov": ov_cache,  
        "/output": output_vol,
    },
    timeout=60 * 60, 
)
def train_and_render_demo():

    os.makedirs("/output", exist_ok=True)

    # first we train the policy using the rl-games training script baked into the image, implemented here:
    # https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/rl_games/train.py
    # this is a thin script that is mainly responsible for instantiating a gymnasium environment based
    # on the provided `task` and mediating the data exchange between rl-games' training runner and the environment.
    run_name = "progress_run"
    subprocess.run([
        "/workspace/isaaclab/isaaclab.sh", "-p", "-m",
        "torch.distributed.run",
        "--standalone", 
        "--nproc_per_node", str(config.num_gpus),
        "scripts/reinforcement_learning/rl_games/train.py",
        "--task", config.train_task, "--headless",
        "--num_envs", str(config.num_envs_per_gpu),
        "--max_iterations", str(config.iterations),
        "--distributed",
        "--kit_args", "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true",
        f"agent.params.config.full_experiment_name={run_name}",
        "agent.params.config.save_frequency=25",
    ], check=True, cwd="/workspace/isaaclab")
    print("Training completed")

    middle_ckpt, middle_iter = _ckpt_at_or_before(run_name, 50)
    end_ckpt, end_iter = _ckpt_at_or_before(run_name, config.iterations)

    phases = [
        ("middle", f"Phase 1 - learning ({middle_iter} iters)", "yellow", middle_ckpt, False),
        ("end", f"Phase 2 - trained ({end_iter} iters)", "cyan", end_ckpt, False),
        ("expert", "Phase 3 - pretrained NVIDIA", "lime", None, True),
    ]

    # Play back each checkpoint and record the video.
    clips = []
    for clip_name, label, color, checkpoint, pretrained in phases:
        clip = _render(clip_name, checkpoint=checkpoint, pretrained=pretrained)
        print(f"{label} -> {clip}")
        clips.append((clip, label, color))

    # Consolidate each video into a single grid video.
    _tile_with_labels(clips, "/output/anymal_c_rough_progress_grid.mp4")

@app.local_entrypoint()
def main():
    train_and_render_demo.remote()
    print("Done. Video at /output/anymal_c_rough_progress_grid.mp4 in the 'isaac-demo-output' volume.")
    print("Download with:  modal volume get isaac-demo-output/anymal_c_rough_progress_grid.mp4")

# Demo rendering and video stitching utilities
def _render(clip_name, checkpoint=None, pretrained=False):
    video_folder = "/output/rendered_clips"
    video_path = os.path.join(video_folder, f"{clip_name}-step-0.mp4")
    os.makedirs(video_folder, exist_ok=True)

    # Similar to train.py, Isaac Lab ships a play.py script for rendering a policy.
    x_vel, y_vel, yaw_vel = config.play_command_velocity
    cmd = [
        "/workspace/isaaclab/isaaclab.sh", "-p",
        "scripts/reinforcement_learning/rl_games/play.py",
        "--task", config.play_task, "--headless", "--enable_cameras",
        "--device", "cuda:0",
        "--num_envs", "1",
        "--video", "--video_length", str(config.video_length),
        "--seed", str(config.play_seed),
        "--kit_args", "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true",
        *(["--use_pretrained_checkpoint"] if pretrained else ["--checkpoint", checkpoint]),
        "env.viewer.origin_type=world",
        "env.viewer.eye=[5.2,5.2,2.7]",
        "env.viewer.lookat=[0.0,0.0,0.55]",
        "env.scene.terrain.terrain_generator.num_rows=1",
        "env.scene.terrain.terrain_generator.num_cols=1",
        "env.scene.terrain.max_init_terrain_level=0",
        "env.scene.terrain.terrain_generator.sub_terrains.pyramid_stairs.proportion=1.0",
        "env.scene.terrain.terrain_generator.sub_terrains.pyramid_stairs_inv.proportion=0.0",
        "env.scene.terrain.terrain_generator.sub_terrains.boxes.proportion=0.0",
        "env.scene.terrain.terrain_generator.sub_terrains.random_rough.proportion=0.0",
        "env.scene.terrain.terrain_generator.sub_terrains.hf_pyramid_slope.proportion=0.0",
        "env.scene.terrain.terrain_generator.sub_terrains.hf_pyramid_slope_inv.proportion=0.0",
        "env.commands.base_velocity.debug_vis=false",
        f"env.commands.base_velocity.ranges.lin_vel_x=[{x_vel},{x_vel}]",
        f"env.commands.base_velocity.ranges.lin_vel_y=[{y_vel},{y_vel}]",
        f"env.commands.base_velocity.ranges.ang_vel_z=[{yaw_vel},{yaw_vel}]",
        "env.commands.base_velocity.heading_command=false",
        "env.commands.base_velocity.ranges.heading=[0.0,0.0]",
    ]
    subprocess.run(cmd, check=True, cwd="/workspace/isaaclab")

    stock_video_path = _latest_pretrained_play_video() if pretrained else os.path.join(os.path.dirname(os.path.dirname(checkpoint)), "videos", "play", "rl-video-step-0.mp4")

    with open(stock_video_path, "rb") as src, open(video_path, "wb") as dst:
        dst.write(src.read())
    return video_path


def _latest_pretrained_play_video() -> str:
    import glob

    videos = glob.glob("/workspace/isaaclab/.pretrained_checkpoints/rl_games/**/videos/play/rl-video-step-0.mp4", recursive=True)
    if not videos:
        raise FileNotFoundError("No pretrained rl-games play video was generated.")
    return max(videos, key=os.path.getmtime)

def _ckpt_at_or_before(run_substr: str, iteration: int):
    ckpts = _ckpts_for_run(run_substr)
    candidates = [item for item in ckpts if item[1] <= iteration]
    return max(candidates or ckpts, key=lambda item: item[1] if item[1] <= iteration else -item[1])

def _ckpts_for_run(run_substr: str):
    import glob
    import re

    ckpts = glob.glob(f"/workspace/isaaclab/logs/rl_games/*/*{run_substr}/nn/*.pth")
    print("TEST ckpts: ", ckpts)
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints for run matching '{run_substr}'.")

    parsed = []
    for path in ckpts:
        filename = os.path.basename(path)
        if filename.endswith(".pth"):
            match = re.search(r"_ep_(\d+)", filename)
            itr = 0
            if match:
                itr = int(match.group(1))
            parsed.append((path, itr))
    if not parsed:
        raise FileNotFoundError(f"No .pth checkpoints for run matching '{run_substr}'.")
    return parsed

def _tile_with_labels(clips, out: str):
    import subprocess
    inputs = []
    for clip, _, _ in clips:
        inputs += ["-i", clip]

    parts = []
    for i, (_, label, color) in enumerate(clips):
        text = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
        parts.append(
            f"[{i}:v]drawtext=text='{text}':x=20:y=20:fontsize=30:fontcolor={color}"
            f":box=1:boxcolor=black@0.5:boxborderw=10[v{i}]"
        )
    streams = "".join(f"[v{i}]" for i in range(len(clips)))
    filtergraph = ";".join(parts) + f";{streams}hstack=inputs={len(clips)}:shortest=1,format=yuv420p[outv]"

    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filtergraph, "-map", "[outv]", out],
        check=True,
    )
