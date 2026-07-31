# Train a robot in a simulated environment with Isaac Lab and Modal.
#
# In this example, we'll use Modal to train a robot model in a simluated environment using an L40S GPU.
# Specifically, we'll run a headless instance of Isaac Lab to train a policy that teaches Anymal-C,
# a quadruped robot, to obey a velocity command and walk over rough terrain.
#
# Isaac Lab is NVIDIA's open source python framework for robot learning with GPUs. It's built on top of Isaac Sim,
# NVIDIA's open source robotics simulation platform. Isaac Sim utilizes Omniverse (simulation and rendering)
# and PhysX (physics engine), which both take advantage of GPUs for acceleration.
#
# Isaac Lab integrates with a variety of RL frameworks. Today, we'll use rl-games, an open source
# reinforcement learning library for robotics training, with PPO as the training algorithm. All of
# these details are transparent to our use, as Isaac Lab ships a pre-made `task` for training a quadruped
# to follow a velocity command.
# see: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity
#
# modal run isaac_lab_rl.py
# Output mp4 is written to the `isaac-demo-output` Volume (and is downloadable).

import shutil
import subprocess
from pathlib import Path

import modal

# NVIDIA's official container bundles Isaac Lab and all the necessary dependencies for this example.
image = modal.Image.from_registry(
    "nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1", add_python="3.11"
)
# IMPORTANT: the isaac-lab image's ENTRYPOINT runs `/isaac-sim/runheadless.sh`, and never
# execs the arguments passed to it, which is a requirement for Modal Images, so we must override it.
# see https://modal.com/docs/guide/existing-images#entrypoint
image = (
    image.entrypoint([])
    .env({"ACCEPT_EULA": "Y", "HYDRA_FULL_ERROR": "1"})
    .run_commands("/workspace/isaaclab/isaaclab.sh -i rl_games")
)

app = modal.App("example-isaac-lab-rl")

# Persisted shader cache at /root/.cache/ov, should reduce cold start times after first run.
ov_cache = modal.Volume.from_name("isaac-ov-cache", create_if_missing=True)
# Outputs (rendered mp4s) so you can grab them after the run.
output_vol = modal.Volume.from_name("isaac-demo-output", create_if_missing=True)

OUTPUT_PATH = "/output"


# Training here is PPO via rl-games: thousands of robots run in parallel at each
# iteration, each gets a random commanded velocity,
# and a reward (track velocity, keep back horizontal, etc) shapes the
# policy. Rough terrain adds a curriculum that ramps up difficulty.
# We train once, then render the final checkpoint and record it as a video.
@app.function(
    image=image,
    gpu="L40S:1",
    volumes={
        "/root/.cache/ov": ov_cache,
        OUTPUT_PATH: output_vol,
    },
    timeout=60 * 60,
)
def train_and_render_demo(
    train_task: str = "Isaac-Velocity-Rough-Anymal-C-v0",
    play_task: str = "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    video_length: int = 200,
    num_envs: int = 4096,
    iterations: int = 125,
    play_seed: int = 3,
):
    import time

    start_time = time.time()

    # First we train the policy using the rl-games training script baked into the image, implemented here:
    # https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/scripts/reinforcement_learning/rl_games/train.py
    # this is a thin script that is mainly responsible for instantiating a Gymnasium environment based
    # on the provided `task` and mediating the data exchange between rl-games' training runner and the environment.
    # `--viz none` disables all visualizers, so the simulation runs headless.
    run_name = "training_run"
    subprocess.run(
        [
            "/workspace/isaaclab/isaaclab.sh",
            "-p",
            "scripts/reinforcement_learning/train.py",
            "--rl_library",
            "rl_games",
            "--task",
            train_task,
            "--viz",
            "none",
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(iterations),
            "--kit_args",
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true",
            f"agent.params.config.full_experiment_name={run_name}",
            "agent.params.config.save_frequency=25",
        ],
        check=True,
        cwd="/workspace/isaaclab",
    )
    print("Training completed")

    # Once we have trained the model, we grab the latest checkpoint and play a demo simulation at it.
    # We'll copy the video to the Volume-mounted output path so that the results are persisted
    # and can be downloaded later.

    checkpoint = _latest_checkpoint(run_name)

    clip = _render(
        checkpoint=checkpoint,
        play_task=play_task,
        video_length=video_length,
        play_seed=play_seed,
    )
    print(f"Download with:  `modal volume get isaac-demo-output {clip}`")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


@app.local_entrypoint()
def main(
    train_task: str = "Isaac-Velocity-Rough-Anymal-C-v0",
    play_task: str = "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    video_length: int = 200,
    num_envs: int = 4096,
    iterations: int = 80,
    play_seed: int = 3,
):
    train_and_render_demo.remote(
        train_task=train_task,
        play_task=play_task,
        video_length=video_length,
        num_envs=num_envs,
        iterations=iterations,
        play_seed=play_seed,
    )


# Demo rendering utility
def _render(
    checkpoint,
    play_task: str,
    video_length: int,
    play_seed: int,
):
    x_vel, y_vel, yaw_vel = 1.0, 0.0, 0.0
    cmd = [
        "/workspace/isaaclab/isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/play.py",
        "--rl_library",
        "rl_games",
        "--task",
        play_task,
        "--viz",
        "none",
        "--enable_cameras",
        "--device",
        "cuda:0",
        "--num_envs",
        "1",
        "--video",
        "--video_length",
        str(video_length),
        "--seed",
        str(play_seed),
        "--kit_args",
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true",
        "--checkpoint",
        checkpoint,
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

    recording_dir = Path(checkpoint).parent.parent / "videos" / "play"
    recordings = sorted(recording_dir.glob("*.mp4"))
    if not recordings:
        raise FileNotFoundError(f"No recorded video in {recording_dir}.")

    # Copy the video to the Volume-mounted output path
    video_name = f"{play_task}.mp4"
    shutil.copyfile(recordings[0], Path(OUTPUT_PATH) / video_name)
    return video_name


def _latest_checkpoint(run_name: str) -> str:
    import glob
    import re

    ckpts = glob.glob(f"/workspace/isaaclab/logs/rl_games/*/*{run_name}/nn/*.pth")
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints for run matching '{run_name}'.")

    def iteration(path: str) -> int:
        match = re.search(r"_ep_(\d+)", Path(path).name)
        return int(match.group(1)) if match else 0

    return max(ckpts, key=iteration)
