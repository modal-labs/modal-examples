"""

Train a robot in a simulated environment with Isaac Lab and Modal.

Pared-down version of isaac_lab_rl.py: trains for 80 iterations on a single
GPU and renders one final clip.

modal run isaac_lab_rl.py
Output mp4 is written to the `isaac-demo-output` Volume (and is downloadable).
"""

import os
import subprocess
from dataclasses import dataclass

import modal

# NVIDIA's official container bundles Isaac Lab and all the necessary dependencies for this example.
image = modal.Image.from_registry(
    "nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1", add_python="3.11"
)
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
    num_gpus: int = 1
    num_envs_per_gpu: int = 4096
    iterations: int = 80
    # the robot is trained to move with a "commanded velocity", which we'll keep
    # consistent at demo time when recording the progress videos at each checkpoint.
    play_command_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # keep the sampled terrain patch/challenge consistent across checkpoint demos.
    play_seed: int = 3


config = Config()


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
    import time

    start_time = time.time()
    os.makedirs("/output", exist_ok=True)

    run_name = "progress_run"
    subprocess.run(
        [
            "/workspace/isaaclab/isaaclab.sh",
            "-p",
            "scripts/reinforcement_learning/rl_games/train.py",
            "--task",
            config.train_task,
            "--headless",
            "--num_envs",
            str(config.num_envs_per_gpu),
            "--max_iterations",
            str(config.iterations),
            "--kit_args",
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true",
            f"agent.params.config.full_experiment_name={run_name}",
            "agent.params.config.save_frequency=25",
        ],
        check=True,
        cwd="/workspace/isaaclab",
    )
    print("Training completed")

    end_ckpt, end_iter = _ckpt_at_or_before(run_name, config.iterations)
    clip = _render("final", checkpoint=end_ckpt)
    print(f"Final checkpoint ({end_iter} iters) -> {clip}")

    import shutil

    shutil.copy(clip, "/output/anymal_c_rough_final.mp4")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


@app.local_entrypoint()
def main():
    train_and_render_demo.remote()
    print(
        "Done. Video at /output/anymal_c_rough_final.mp4 in the 'isaac-demo-output' volume."
    )
    print("Download with:  modal volume get isaac-demo-output/anymal_c_rough_final.mp4")


# Demo rendering utility
def _render(clip_name, checkpoint):
    video_folder = "/output/rendered_clips"
    video_path = os.path.join(video_folder, f"{clip_name}-step-0.mp4")
    os.makedirs(video_folder, exist_ok=True)

    x_vel, y_vel, yaw_vel = config.play_command_velocity
    cmd = [
        "/workspace/isaaclab/isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/rl_games/play.py",
        "--task",
        config.play_task,
        "--headless",
        "--enable_cameras",
        "--device",
        "cuda:0",
        "--num_envs",
        "1",
        "--video",
        "--video_length",
        str(config.video_length),
        "--seed",
        str(config.play_seed),
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

    stock_video_path = os.path.join(
        os.path.dirname(os.path.dirname(checkpoint)),
        "videos",
        "play",
        "rl-video-step-0.mp4",
    )
    with open(stock_video_path, "rb") as src, open(video_path, "wb") as dst:
        dst.write(src.read())
    return video_path


def _ckpt_at_or_before(run_substr: str, iteration: int):
    ckpts = _ckpts_for_run(run_substr)
    candidates = [item for item in ckpts if item[1] <= iteration]
    return max(
        candidates or ckpts,
        key=lambda item: item[1] if item[1] <= iteration else -item[1],
    )


def _ckpts_for_run(run_substr: str):
    import glob
    import re

    ckpts = glob.glob(f"/workspace/isaaclab/logs/rl_games/*/*{run_substr}/nn/*.pth")
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
