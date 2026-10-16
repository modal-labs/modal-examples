# ---
# args: ["--iterations", "50"]
# ---

# # Train a robot in a simulated environment with Isaac Lab
#
# Reinforcement learning teaches a robot by trial and error. A policy (a neural network) maps what the robot
# observes to how it moves its joints, and rewards for good behavior improve the policy over many attempts in simulation.
#
# In this example, we'll run a headless instance of Isaac Lab to train a policy that teaches
# [Anymal-D](https://www.anybotics.com/robotics/anymal), a quadruped robot, to obey a velocity command and walk over
# rough terrain. We train on a multi-GPU [Modal Function](https://modal.com/docs/guide/gpu) and save two videos to a
# [Volume](https://modal.com/docs/guide/volumes): the policy at iteration 91 beside NVIDIA's pretrained policy, and
# the policy progression accross three checkpoints.
#
# <video controls autoplay loop muted playsinline>
# <source src="https://modal-cdn.com/cdnbot/anymal_d_final_vs_pretrainedhr0phbwj_01e82ae5.mp4" type="video/mp4">
# </video>
#
# <video controls autoplay loop muted playsinline>
# <source src="https://modal-cdn.com/cdnbot/anymal_d_training_progressk9ghp9og_d677ca5a.mp4" type="video/mp4">
# </video>
#
# [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) is NVIDIA's open source Python framework for robot learning with GPUs.
# It's built on top of [Isaac Sim](https://developer.nvidia.com/isaac/sim), NVIDIA's open source robotics simulation platform.
# Isaac Sim utilizes [Omniverse](https://developer.nvidia.com/omniverse) (simulation and rendering) and
# [PhysX](https://developer.nvidia.com/physx-sdk) (physics engine), which both take advantage of GPUs for acceleration.
#
# Isaac Lab integrates with a variety of RL frameworks. Here, we'll use [rsl-rl](https://github.com/leggedrobotics/rsl_rl),
# an open source reinforcement learning library for robotics training, with PPO as the training algorithm. All of
# these details are transparent to our use, as Isaac Lab ships a pre-made
# [task](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/velocity)
# which we'll take advantage of.
#
# Run this example with:
#
# ```
# modal run isaac_lab.py
# ```

# ## Set up

import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

# We train on four GPUs, with 4096 simulated robots per GPU, for 120 iterations.

NUM_GPUS = 4
MINUTES = 60
TASK = "Isaac-Velocity-Rough-AnymalD"
TRAINING_ITERATIONS = 121
RENDER_ITERATIONS = (60, 70, 91)

# Isaac Lab and its Python environment live at these paths in NVIDIA's image.

ISAACLAB_DIR = Path("/workspace/isaaclab")
ISAAC_PYTHON = ISAACLAB_DIR / "_isaac_sim/python.sh"

# Isaac Lab uses Omniverse Kit to manage the simulation and rendering.

KIT_ARGS = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error --/omni.kit.plugin/usdMuteDiagnosticMessage=true"

# We'll replay our policy at several checkpoints and render the output into videos.
# The first Kit launch in the container spends minutes while the NVIDIA driver
# compiles RTX shaders into this cache. Subsequent launches to render videos will reuse it.

DRIVER_SHADER_CACHE_DIR = Path("/isaac-sim/kit/cache/nv_shadercache")

# Both stitched videos are saved to a Volume so you can grab them after the run.

OUTPUT_DIR = "/output"
COMPARISON_VIDEO_NAME = "anymal_d_final_vs_pretrained.mp4"
PROGRESS_VIDEO_NAME = "anymal_d_training_progress.mp4"
output_vol = modal.Volume.from_name("isaac-demo-output", create_if_missing=True)

# ## Install dependencies
#
# NVIDIA's official container bundles Isaac Lab and all the necessary dependencies for this example.
# We add moviepy, which Isaac Lab uses to record videos, and ffmpeg, which tiles the clips together.

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:3.0.0-rc1", add_python="3.11")
    .entrypoint([])
    .apt_install("ffmpeg")
    .env({"ACCEPT_EULA": "Y"})
    .run_commands(f"{ISAAC_PYTHON} -m pip install moviepy==1.0.3")
)

ffmpeg_image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")

app = modal.App("example-isaac-lab")


# ## Train the policy
#
# During training, each iteration spawns thousands of robots in parallel.
# Each gets a random commanded velocity and a reward (track velocity,
# keep back horizontal, etc.) that shapes the policy. Curriculum learning
# is enforced by rough terrain that ramps up in difficulty over time.


@app.function(image=image, gpu=f"L40S:{NUM_GPUS}", timeout=60 * MINUTES)
def train(iterations: int = TRAINING_ITERATIONS) -> dict[int, tuple[int, bytes]]:
    subprocess.run(
        [
            str(ISAAC_PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(NUM_GPUS),
            "scripts/reinforcement_learning/train.py",
            "--rl_library",
            "rsl_rl",
            "--task",
            TASK,
            "--num_envs",
            "4096",
            "--max_iterations",
            str(iterations),
            "--distributed",
            "--run_name",
            "training_run",
            "--kit_args",
            KIT_ARGS,
            "agent.save_interval=10",
        ],
        check=True,
        cwd=ISAACLAB_DIR,
        env={**os.environ, "NCCL_CUMEM_HOST_ENABLE": "0"},
    )
    print("Training completed.")

    checkpoints = {}
    for iteration in RENDER_ITERATIONS:
        actual_iteration, path = _checkpoint_at_or_before(iteration)
        checkpoints[iteration] = (actual_iteration, path.read_bytes())
    return checkpoints


# ## Render each checkpoint
#
# Each checkpoint is replayed in one container. The first launch compiles RTX shaders into a cache on
# that container's disk, and later renders copy the cache instead of compiling again. Up to two renders run
# at once. Each of those copies the cache into its own directory first, because the NVIDIA driver gives
# every process its own cache file, and a second process sharing the directory would recompile.


@app.cls(
    image=image,
    gpu="L40S",
    timeout=60 * MINUTES,
    max_containers=1,
    scaledown_window=10 * MINUTES,
)
@modal.concurrent(max_inputs=2)
class Renderer:
    @modal.method()
    def render(self, clip_name: str, checkpoint: bytes | None = None) -> bytes:
        checkpoint_path = None
        if checkpoint is None:
            DRIVER_SHADER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = DRIVER_SHADER_CACHE_DIR
        else:
            cache_path = Path(f"/tmp/nv_shadercache_{clip_name}")
            if DRIVER_SHADER_CACHE_DIR.is_dir():
                shutil.copytree(DRIVER_SHADER_CACHE_DIR, cache_path, dirs_exist_ok=True)
            run_dir = Path("/tmp/play_runs") / clip_name
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / f"{clip_name}.pt"
            checkpoint_path.write_bytes(checkpoint)

        kit_args = KIT_ARGS + f" --/rtx/shaderDb/driverShaderCachePath={cache_path}"
        return _render(checkpoint_path, kit_args)


# ## Tile the clips


@app.function(image=ffmpeg_image, volumes={OUTPUT_DIR: output_vol})
def tile_clips(
    comparison: list[tuple[bytes, str, str]],
    progress: list[tuple[bytes, str, str]],
):
    _write_video(comparison, COMPARISON_VIDEO_NAME)
    _write_video(progress, PROGRESS_VIDEO_NAME)


def _write_video(clips: list[tuple[bytes, str, str]], video_name: str):
    inputs, filters = [], []
    for i, (video, label, color) in enumerate(clips):
        path = Path(f"/tmp/{video_name}_{i}.mp4")
        path.write_bytes(video)
        inputs += ["-i", str(path)]
        text = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
        filters.append(
            f"[{i}:v]drawtext=text='{text}':x=20:y=20:fontsize=30:fontcolor={color}"
            f":box=1:boxcolor=black@0.5:boxborderw=10[v{i}]"
        )
    streams = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(
        f"{streams}hstack=inputs={len(clips)}:shortest=1,format=yuv420p[outv]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            f"{OUTPUT_DIR}/{video_name}",
        ],
        check=True,
    )


# ## Put it all together
#
# First, we kick off rendering the pretrained clip. It doesn't depend on training, and the first render
# compiles the RTX shaders so the later clips reuse them. In parallel, we start training the policy.
# Then we render the checkpoints in the same warm container.
# Finally, we stitch two videos from those renders.


@app.local_entrypoint()
def main(iterations: int = TRAINING_ITERATIONS):
    renderer = Renderer()
    expert_call = renderer.render.spawn("expert")

    checkpoints = train.remote(iterations)
    expert_video = expert_call.get()

    rendered = list(
        renderer.render.starmap(
            [
                (str(iteration), checkpoints[iteration][1])
                for iteration in RENDER_ITERATIONS
            ]
        )
    )
    videos = dict(zip(RENDER_ITERATIONS, rendered))
    final = RENDER_ITERATIONS[-1]

    tile_clips.remote(
        [
            (videos[final], f"{checkpoints[final][0]} iters", "cyan"),
            (expert_video, "pretrained", "lime"),
        ],
        [
            (videos[60], f"{checkpoints[60][0]} iters", "yellow"),
            (videos[70], f"{checkpoints[70][0]} iters", "orange"),
            (videos[final], f"{checkpoints[final][0]} iters", "cyan"),
        ],
    )
    print("Done. Download the videos with:")
    print(f"  modal volume get isaac-demo-output {COMPARISON_VIDEO_NAME}")
    print(f"  modal volume get isaac-demo-output {PROGRESS_VIDEO_NAME}")


# ## Demo rendering and video stitching utilities


def _checkpoint_at_or_before(iteration: int) -> tuple[int, Path]:
    by_iteration = {}
    for path in ISAACLAB_DIR.glob("logs/rsl_rl/*/*training_run/model_*.pt"):
        match = re.search(r"model_(\d+)\.pt$", path.name)
        if match is None:
            continue
        by_iteration[int(match.group(1))] = path
    eligible = [i for i in by_iteration if i <= iteration]
    if not eligible:
        saved = ", ".join(str(i) for i in sorted(by_iteration)) or "none"
        raise FileNotFoundError(
            f"No checkpoint at or before iteration {iteration}. Saved checkpoints: {saved}."
        )
    actual = max(eligible)
    return actual, by_iteration[actual]


def _render(checkpoint_path: Path | None, kit_args: str) -> bytes:
    # Similar to train.py, Isaac Lab ships a play.py script for rendering this policy.
    terrain = "env.scene.terrain.terrain_generator"
    velocity = "env.commands.base_velocity"
    subprocess.run(
        [
            str(ISAAC_PYTHON),
            "scripts/reinforcement_learning/play.py",
            "--rl_library",
            "rsl_rl",
            "--task",
            TASK,
            "--device",
            "cuda:0",
            "--num_envs",
            "1",
            "--video",
            "--video_length",
            "200",
            "--seed",
            "3",
            "--kit_args",
            kit_args,
            "--checkpoint",
            str(checkpoint_path) if checkpoint_path else "pretrained",
            "env.viewer.eye=[5.2,5.2,2.7]",
            "env.viewer.lookat=[0.0,0.0,0.55]",
            f"{terrain}.num_rows=1",
            f"{terrain}.num_cols=1",
            f"{terrain}.sub_terrains.pyramid_stairs_inv.proportion=0.0",
            f"{terrain}.sub_terrains.boxes.proportion=0.0",
            f"{terrain}.sub_terrains.random_rough.proportion=0.0",
            f"{terrain}.sub_terrains.hf_pyramid_slope.proportion=0.0",
            f"{terrain}.sub_terrains.hf_pyramid_slope_inv.proportion=0.0",
            f"{velocity}.debug_vis=false",
            f"{velocity}.heading_command=false",
            f"{velocity}.ranges.lin_vel_x=[1.0,1.0]",
            f"{velocity}.ranges.lin_vel_y=[0.0,0.0]",
            f"{velocity}.ranges.ang_vel_z=[0.0,0.0]",
        ],
        check=True,
        cwd=ISAACLAB_DIR,
    )

    # play.py records to <checkpoint dir>/videos/play/<name>.mp4
    if checkpoint_path:
        videos = list(checkpoint_path.parent.glob("videos/play/*.mp4"))
        searched = checkpoint_path.parent / "videos/play"
    else:
        searched = ISAACLAB_DIR / ".pretrained_checkpoints" / "rsl_rl"
        videos = list(searched.glob("**/videos/play/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No play video was generated under {searched}.")
    latest = max(videos, key=lambda path: path.stat().st_mtime)
    return latest.read_bytes()
