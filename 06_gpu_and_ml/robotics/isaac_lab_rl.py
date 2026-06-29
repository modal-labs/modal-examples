"""
Isaac Lab robot training example on Modal.

In this example, we'll use Modal to quickly and easily train a robot using 4 L40S GPUs. 
Specifically, we'll run a headless instance of Isaac Lab and train a policy that teaches Anymal-D,
a quadruped robot, to obey a velocity command and walk over rough terrain.

Isaac Lab is NVIDIA's open source python framework for robot learning with GPUs. It's built on top of Isaac Sim, 
NVIDIA's open source robotics simulation framework. Isaac Sim is built on top of Omniverse (rendering), 
and PhysX (physics engine), which both take advantage of GPUs to accelerate simulation.

Isaac Lab integrates with a variety of RL frameworks. Today, we'll use rsl_rl, a light open source
reinforcement learning library, with PPO as the training algorithm. All of these details are transparent to our use, 
as Isaac Lab ships a pre-made `task` for training a quadruped to follow a velocity command.
@see https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity

NOTE: this example requires the `runc` container runtime on Modal. Reach out to support@modal.com to request access.

MODAL_FUNCTION_RUNTIME=runc modal run isaac_lab_rl.py
Output mp4 is written to the `isaac-demo-output` Volume (and downloadable).
"""

import modal
import os
import subprocess

# NVIDIA's official container bundles Isaac Lab and all the necessary dependencies for this example.
ISAAC_LAB_IMAGE = "nvcr.io/nvidia/isaac-lab:2.1.0"  

image = (
    modal.Image.from_registry(ISAAC_LAB_IMAGE, add_python="3.11")
    # IMPORTANT: the isaac-lab image's ENTRYPOINT runs `/isaac-sim/runheadless.sh`,
    # and never execs the arguments passed to it, which is a requirement for Modal.
    # @see https://modal.com/docs/guide/existing-images#entrypoint 
    .entrypoint([])
    .apt_install("ffmpeg")  # usually present; ensures clip stitching works
    .env({
        "ACCEPT_EULA": "Y",
        # "PRIVACY_CONSENT": "Y",
        # "OMNI_KIT_ACCEPT_EULA": "YES",
    })
    .add_local_file(
        "play_demo.py",
        "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play_demo.py",
        copy=True,
    )   
)

app = modal.App("isaac-sim-headless-demo")

# Persisted shader cache at /root/.cache/ov, should reduce cold start times after first run.
ov_cache = modal.Volume.from_name("isaac-ov-cache", create_if_missing=True)
# Outputs (rendered mp4s) so you can grab them after the run.
output_vol = modal.Volume.from_name("isaac-demo-output", create_if_missing=True)

TASK = "Isaac-Velocity-Rough-Anymal-D-v0"
VIDEO_LENGTH = 300  # sim steps recorded
NUM_GPUS = 4
NUM_ENVS_PER_GPU = 4096
ITERATIONS = 150
# the robot is trained to move with a "commanded velocity",
# which we'll keep consistent when recording the progress videos at each checkpoint.
PLAY_COMMAND_VELOCITY = (1.0, 0.0, 0.0)  

# Training here is PPO via rsl_rl: thousands (`NUM_ENVS_PER_GPU` * `NUM_GPUS`) of
# robots run in parallel, each gets a random commanded velocity, and a
# reward (track velocity, stay upright, don't waste energy) shapes the
# policy. Rough terrain adds a curriculum that ramps up difficulty. 
# We train once, then render multiple checkpoints from that same run 
# so the video shows one policy improving over time:
#   Phase 1 (beginning): early checkpoint from the run -> stumbles / falls
#   Phase 2 (middle):    midpoint checkpoint from the run -> improving
#   Phase 3 (end):       final checkpoint from the run -> competent
#   Phase 4 (expert):    NVIDIA's pretrained checkpoint -> robust
@app.function(
    image=image,
    gpu=f"L40S:{NUM_GPUS}",
    volumes={
        "/root/.cache/ov": ov_cache,  
        "/output": output_vol,
    },
    timeout=60 * 60, 
)
def train_and_render_demo():

    os.makedirs("/output", exist_ok=True)

    # first we train the policy using the rsl_rl training script baked into the image, implemented here:
    # https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/rsl_rl/train.py
    # this is a thin script that is mainly respoinsible for instantiating a gymnasium environment based
    # on the provided `task` and mediating the data exchange between rsl_rl's training runner and the gymnasium environment.
    run_name = "progress_run"
    subprocess.run([
        "/workspace/isaaclab/isaaclab.sh", "-p", "-m",
        "torch.distributed.run",
        "--standalone", 
        "--nproc_per_node=4",
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "--task", TASK, "--headless",
        "--num_envs", str(NUM_ENVS_PER_GPU),
        "--max_iterations", str(ITERATIONS),
        "--run_name", run_name,
        "--distributed"
    ], check=True, cwd="/workspace/isaaclab")
    print("Training completed")

    # collect paths to policy checkpoints at each stage
    beginning_ckpt, beginning_iter = _ckpt_at_or_before(run_name, 50)
    middle_ckpt, middle_iter = _ckpt_at_or_before(run_name, 100)
    end_ckpt, end_iter = _ckpt_at_or_before(run_name, ITERATIONS)

    # render the checkpoints into short videos
    beginning_clip = _render("beginning", checkpoint=beginning_ckpt)
    print(f"Phase 1 clip -> {beginning_clip}")
    middle_clip = _render("middle", checkpoint=middle_ckpt)
    print(f"Phase 2 clip -> {middle_clip}")
    end_clip = _render("end", checkpoint=end_ckpt)
    print(f"Phase 3 clip -> {end_clip}")
    expert_clip = _render("expert", pretrained=True)
    print(f"Phase 4 clip -> {expert_clip}")

    # stitch each training phase video into a single video
    out = f"/output/anymal_d_rough_four_phases.mp4"
    _concat_with_labels([
        (beginning_clip, f"Phase 1 - beginning training ({beginning_iter} iters) - fails", "red"),
        (middle_clip,    f"Phase 2 - middle training ({middle_iter} iters) - improving", "yellow"),
        (end_clip,       f"Phase 3 - end training ({end_iter} iters) - competent", "cyan"),
        (expert_clip,    "Phase 4 - fully trained (NVIDIA) - succeeds", "lime"),
    ], out)

    # persist to our modal volume
    output_vol.commit()
    ov_cache.commit()  
    print(f"Wrote {out}")
    return out

def _render(clip_name, checkpoint=None, pretrained=False):
    video_folder = "/output/rendered_clips"
    video_prefix = clip_name
    video_path = os.path.join(video_folder, f"{video_prefix}-step-0.mp4")
    os.makedirs(video_folder, exist_ok=True)
    if os.path.exists(video_path):
        os.remove(video_path)

    # similar to the train script there is a play script included in the image that can 
    # be used to render a checkpoint of the policy. We have copied and slightly modified
    # it to better suite this headless example.
    cmd = [
        "/workspace/isaaclab/isaaclab.sh", "-p",
        "scripts/reinforcement_learning/rsl_rl/play_demo.py",
        "--task", TASK, "--headless", "--enable_cameras",
        "--num_envs", "1",
        "--video", "--video_length", str(VIDEO_LENGTH),
        "--video_folder", video_folder,
        "--video_name_prefix", video_prefix,
        "--command_velocity", *(str(v) for v in PLAY_COMMAND_VELOCITY),
    ]
    cmd += ["--use_pretrained_checkpoint"] if pretrained else ["--checkpoint", checkpoint]
    subprocess.run(cmd, check=True, cwd="/workspace/isaaclab")
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise FileNotFoundError(f"Expected rendered video at {video_path}, but it was not produced.")
    return video_path

def _ckpt_at_or_before(run_substr: str, iteration: int):
    ckpts = _ckpts_for_run(run_substr)
    candidates = [item for item in ckpts if item[1] <= iteration]
    return max(candidates or ckpts, key=lambda item: item[1] if item[1] <= iteration else -item[1])


def _ckpts_for_run(run_substr: str):
    import glob
    import re

    ckpts = glob.glob(f"/workspace/isaaclab/logs/rsl_rl/*/*{run_substr}/model_*.pt")
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt for run matching '{run_substr}'.")

    parsed = []
    for path in ckpts:
        match = re.search(r"model_(\d+)\.pt$", path)
        if match:
            parsed.append((path, int(match.group(1))))
    if not parsed:
        raise FileNotFoundError(f"No numbered model_<n>.pt for run matching '{run_substr}'.")
    return parsed


def _concat_with_labels(clips, out: str):
    import subprocess
    inputs = []
    for clip, _, _ in clips:
        inputs += ["-i", clip]

    parts = []
    for i, (_, label, color) in enumerate(clips):
        # drawtext uses ':' as an option separator, so escape any in the text.
        text = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
        parts.append(
            f"[{i}:v]drawtext=text='{text}':x=20:y=20:fontsize=30:fontcolor={color}"
            f":box=1:boxcolor=black@0.5:boxborderw=10[v{i}]"
        )
    streams = "".join(f"[v{i}]" for i in range(len(clips)))
    filtergraph = ";".join(parts) + f";{streams}concat=n={len(clips)}:v=1[outv]"

    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filtergraph, "-map", "[outv]", out],
        check=True,
    )


@app.local_entrypoint()
def main():
    out_path = train_and_render_demo.remote()
    print(f"Done. Video at {out_path} in the 'isaac-demo-output' volume.")
    print("Download with:  modal volume get isaac-demo-output "
          f"{out_path.split('/output/')[-1]} ./")
