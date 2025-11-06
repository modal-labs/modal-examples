# # Deploy Hunyuan GameCraft Gradio demo with Modal
#
# This script demonstrates how to deploy the Hunyuan GameCraft interactive web UI on Modal.
#
# ## Overview
# First, download the model weights to the Modal Volume. This can take up to an hour.
# ```bash
# modal run misc/hunyuan_gamecraft.py::download_model
# ```
#
# - Deploy the Gradio web app:
#   ```bash
#   modal deploy misc/hunyuan_gamecraft.py
#   ```
#
# - Generate a video locally via the CLI:
#   ```bash
#   modal run misc/hunyuan_gamecraft.py
#   ```
#
# ## Description
#
# The Hunyuan GameCraft world model lets you generate consistent, first-person style videos by providing:
# - a starting image
# - a text prompt
# - a sequence of actions (e.g., move forward/backward, turn left/right)
#
# The app produces a video simulating movement within the generated world!

import modal

########## CONSTANTS ##########
APP_NAME = "hunyuan-gamecraft-gradio"
VOLUME_NAME = "example_hunyuan_gamecraft_demo"
MOUNT_PATH = "/mnt/example-hunyuan-gamecraft-demo"

########## IMAGE DEFINITION ##########
# Using Hunyuan GameCraft container using their provided Dockerfile
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

web_image = (
    modal.Image.from_registry("hunyuanvideo/hunyuanvideo:cuda_12")
    .apt_install("ffmpeg", "libsm6", "libxext6", "git")
    .uv_pip_install(
        "huggingface_hub==0.36.0",
        "diffusers==0.34.0",
        "transformers==4.54.1",
        "hf_transfer==0.1.9",
        "gradio==5.1.0",
        "fastapi[standard]==0.112.4",
        "opencv-python==4.9.0.80",
        "loguru==0.7.2",
        "pillow==10.4.0",
        "numpy==1.24.4",
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0.git /workspace/Hunyuan-GameCraft-1.0",
    )
    .entrypoint([])
)

########## VOLUME SETUP ##########


# Download model weights to Modal Volume for fast access
@app.function(
    image=web_image,
    volumes={MOUNT_PATH: volume},
    timeout=60 * 60,  # one hour
)
def download_model(model_id="tencent/Hunyuan-GameCraft-1.0", model_dir=MOUNT_PATH):
    """Retrieve model from HuggingFace Hub and save into
    specified path within the modal container.

    Args:
        model_id (str): HuggingFace Model ID.
        model_dir (str): Path to save model weights in container.
    """
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(repo_id=model_id, local_dir=model_dir)


########## GRADIO APP ##########
# Taking care to wrap the Gradio app according to best practices
# https://www.gradio.app/guides/deploying-gradio-with-modal

NUM_GPUS = 1


@app.function(
    image=web_image,
    volumes={MOUNT_PATH: volume},
    gpu=f"H100:{NUM_GPUS}",
    timeout=3600,
    max_containers=1,  # Sticky sessions for Gradio
)
@modal.concurrent(max_inputs=100)  # Allow multiple users
@modal.asgi_app()
def gradio_app():
    """Serve Gradio demo for Hunyuan GameCraft on Modal"""

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    print("🚀 Starting Hunyuan GameCraft Gradio on Modal...")

    # Create Gradio UI
    with gr.Blocks(title="Hunyuan GameCraft") as demo:
        gr.Markdown("# 🎮 Hunyuan GameCraft Video Generator")
        gr.Markdown("Generate game-style videos using a world model on Modal")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                prompt = gr.Textbox(
                    label="Prompt",
                    value="A majestic ancient temple with intricate details",
                    lines=2,
                )

                with gr.Row():
                    actions = gr.Textbox(
                        label="Actions (e.g., 'w w s')",
                        value="w",
                        info="w=forward, s=backward, a=left, d=right",
                    )
                    speeds = gr.Textbox(
                        label="Speeds (e.g., '0.2 0.3')",
                        value="0.1",
                        info="One speed per action",
                    )

                generate_btn = gr.Button("🚀 Generate", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                status = gr.Textbox(label="Status")

        generate_btn.click(
            fn=generate_video.local,
            inputs=[input_image, prompt, actions, speeds],
            outputs=[output_video, status],
        )

        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                [
                    "/workspace/Hunyuan-GameCraft-1.0/asset/temple.png",
                    "A majestic ancient temple with towering columns",
                    "w",
                    "0.1",
                ]
            ],
            inputs=[input_image, prompt, actions, speeds],
        )

    # Mount on FastAPI
    fastapi_app = FastAPI()

    return mount_gradio_app(app=fastapi_app, blocks=demo, path="/")


########## GENERATE VIDEO FUNCTION ##########
@app.function(
    image=web_image,
    volumes={MOUNT_PATH: volume},
    gpu=f"H100:{NUM_GPUS}",
    timeout=60 * 60,
)
def generate_video(image, prompt, actions, speeds):
    """Generate video using Hunyuan GameCraft"""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    if image is None:
        print("No image provided")
        return None, "Please upload an image"

    # Set up paths
    sys.path.insert(0, "/workspace/Hunyuan-GameCraft-1.0")
    os.chdir("/workspace/Hunyuan-GameCraft-1.0")

    # Set up path variables
    os.environ["MODEL_BASE"] = f"{MOUNT_PATH}/stdmodels"
    checkpoint_path = (
        f"{MOUNT_PATH}/gamecraft_models/mp_rank_00_model_states_distill.pt"
    )

    # Save uploaded image temporarily to the Image
    timestamp = int(time.time())
    img_filename = f"temp_input_{timestamp}.png"
    img_path = f"{MOUNT_PATH}/{img_filename}"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if isinstance(image, bytes):
        Path(img_path).write_bytes(image)
    else:  # PIL Image
        image.save(img_path)

    # Parse actions and speeds
    action_list = actions.split() if isinstance(actions, str) else actions
    speed_list = (
        [float(s) for s in speeds.split()] if isinstance(speeds, str) else speeds
    )

    # Create output directory
    output_dir = f"{MOUNT_PATH}/results"
    os.makedirs(output_dir, exist_ok=True)

    current_time = time.strftime("%Y.%m.%d-%H.%M.%S")
    outprefix = f"{output_dir}/{current_time}_gamecraft"

    action_str = " ".join(action_list)
    speed_str = " ".join(map(str, speed_list))

    cmd = f"""
    export PYTHONPATH=/workspace/Hunyuan-GameCraft-1.0:$PYTHONPATH && \
    torchrun --nnodes=1 --nproc_per_node={NUM_GPUS} --master_port 29605 hymm_sp/sample_batch.py \
        --image-path "{img_path}" \
        --prompt "{prompt}" \
        --add-pos-prompt "Realistic, High-quality." \
        --add-neg-prompt "overexposed, low quality, deformation, bad composition" \
        --ckpt {checkpoint_path} \
        --video-size 1024 576 \
        --cfg-scale 1 \
        --image-start \
        --action-list {action_str} \
        --action-speed-list {speed_str} \
        --seed 250160 \
        --infer-steps 2 \
        --flow-shift-eval-video 5.0 \
        --sample-n-frames 2 \
        --save-path "{outprefix}"
    """

    subprocess.run(cmd, shell=True, check=True)

    # Find generated video in the Volume. It's named after the input image!
    # e.g. temp_input_1698412345.mp4
    expected_video_name = f"{img_filename.split('.')[0]}.mp4"
    video_path = f"{outprefix}/{expected_video_name}"

    if os.path.exists(video_path):
        print(f"Found video: {video_path}")
        return video_path, "Success!"

    # If not found, list all mp4s for debugging
    mp4_files = [f for f in os.listdir(outprefix) if f.endswith(".mp4")]
    return None, f"Video not found. Expected: {expected_video_name}, Found: {mp4_files}"


########## LOCAL ENTRYPOINT ##########
# Run the Hunyuan Gamecraft model via CLI: modal run hunyuan_gamecraft_gradio.py
# You can specify the image, prompt, actions, and speeds, or leave as defaults.
@app.local_entrypoint()
def main(
    image_path: str | None = None,
    prompt: str = "A majestic ancient temple with intricate details",
    actions: str = "w",
    speeds: str = "0.1",
):
    import sys
    import urllib.request
    from pathlib import Path

    # Load the initial image file
    if image_path is None:
        image_url = "https://modal-cdn.com/example-hunyuan-gamecraft-temple.png"
        print(f"generating video from image at URL {image_url}")
        request = urllib.request.Request(image_url)
        with urllib.request.urlopen(request) as response:
            image_bytes = response.read()
    else:
        image_bytes = image_path.read_bytes()

    # Generate video using the remote Modal function
    mp4_name, status = generate_video.remote(image_bytes, prompt, actions, speeds)

    mp4_name = "/mnt/example-hunyuan-gamecraft-demo/results/2025.10.30-09.27.22_gamecraft/temp_input_1761787642.mp4"

    # Check if video generated successfully
    if mp4_name is None:
        print(f"Error: {status}")
        sys.exit(1)
    print("Video generated successfully")
    mp4_name = Path(mp4_name)

    # Retrieve video from Modal Volume and save to local disk
    output_dir = Path("/tmp") / "hunyuan_gamecraft_output"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / mp4_name.name

    mp4_path = mp4_name.relative_to(MOUNT_PATH)
    output_path.write_bytes(b"".join(volume.read_file(str(mp4_path))))
    print(f"🎥 Video saved to {output_path}")
