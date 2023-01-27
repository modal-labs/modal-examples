# ---
# output-directory: "/tmp/render"
# ---
# # Render a video with Blender on GPUs
#
# This example shows how you can render an animated 3D scene using [Blender](https://www.blender.org/)'s Python interface.
# We use Modal's GPU workers for this.

# ## Basic setup

import os
import tempfile

import modal

# The S3 locations of the assets we want to render, and the frame ranges.

SCENE_FILENAME = "https://modal-public-assets.s3.amazonaws.com/living_room_cam.blend"
MATERIALS_FILENAME = "https://modal-public-assets.s3.amazonaws.com/living_room_final.mtl"

START_FRAME = 32
END_FRAME = 34

# ## Defining the image
#
# Blender requires a very custom image in order to run properly.
# In order to save you some time, we have precompiled the Python packages
# and stored them in a Dockerhub image.


dockerfile_commands = [
    "RUN export DEBIAN_FRONTEND=noninteractive && "
    "chown root:root /var /etc /usr /var/lib /var/log / && "  # needed for some weird systemd error
    '    echo "deb http://deb.debian.org/debian testing main contrib non-free" > /etc/apt/sources.list.d/testing.list && '
    "    apt update && "
    "    apt install -yq --no-install-recommends libcrypt1 && "
    "    apt install -yq --no-install-recommends"
    "        libgomp1 "
    "        xorg "
    "        openbox "
    "        xvfb "
    "        libxxf86vm1 "
    "        libxfixes3 "
    "        libgl1",
    "COPY --from=akshatb42/bpy:2.93-gpu"
    "     /usr/local/lib/python3.9/dist-packages/"
    "     /usr/local/lib/python3.9/site-packages/",
    "RUN apt install -yq curl",
    f"RUN curl -L -o scene.blend -C - '{SCENE_FILENAME}'",
    f"RUN curl -L -o scene.mtl -C - '{MATERIALS_FILENAME}'",
]
stub = modal.Stub(
    "example-blender-video",
    image=modal.Image.debian_slim(python_version="3.9").dockerfile_commands(dockerfile_commands),
)


# ## Setting things up in the containers
#
# We need various global configuration that we want to happen inside the containers (but not locally), such as
# enabling the GPU device.
# To do this, we use the `stub.is_inside()` conditional, which will evaluate to `False` when the script runs
# locally, but to `True` when imported in the cloud.

if stub.is_inside():
    import bpy

    # NOTE: Blender segfaults if you try to do this after the other imports.
    bpy.ops.wm.open_mainfile(filepath="/scene.blend")
    bpy.data.scenes["Scene"].camera = bpy.data.objects.get("Camera.001")

    bpy.data.scenes[0].render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU

    print(
        "Has active device:",
        bpy.context.preferences.addons["cycles"].preferences.has_active_device(),
    )

    bpy.data.scenes[0].render.tile_x = 64
    bpy.data.scenes[0].render.tile_y = 64
    bpy.data.scenes[0].cycles.samples = 200


# ## Use a GPU from a Modal function
#
# Now, let's define the function that renders each frame in parallel.
# Note the `gpu="any"` argument which tells Modal to use GPU workers.


@stub.function(gpu="any")
def render_frame(i):
    print(f"Using frame {i}")

    scn = bpy.context.scene
    scn.render.resolution_x = 400
    scn.render.resolution_y = 400
    scn.render.resolution_percentage = 100
    scn.frame_set(i)

    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        scn.render.filepath = tf.name
        # Render still frame
        bpy.ops.render.render(write_still=True)
        with open(tf.name, "rb") as image:
            img_bytes = bytearray(image.read())
            return i, img_bytes


# ## Entrypoint
#
# The code that gets run locally.
# Note that it doesn't require Blender present to run it.
# In order to render in parallel, we use the `.map` method on the `render_frame` function.
# This spins up as many workers as are needed—as
# many as one for each frame, doing everything in parallel.


OUTPUT_DIR = "/tmp/render"


@stub.local_entrypoint
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Render the frames in parallel using modal, and write them to disk.
    for idx, frame in render_frame.map(range(START_FRAME, END_FRAME + 1)):
        with open(os.path.join(OUTPUT_DIR, f"scene_{idx:03}.png"), "wb") as f:
            f.write(frame)

    # Stitch together frames into a gif.
    import glob

    from PIL import Image

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "scene*.png")))]
    img.save(
        fp=os.path.join(OUTPUT_DIR, "scene.gif"),
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=200,
        loop=0,
    )
