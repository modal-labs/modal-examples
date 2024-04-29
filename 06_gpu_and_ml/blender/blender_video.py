# ---
# output-directory: "/tmp/render"
# ---
# # Render a video with Blender on many GPUs or CPUs in parallel
#
# This example shows how you can render an animated 3D scene using
# [Blender](https://www.blender.org/)'s Python interface.
#
# You can run it on CPUs to scale out on one hundred of containers
# or run it on GPUs to get higher throughput per node.
# Even with this simple scene, GPUs render 2x faster than CPUs.
#
# The final render looks something like this:
#
# ![Spinning Modal logo](https://modal-public-assets.s3.amazonaws.com/modal-blender-render.gif)
#
# ## Defining a Modal app

import io
import math
from pathlib import Path

import modal

# Modal runs your Python functions for you in the cloud.
# You organize your code into apps, collections of functions that work together.

app = modal.App("examples-blender-logo")

# We need to define the environment each function runs in --  its container image.
# The block below defines a container image, starting from a basic Debian Linux image
# adding Blender's system-level dependencies
# and then installing the `bpy` package, which is Blender's Python API.

rendering_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("xorg", "libxkbcommon0")  # X11 (Unix GUI) dependencies
    .pip_install("bpy")  # Blender as a Python package
)

# ## Rendering a single frame
#
# We define a function that renders a single frame. We'll scale this function out on Modal later.
#
# Functions in Modal are defined along with their hardware and their dependencies.
# This function can be run with GPU acceleration or without it, and we'll use a global flag in the code to switch between the two.

WITH_GPU = True  # try changing this to False to run rendering massively in parallel on CPUs!

# We decorate the function with `@app.function` to define it as a Modal function.
# Note that in addition to defining the hardware requirements of the function,
# we also specify the container image that the function runs in (the one we defined above).

# The details of the rendering function aren't too important for this example,
# so we abstract them out into functions defined at the end of the file.
# We draw a simple version of the Modal logo:
# two neon green rectangular prisms facing different directions.
# We include a parameter to rotate the prisms around the vertical/Z axis,
# which we'll use to animate the logo.


@app.function(
    gpu="T4" if WITH_GPU else None,
    concurrency_limit=10
    if WITH_GPU
    else 100,  # default limits on Modal free tier
    image=rendering_image,
)
def render(angle: int = 0) -> bytes:
    """
    Renders Modal's logo, two neon green rectangular prisms.


    Args:
        angle: How much to rotate the two prisms around the vertical/Z axis, in degrees.

    Returns:
        The rendered frame as a PNG image.
    """
    import bpy

    # clear existing objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    # ctx: the current Blender state, which we mutate
    ctx = bpy.context

    # scene: the 3D environment we are rendering and its camera(s)
    scene = ctx.scene

    # configure rendering -- CPU or GPU, resolution, etc.
    # see function definition below for details
    configure_rendering(ctx, WITH_GPU)

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = "output.png"

    # set background to black
    black = (0, 0, 0, 1)
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = black

    # add the Modal logo: two neon green rectangular prisms
    iridescent_material = create_iridescent_material()

    add_prism(ctx, (-1, 0, 0), 45, angle, iridescent_material)
    add_prism(ctx, (3, 0, 0), -45, angle, iridescent_material)

    # set up the lighting and camera
    bpy.ops.object.light_add(type="POINT", location=(5, 5, 5))
    bpy.context.object.data.energy = 10
    bpy.ops.object.camera_add(location=(7, -7, 5))
    scene.camera = bpy.context.object
    ctx.object.rotation_euler = (1.1, 0, 0.785)

    # render
    bpy.ops.render.render(write_still=True)

    # return the bytes to the caller
    with open(scene.render.filepath, "rb") as image_file:
        image_bytes = image_file.read()

    return image_bytes


# ### Rendering with acceleration
#
# We can configure the rendering process to use GPU acceleration with NVIDIA CUDA.
# We select the [Cycles rendering engine](https://www.cycles-renderer.org/), which is compatible with CUDA,
# and then activate the GPU.


def configure_rendering(ctx, with_gpu: bool):
    # configure the rendering process
    ctx.scene.render.engine = "CYCLES"
    ctx.scene.render.resolution_x = 1920
    ctx.scene.render.resolution_y = 1080
    ctx.scene.render.resolution_percentage = 100

    # add GPU acceleration if available
    if with_gpu:
        ctx.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        ctx.scene.cycles.device = "GPU"

        # reload the devices to update the configuration
        ctx.preferences.addons["cycles"].preferences.get_devices()
        for device in ctx.preferences.addons["cycles"].preferences.devices:
            device.use = True

    else:
        ctx.scene.cycles.device = "CPU"

    # report rendering devices -- a nice snippet for debugging and ensuring the accelerators are being used
    for dev in ctx.preferences.addons["cycles"].preferences.devices:
        print(
            f"ID:{dev['id']} Name:{dev['name']} Type:{dev['type']} Use:{dev['use']}"
        )


# ## Combining frames into a GIF
#
# Rendering 3D images is fun, and GPUs can make it faster, but rendering 3D videos is better!
# We add another function to our app, running on a different, simpler container image
# and different hardware, to combine the frames into a GIF.

combination_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pillow==10.3.0"
)

# The video has a few parameters, which we set here.

FPS = 60
FRAME_DURATION_MS = 1000 // FPS
NUM_FRAMES = 360  # drop this for faster iteration while playing around

# The function to combine the frames into a GIF takes a sequence of byte sequences, one for each rendered frame,
# and converts them into a single sequence of bytes, the GIF.


@app.function(image=combination_image)
def combine(
    frames_bytes: list[bytes], frame_duration: int = FRAME_DURATION_MS
) -> bytes:
    print("🎞️ combining frames into a gif")
    from PIL import Image

    frames = [
        Image.open(io.BytesIO(frame_bytes)) for frame_bytes in frames_bytes
    ]

    gif_image = io.BytesIO()
    frames[0].save(
        gif_image,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )

    gif_image.seek(0)

    return gif_image.getvalue()


# ## Rendering in parallel in the cloud from the comfort of the command line
#
# With these two functions defined, we need only a few more lines to run our rendering at scale on Modal.
#
# First, we need a function that coordinates our functions to `render` frames and `combine` them.
# We decorate that function with `@app.local_entrypoint` so that we can run it with `modal run blender_video.py`.
#
# In that function, we use `render.map` to map the `render` function over a `range` of `angle`s,
# so that the logo will appear to spin in the final video.
#
# We collect the bytes from each frame into a `list` locally and then send it to `combine` with `.remote`.
#
# The bytes for the video come back to our local machine, and we write them to a file.
#
# The whole rendering process (for six seconds of 1080p 60 FPS video) takes between five and ten minutes on 10 T4 GPUs.


@app.local_entrypoint()
def main():
    output_directory = Path("/tmp") / "render"
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = output_directory / "output.gif"
    with open(filename, "wb") as out_file:
        out_file.write(
            combine.remote(list(render.map(range(0, 360, 360 // NUM_FRAMES))))
        )
    print(f"Image saved to {filename}")


# ## Addenda
#
# The remainder of the code in this example defines the details of the render.
# It's not particularly interesting, so we put it the end of the file.


def add_prism(ctx, location, initial_rotation, angle, material):
    """Add a prism at a given location, rotation, and angle, made of the provided material."""
    import bpy
    import mathutils

    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = ctx.object  # the newly created object

    # assign the material to the object
    obj.data.materials.append(material)

    obj.scale = (1, 1, 2)  # square base, 2x taller than wide
    # Modal logo is rotated 45 degrees
    obj.rotation_euler[1] = math.radians(initial_rotation)

    # apply initial transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # to "animate" the rendering, we rotate the prisms around the Z axis
    angle_radians = math.radians(angle)
    rotation_matrix = mathutils.Matrix.Rotation(angle_radians, 4, "Z")
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def create_iridescent_material():
    import bpy

    mat = bpy.data.materials.new(name="IridescentGreen")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    emission_node = nodes.new(type="ShaderNodeEmission")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")

    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    color_ramp.color_ramp.elements[1].color = (0, 1, 0, 1)
    layer_weight.inputs["Blend"].default_value = 0.4

    links.new(layer_weight.outputs["Fresnel"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission_node.inputs["Color"])

    emission_node.inputs["Strength"].default_value = 5.0
    emission_node.inputs["Color"].default_value = (0.0, 1.0, 0.0, 1)

    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return mat
