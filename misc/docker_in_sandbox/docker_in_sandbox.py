import os

import modal
from modal.stream_type import StreamType

# Use the 2025.06 Modal Image Builder which avoids the need to install Modal client
# dependencies into the container image.

os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"

app = modal.App.lookup("docker-test", create_if_missing=True)
image = modal.Image.from_registry(
    "ghcr.io/thomasjpfan/docker-in-sandbox:0.0.12"
)

with modal.enable_output():
    sb = modal.Sandbox.create(
        timeout=60 * 10,
        app=app,
        image=image,
        experimental_options={"enable_docker": True},
    )

# A simple Dockerfile that we'll build and run within Modal.
dockerfile = """
FROM ubuntu
RUN apt-get update
RUN apt-get install -y cowsay curl
RUN mkdir -p /usr/share/cowsay/cows/
RUN curl -o /usr/share/cowsay/cows/docker.cow https://raw.githubusercontent.com/docker/whalesay/master/docker.cow
ENTRYPOINT ["/usr/games/cowsay", "-f", "docker.cow"]
"""
with sb.open("/build/Dockerfile", "w") as f:
    f.write(dockerfile)

print("Building docker image")
p = sb.exec(
    "docker",
    "build",
    "--network=host",
    "-t",
    "whalesay",
    "/build",
    stdout=StreamType.STDOUT,
    stderr=StreamType.STDOUT,
)
p.wait()
if p.returncode != 0:
    raise Exception("Docker build failed")

# Get the Sandbox to run the built image and show this:
#
#  ________
# < Hello! >
#  --------
#     \
#      \
#       \
#                     ##         .
#               ## ## ##        ==
#            ## ## ## ## ##    ===
#        /"""""""""""""""""\___/ ===
#       {                       /  ===-
#        \______ O           __/
#          \    \         __/
#           \____\_______/

print("Running Docker image")
# Note we can't use -it here because we're not in a TTY.
p = sb.exec(
    "docker",
    "run",
    "--rm",
    "whalesay",
    "Hello!",
    stdout=StreamType.STDOUT,
    stderr=StreamType.STDOUT,
)
p.wait()
if p.returncode != 0:
    raise Exception("Docker run failed")
sb.terminate()
