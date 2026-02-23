# ---
# cmd: ["python", "13_sandboxes/anthropic_computer_use.py"]
# pytest: false
# ---

# # Run Anthropic's computer use demo in a Modal Sandbox

# This example demonstrates how to run Anthropic's [Computer Use demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
# in a Modal [Sandbox](https://modal.com/docs/guide/sandbox).

# ## Sandbox Setup

# All Sandboxes are associated with an App.

# We start by looking up an existing App by name, or creating one if it doesn't exist.

import time
import urllib.request

import modal
import modal.experimental

app = modal.App.lookup("example-anthropic-computer-use", create_if_missing=True)

# The Computer Use [quickstart](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
# provides a prebuilt Docker image. We use this hosted image to create our sandbox environment.

sandbox_image = (
    modal.experimental.raw_registry_image(
        "ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest",
    )
    .env({"WIDTH": "1920", "HEIGHT": "1080"})
    .workdir("/home/computeruse")
    .entrypoint([])
)

# We'll provide the Anthropic API key via a Modal [Secret](https://modal.com/docs/guide/secrets)
# which the sandbox can access at runtime.

secret = modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"])

# Now, we can start our Sandbox.
# We use `modal.enable_output()` to print the Sandbox's image build logs to the console.
# We'll also expose the ports required for the demo's interfaces:

# - Port 8501 serves the Streamlit UI for interacting with the agent loop
# - Port 6080 serves the VNC desktop view via a browser-based noVNC client

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "sudo",
        "--preserve-env=ANTHROPIC_API_KEY,DISPLAY_NUM,WIDTH,HEIGHT,PATH",
        "-u",
        "computeruse",
        "./entrypoint.sh",
        app=app,
        image=sandbox_image,
        secrets=[secret],
        encrypted_ports=[8501, 6080],
        timeout=60 * 60,  # stay alive for one hour, maximum one day
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")

# After starting the sandbox, we retrieve the public URLs for the exposed ports.

tunnels = sandbox.tunnels()
for port, tunnel in tunnels.items():
    print(f"Waiting for service on port {port} to start at {tunnel.url}")

# We can check on each server's status by making an HTTP request to the server's URL
# and verifying that it responds with a 200 status code.


def is_server_up(url):
    try:
        response = urllib.request.urlopen(url)
        return response.getcode() == 200
    except Exception:
        return False


timeout = 60  # seconds
start_time = time.time()
up_ports = set()
while time.time() - start_time < timeout:
    for port, tunnel in tunnels.items():
        if port not in up_ports and is_server_up(tunnel.url):
            print(f"🏖️  Server is up and running on port {port}!")
            up_ports.add(port)
    if len(up_ports) == len(tunnels):
        break
    time.sleep(1)
else:
    print("🏖️  Timed out waiting for server to start.")


# You can now open the URLs in your browser to interact with the demo!
# Note: The sandbox logs may mention `localhost:8080`.
# Ignore this and use the printed tunnel URLs instead.

# When finished, you can terminate the sandbox from your [Modal dashboard](https://modal.com/containers)
# or by running `Sandbox.from_id(sandbox.object_id).terminate()`.
# The Sandbox will also spin down after one hour.
