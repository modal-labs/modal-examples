# ---
# cmd: ["python", "13_sandboxes/computer_use.py"]
# pytest: false
# ---

# # Run Anthropic's Computer Use demo in a Modal Sandbox

# This example demonstrates how to run Anthropic's Computer Use demo
# in a Modal [Sandbox](https://modal.com/docs/guide/sandbox).

# ## Sandbox Setup

# All Sandboxes are associated with an App.

# We start by looking up an existing App by name, or creating one if it doesn't exist.

import modal
import modal.experimental

app = modal.App.lookup("example-computer-use", create_if_missing=True)

# The Computer Use [quickstart](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
# provides a prebuilt Docker image. We use this hosted image to create our sandbox environment.

sandbox_image = (
    modal.experimental.raw_registry_image(
        "ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest",
    )
    .env({"WIDTH": "1920", "HEIGHT": "1080"})
    .workdir("/home/computeruse")
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
        app=app,
        image=sandbox_image,
        secrets=[secret],
        encrypted_ports=[8501, 6080],
        timeout=60 * 60,
    )

print(f"Sandbox ID: {sandbox.object_id}")

# After starting the sandbox, we retrieve the public URLs for the exposed ports.

tunnels = sandbox.tunnels()
for port, tunnel in tunnels.items():
    print(f"Port {port}: {tunnel.url}")

# You can now open the URLs in your browser to access the demo!

# When you're done, terminate the sandbox using your [Modal dashboard](https://modal.com/sandboxes)
# or by running `Sandbox.from_id(sandbox.object_id).terminate()`.
