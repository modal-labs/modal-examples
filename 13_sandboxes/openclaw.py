# ---
# cmd: ["python", "13_sandboxes/openclaw.py"]
# pytest: false
# ---

# # Run OpenClaw in a Modal Sandbox

# This example demonstrates how to run [OpenClaw](https://github.com/openclaw/openclaw),
# an open-source AI agent, in a Modal [Sandbox](https://modal.com/docs/guide/sandbox).

# OpenClaw is a local AI assistant that gives language models "hands" to control
# your computer. Running it in a Modal Sandbox provides a secure, isolated environment
# where the agent can operate without risk to your local machine.

# ## Set up the Sandbox

# All Sandboxes are associated with an App.
# We start by looking up an existing App by name, or creating one if it doesn't exist.

import time
import urllib.request

import modal

MINUTES = 60  # seconds

app = modal.App.lookup("example-openclaw", create_if_missing=True)

# OpenClaw requires Node.js 22+ and is installed via npm.
# We build a custom image with the necessary dependencies.

sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "curl",
        "ca-certificates",
        "gnupg",
    )
    .run_commands(
        # Install Node.js 22 from NodeSource
        "mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg",
        'echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list',
        "apt-get update",
        "apt-get install -y nodejs",
    )
    .run_commands(
        # Install OpenClaw globally
        "npm install -g openclaw@latest",
    )
)

# We provide the API key for the LLM provider via a Modal [Secret](https://modal.com/docs/guide/secrets).
# OpenClaw supports multiple providers. This example uses Anthropic's Claude.
# Create your secret at https://modal.com/secrets with the key `ANTHROPIC_API_KEY`.

secret = modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"])

# ## Start the OpenClaw gateway

# Now we create the Sandbox and start the OpenClaw gateway.
# We use `modal.enable_output()` to print the Sandbox's image build logs to the console.

# Port 18789 is the default gateway port where OpenClaw serves its interface.

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "openclaw",
        "gateway",
        "--port",
        "18789",
        "--verbose",
        app=app,
        image=sandbox_image,
        secrets=[secret],
        encrypted_ports=[18789],
        timeout=60 * MINUTES,
    )

print(f"Sandbox ID: {sandbox.object_id}")

# After starting the sandbox, we retrieve the public URL for the exposed port.

tunnels = sandbox.tunnels()
gateway_url = tunnels[18789].url
print(f"Waiting for OpenClaw gateway at {gateway_url}")

# We poll the gateway until it's ready to accept requests.


def is_server_up(url):
    try:
        response = urllib.request.urlopen(url, timeout=5)
        return response.getcode() == 200
    except Exception:
        return False


timeout = 2 * MINUTES
start_time = time.time()
while time.time() - start_time < timeout:
    if is_server_up(gateway_url):
        print(f"OpenClaw gateway is up at {gateway_url}")
        break
    time.sleep(2)
else:
    print("Timed out waiting for OpenClaw gateway to start.")
    print("Check the sandbox logs for errors.")

# ## Connect to OpenClaw

# Once the gateway is running, you can connect to it from your messaging apps
# (Signal, Telegram, Discord, or WhatsApp) or use the CLI directly.

# To interact via the CLI, you can exec commands in the sandbox:

print("\nTo send a message via CLI, run:")
print(f'  modal sandbox exec {sandbox.object_id} openclaw agent --message "Hello"')

# ## Clean up

# When finished, you can terminate the sandbox from your [Modal dashboard](https://modal.com/containers)
# or by running:

print(f"\nTo terminate: modal sandbox terminate {sandbox.object_id}")
print("The sandbox will also terminate automatically after one hour.")
