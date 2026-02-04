# ---
# cmd: ["python", "13_sandboxes/opencode_server.py"]
# pytest: false
# ---

# # Run OpenCode in a Modal Sandbox

# This example demonstrates how to run [OpenCode](https://opencode.ai/docs/)
# remotely and connect to it from your local terminal or browser.

# Combine self-hosted OpenCode with [serving a big, smart model](https://modal.com/docs/examples/very_large_models)
# on Modal and you've got "coding agents at home"!

# Coding agents are more useful when they have more context and more tools,
# so this example also demonstrates some patterns for passing local data and setting up OpenCode.
# Here, we pass in [this Modal examples repository](https://github.com/modal-labs/modal-examples)
# and give the agent the ability to run and debug the examples -- including this one! Meta.

# ![A screenshot of the OpenCode Web UI showing this coding agent running its own code](https://modal-cdn.com/examples-opencode-server-webui.png)

# ## Set up OpenCode on Modal

import os
import secrets
from pathlib import Path

import modal

app = modal.App.lookup("example-opencode-server", create_if_missing=True)
here = Path(__file__)

# First, we define a Modal container [Image](https://modal.com/docs/guide/images)
# with OpenCode installed.

image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands("curl -fsSL https://opencode.ai/install | bash")  # install opencode
    .env({"PATH": "/root/.opencode/bin:${PATH}"})  # post-installation step
)

# ## Add OpenCode configuration

# Next, we need to add the tools our agent needs to work with the code it's operating on.
# Examples in our repo should run with nothing more than `modal` installed --
# except for a few that use `fastapi`.

image = image.uv_pip_install("modal", "fastapi~=0.128.0")

# We bring the global default OpenCode configuration along for the ride.

CONFIG_PATH = (Path("~") / ".config" / "opencode" / "opencode.json").expanduser()
if CONFIG_PATH.exists():
    print("🏖️  Including config from", CONFIG_PATH)
    image = image.add_local_file(CONFIG_PATH, "/root/.config/opencode/")

# And, because we are developing code against Modal,
# we also grant our OpenCode agent our Modal permissions.

MODAL_PATH = (Path("~") / ".modal.toml").expanduser()
if not MODAL_PATH.exists():
    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    if modal_token_id is None or modal_token_secret is None:
        raise FileNotFoundError(
            "Modal configuration file not found. Make sure you set up Modal with `modal setup` first!"
        )
    image = image.env(
        {"MODAL_TOKEN_ID": modal_token_id, "MODAL_TOKEN_SECRET": modal_token_secret}
    )
else:
    image = image.add_local_file(MODAL_PATH, "/root/.modal.toml")


# Finally, we copy over the code we want to work on.

repo_root = here.parent.parent
remote_repo_root = f"/root/{repo_root.name}"
image = image.add_local_dir(repo_root, remote_repo_root)

# Let's also secure the server. This code uses a temporary password
# generated with the `secrets` library from the Python stdlib.
# We create an ephemeral [Modal Secret](https://modal.com/docs/guide/secrets)
# to pass this to our Modal infrastructure.

password = secrets.token_urlsafe(13)
password_secret = modal.Secret.from_dict({"OPENCODE_SERVER_PASSWORD": password})

# ## Starting a Modal Sandbox with OpenCode in it

# Now, we create a [Modal Sandbox](https://modal.com/docs/guide/sandboxes)
# to run our coding agent session.
# This Sandbox has our environment Image and our password Secret.

# We open up the `OPENCODE_PORT` so that it can be accessed
# over the Internet.

print("🏖️  Creating sandbox")

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
OPENCODE_PORT = 4096

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "opencode",
        "serve",
        "--hostname=0.0.0.0",
        f"--port={OPENCODE_PORT}",
        "--log-level=DEBUG",
        "--print-logs",
        encrypted_ports=[OPENCODE_PORT],
        secrets=[password_secret],
        timeout=12 * HOURS,
        image=image,
        app=app,
        workdir=remote_repo_root,
    )

# ## Talking to OpenCode running remotely on Modal

# OpenCode is truly open -- there are many interfaces to the underlying
# coding agent server, and it's even super easy to add your own.
# That's one reason why [Ramp chose OpenCode on Modal](https://builders.ramp.com/post/why-we-built-our-background-agent)
# to deploy their in-house background agent platform.

# The commands below will print the information you need to
# - directly access the underlying Modal Sandbox for debugging or "pair coding" with the agent
# - access the Web UI from a local browser (with authentication!)
# - acess the TUI from your local terminal

print("🏖️  Access the sandbox directly:", f"modal shell {sandbox.object_id}", sep="\n\t")

tunnel = sandbox.tunnels()[OPENCODE_PORT]
print(
    "🏖️  Access the WebUI",
    f"{tunnel.url}",
    "Username: opencode",
    f"Password: {password}",
    sep="\n\t",
)
print(
    "🏖️  Access the TUI:",
    f"OPENCODE_SERVER_PASSWORD={password} opencode attach {tunnel.url}",
    sep="\n\t",
)

# Try it yourself by running this code with

# ```
# python 13_sandboxes/opencode_server.py
# ```
