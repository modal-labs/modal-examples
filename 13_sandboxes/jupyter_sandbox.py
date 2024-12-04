# ---
# cmd: ["python", "13_sandboxes/jupyter_sandbox.py"]
# pytest: false
# ---

# # Run a Jupyter notebook in a Modal Sandbox

# This example demonstrates how to run a Jupyter notebook in a Modal
# [Sandbox](https://modal.com/docs/guide/sandbox).

# ## Setting up the Sandbox

# All Sandboxes are associated with an App.

# We look up our app by name, creating it if it doesn't exist.

import json
import secrets
import time
import urllib.request

import modal

app = modal.App.lookup("example-jupyter", create_if_missing=True)

# We define a custom Docker image that has Jupyter and some other dependencies installed.
# Using a pre-defined image allows us to avoid re-installing packages on every Sandbox startup.

image = (
    modal.Image.debian_slim(python_version="3.12").pip_install("jupyter~=1.1.0")
    # .pip_install("pandas", "numpy", "seaborn")  # Any other deps
)

# ## Starting a Jupyter server in a Sandbox

# Since we'll be exposing a Jupyter server over the Internet, we need to create a password.
# We'll use `secrets` from the standard library to create a token
# and then store it in a Modal [Secret](https://modal.com/docs/guide/secrets).

token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})

# Now, we can start our Sandbox. Note our use of the `encrypted_ports` argument, which
# allows us to securely expose the Jupyter server to the public Internet. We use
# `modal.enable_output()` to print the Sandbox's image build logs to the console.

JUPYTER_PORT = 8888

print("🏖️  Creating sandbox")

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "jupyter",
        "notebook",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.allow_remote_access=1",
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=5 * 60,  # 5 minutes
        image=image,
        app=app,
        gpu=None,  # add a GPU if you need it!
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")

# ## Communicating with a Jupyter server

# Next, we print out a URL that we can use to connect to our Jupyter server.
# Note that we have to call [`Sandbox.tunnels`](https://modal.com/docs/reference/modal.Sandbox#tunnels)
# to get the URL. The Sandbox is not publicly accessible until we do so.

tunnel = sandbox.tunnels()[JUPYTER_PORT]
url = f"{tunnel.url}/?token={token}"
print(f"🏖️  Jupyter notebook is running at: {url}")

# Jupyter servers expose a [REST API](https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html)
# that you can use for programmatic manipulation.

# For example, we can check the server's status by
# sending a GET request to the `/api/status` endpoint.


def is_jupyter_up():
    try:
        response = urllib.request.urlopen(
            f"{tunnel.url}/api/status?token={token}"
        )
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
    except Exception:
        return False
    return False


# We'll now wait for the Jupyter server to be ready by hitting that endpoint.

timeout = 60  # seconds
start_time = time.time()
while time.time() - start_time < timeout:
    if is_jupyter_up():
        print("🏖️  Jupyter is up and running!")
        break
    time.sleep(1)
else:
    print("🏖️  Timed out waiting for Jupyter to start.")


# You can now open this URL in your browser to access the Jupyter notebook!

# When you're done, terminate the sandbox using your [Modal dashboard](https://modal.com/sandboxes)
# or by running `Sandbox.from_id(sandbox.object_id).terminate()`.
