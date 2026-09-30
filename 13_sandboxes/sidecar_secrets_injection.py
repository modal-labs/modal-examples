# ---
# cmd: ["python", "13_sandboxes/sidecar_secrets_injection.py"]
# pytest: false
# ---

# # Inject Secrets into a Sandbox with a proxy Sidecar

# Code running in a [Sandbox](https://modal.com/docs/guide/sandbox) often needs to call an
# external API that requires an API key. Rather than passing that key into the Sandbox,
# you can keep it in a separate, trusted container and let that container add it to
# outbound requests:

# ```
# Sandbox code  ->  proxy Sidecar  ->  gated Web Function
# (no API key)      (adds the key)
# ```

# The proxy runs as a [Sidecar](https://modal.com/docs/guide/sandbox-sidecars): a sibling
# container that shares a private network with the Sandbox. That network is reachable only
# from inside the Sandbox, so the proxy is available to your Sandbox code and to nothing
# else, while the key itself stays in a Modal
# [Secret](https://modal.com/docs/guide/secrets) mounted on the Sidecar alone.

# We use [Caddy](https://caddyserver.com/) as the proxy and a Modal
# [Web Function](https://modal.com/docs/guide/webhooks) as the
# service we are trying to access. Any proxy that can set request headers (nginx, Envoy, or something you
# write yourself) and any authenticated API will do.


import argparse
import hmac
import os
import secrets
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import fastapi
import modal

app = modal.App.lookup("example-sidecar-secrets-injection", create_if_missing=True)

MINUTES = 60  # seconds

# ## Gate an API behind a Secret

# We start by creating a secret and storing it in a Modal
# [Secret](https://modal.com/docs/guide/secrets)
#

modal.Secret.objects.create(
    "example-sidecar-api-key",
    {"API_KEY": secrets.token_urlsafe(32)},
    allow_existing=True,
)
api_secret = modal.Secret.from_name(
    "example-sidecar-api-key", required_keys=["API_KEY"]
)

# We create the Secret in this script, so there is nothing to set up
# ahead of time, but this can also be done manually as
#
# ```
# modal secret create example-api-secret API_KEY=...
# ```

# We then deploy a Web Function that requires authentication to mimic any
# external gated service you may want to access.
#
# The Sandbox will never call this function directly; it will call it
# throught the Sidecar as a proxy which in turn will call the function.

web_app = modal.App("example-sidecar-secrets-injection-web")
web_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.139.2")


@web_app.function(image=web_image, secrets=[api_secret], serialized=True)
@modal.fastapi_endpoint()
def greet(request: fastapi.Request):
    expected = os.environ["API_KEY"]
    if not hmac.compare_digest(request.headers.get("x-api-key", ""), expected):
        raise fastapi.HTTPException(status_code=401, detail="unauthorized")
    return {"message": "Hello from a secret-gated endpoint."}


with modal.enable_output():
    web_app.deploy()

upstream_url = greet.get_web_url()
if not upstream_url:
    raise RuntimeError("Web Function did not publish a URL")
upstream_host = urlparse(upstream_url).hostname
print(f"Upstream URL: {upstream_url}")

# ## Configure the proxy

# Each Sidecar is reachable from the Sandbox at the name we give it, resolved through
# `/etc/hosts` on the shared bridge network.

SIDECAR_NAME = "egress-proxy"
SIDECAR_PORT = 8080
PROXY_URL = f"http://{SIDECAR_NAME}:{SIDECAR_PORT}"

# The proxy's behavior is defined by a
# [Caddyfile](https://caddyserver.com/docs/caddyfile). Ours forwards every request to the
# Web Function, filling in the real key from the Sidecar's own environment. It also
# drops any `Authorization` header that came from the Sandbox, so the proxy decides which
# key reaches the upstream.

DEFAULT_CADDYFILE = """\
{
    admin off
}

:8080 {
    reverse_proxy https://%s {
        header_up Host %s
        header_up x-api-key {env.API_KEY}
        header_up -Authorization
    }
}
""" % (upstream_host, upstream_host)


# To point the proxy at a different upstream, or to add rules of your own — rate limits,
# path restrictions, extra headers — pass your own config instead:

parser = argparse.ArgumentParser()
parser.add_argument(
    "--caddyfile",
    type=Path,
    default=None,
    help="path to a Caddyfile to use instead of the default",
)
args = parser.parse_args()

caddyfile = args.caddyfile.read_text() if args.caddyfile else DEFAULT_CADDYFILE

# ## Build the Images

# We write the config out and copy it to the path the `caddy` Image already reads on
# startup, so the Sidecar needs no command of its own.

# Sidecars can't build their Image lazily on startup, so we
# [build it up front](https://modal.com/docs/guide/sandboxes#separating-image-builds-from-sandbox-creation)
# with `Image.build` and pass the resolved Image along.

with tempfile.TemporaryDirectory() as tmp_dir:
    caddyfile_path = Path(tmp_dir) / "Caddyfile"
    caddyfile_path.write_text(caddyfile)
    with modal.enable_output():
        sidecar_image = (
            modal.Image.from_registry("caddy:2.11")
            .add_local_file(caddyfile_path, "/etc/caddy/Caddyfile", copy=True)
            .build(app)
        )

sandbox_image = modal.Image.debian_slim(python_version="3.12")

# ## Start the Sandbox

# The Sandbox is told where the proxy is but is given no Secrets. An empty
# `outbound_cidr_allowlist` cuts off its access to the public internet while leaving the
# private network to its Sidecars intact, so the proxy becomes the Sandbox's only way out.

# We pass no command, so the Sandbox stays alive waiting for us to send it work.

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        env={"API_BASE_URL": PROXY_URL},
        outbound_cidr_allowlist=[],
        timeout=5 * MINUTES,
    )
print(f"Sandbox ID: {sandbox.object_id}")

# ## Start the proxy Sidecar

# Now we start Caddy alongside it, with the API Secret mounted here and only here.

sidecar = sandbox._experimental_sidecars.create(
    name=SIDECAR_NAME,
    image=sidecar_image,
    secrets=[api_secret],
)
print(f"Sidecar ID: {sidecar.object_id}")

# Creating a Sidecar returns as soon as its container starts, which is before Caddy is
# listening, so we wait for the port to start accepting connections.

sandbox.exec(
    "bash",
    "-c",
    f"until (echo > /dev/tcp/{SIDECAR_NAME}/{SIDECAR_PORT}) 2>/dev/null; do sleep 0.1; done",
    timeout=1 * MINUTES,
).wait()

# ## Call the API from the Sandbox

# The code below runs inside the Sandbox. It first tries to connect to the Web Function
# directly, which is blocked by the allowlist, then calls through the proxy, which swaps
# in the real key on the way through.

# Note that the call itself is unmodified from what it would be without a proxy: it
# picks up its base URL from the environment, the same way most SDKs and agent harnesses do.

UNTRUSTED_CODE = f"""
import json
import os
import socket
import urllib.request

print(f"API_KEY in Sandbox env: {{'API_KEY' in os.environ}}")

try:
    socket.create_connection(({upstream_host!r}, 443), timeout=10).close()
    print("Direct connection to {upstream_host}: succeeded")
except OSError as exc:
    print(f"Direct connection to {upstream_host}: blocked ({{type(exc).__name__}})")

url = os.environ["API_BASE_URL"]
print(f"Calling {{url}} with an invalid key")
req = urllib.request.Request(url, headers={{"x-api-key": "not-a-real-key"}})
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read())
print(f"Response: {{body['message']}}")
"""

process = sandbox.exec("python", "-c", UNTRUSTED_CODE)
for line in process.stdout:
    print(line, end="")
if process.wait() != 0:
    raise RuntimeError(process.stderr.read())

# The output should look something like

# ```
# API_KEY in Sandbox env: False
# Direct connection to workspace--example-sidecar-secrets-injection-web-greet.modal.run: blocked (TimeoutError)
# Calling http://egress-proxy:8080 with an invalid key
# Response: Hello from a secret-gated endpoint.
# ```

# Terminating the Sandbox tears down its Sidecars along with it.

sandbox.terminate()

# It's worth knowing where this boundary ends: the Sandbox never sees the key, but
# it can still use it, and can make as many calls as it likes. If that matters
# for your workload, the proxy is also the natural place to add rate limits or to restrict
# which paths it forwards.
