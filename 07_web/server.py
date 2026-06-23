# ---
# mypy: ignore-errors
# ---

# # Deploy HTTP Servers with ultra low latency on Modal

# Modal offers a primitive for edge-deployed, low latency web services:
# the Modal Server.

# Modal Servers are designed for applications with very demanding
# latency requirements, where a few tens of milliseconds of round-trip latency is unacceptable,
# like [low latency LLM inference](https://modal.com/docs/guide/high-performance-llm-inference).
# That ends up meaning users and clients are required to do more work.
# For Modal's higher-level primitives for web serving, see
# [this guide](https://modal.com/docs/guide/webhooks).

# This example documents a minimal Modal Server and client.

# ## How to define a Modal Server

from pathlib import Path

import modal

# To make a Modal Server, define a Python class
# with a [`modal.enter`-decorated](https://modal.com/docs/guide/lifecycle-functions) method
# that creates a subtask (thread or process) that listens for HTTP requests on some port.

# Then wrap that class in the `@app.server` decorator,
# passing in the `port` your server task is listening on
# and a `routing_region` to specify where Modal should proxy your requests through.
# This proxy will communicate directly with the containers running your server.

# To reduce end-to-end latency, include a compute Region
# that matches the routing Region and containers will be deployed into that Region.
# Note that region-pinning has cost and resource availability implications!
# See [the guide](https://modal.com/docs/guide/region-selection)
# for details.

# You can also pass the rest of your resource definitions,
# like [distributed Volume storage](https://modal.com/docs/guide/volumes),
# [CPU/memory resources](https://modal.com/docs/guide/resources),
# and [GPU type and count](https://modal.com/docs/guide/gpu),
# to `@app.server`.

# Altogether, the minimal version of a Modal Server looks something like:

PORT = 8000
COMPUTE_REGION = "us"
ROUTING_REGION = "us-east"

app = modal.App("example-server")


@app.server(
    compute_region=COMPUTE_REGION,
    routing_region=ROUTING_REGION,
    port=PORT,
    unauthenticated=True,
)
class FileServer:
    @modal.enter()
    def start(self):
        import subprocess

        subprocess.Popen(["python", "-m", "http.server", f"{PORT}"])


# ## How to write a client and tests for a Modal Server

# We test the file server defined above by requesting file from it.
# This one will do nicely.

# We put the test in a `local_entrypoint` so that we can execute it from the command line:

# ```bash
# modal run server.py
# ```


@app.local_entrypoint()
def ping():
    from urllib.error import HTTPError
    from urllib.request import urlopen

    url = FileServer.get_url()

    this = Path(__file__).name

    print(f"requesting {this} from Modal Server at {url}")

    while True:
        try:
            print(urlopen(url + f"/{this}").read().decode("utf-8"))
            break
        except HTTPError as e:
            if e.code == 503:
                import time

                time.sleep(1)
                continue
            else:
                raise e


# Notice the retry loop! Modal Clses and Functions are serverless and scale to zero by default.
# When a Modal Server has scaled to zero, clients will get a
# [503 Service Unavailable](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# error response from Modal. Those requests still trigger scale up, and once a container is ready,
# the 503s will stop and clients will receive the server's responses.

# Modal Servers also support "sticky routing" for improved cache locality within client sessions.
# For details, see [this example](https://modal.com/docs/examples/server_sticky).
