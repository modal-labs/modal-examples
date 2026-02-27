# ---
# mypy: ignore-errors
# ---

# # Deploy HTTP servers with ultra low latency on Modal

# Modal offers a primitive for edge-deployed, low latency web services:
# the Modal HTTP Server.

# Modal HTTP Servers are designed for applications with very demanding
# latency requirements, where a few tens of milliseconds of round-trip latency is unacceptable,
# like [low latency LLM inference](https://modal.com/docs/guide/high-performance-llm-inference).
# That ends up meaning users and clients are required to do more work.
# For Modal's higher-level primitives for web serving, see
# [this guide](https://modal.com/docs/guide/webhooks).

# This example documents a minimal Modal HTTP Server and client.

# ## How to define a Modal HTTP Server

from pathlib import Path

import modal
import modal.experimental

# Notice that we imported `modal.experimental` above.
# Modal HTTP Servers are still under development,
# so the interface is subject to change.

# To make a Modal HTTP Server, define a Python class
# with a [`modal.enter`-decorated](https://modal.com/docs/guide/lifecycle-functions) method
# that creates a subtask (thread or process) that listens for HTTP requests on some port.

# Then wrap that class in the `modal.experimental.http_server` decorator,
# passing in the `port` your server task is listening on
# and a list of `proxy_regions` where Modal should add your server to an edge proxy
# that communicates directly with the containers running your server.

# Finally, add one more decorator, `app.cls`, with the rest of your resource definitions,
# like [distributed Volume storage](https://modal.com/docs/guide/volumes)
# [CPU/memory resources](https://modal.com/docs/guide/resources),
# and [GPU type and count](https://modal.com/docs/guide/gpus).
# To reduce end-to-end latency, include a [Region](https://modal.com/docs/guide/region-selection)
# in this decorator that matches the proxy region and containers will be deployed into that Region.
# Note that region-pinning has cost and resource availability implications!
# See [the guide](https://modal.com/docs/guide/region-selection)
# for details.

# Altogether, the minimal version of a Modal HTTP Server looks something like:

PORT = 8000
REGION = "us"
PROXY_REGION = "us-east"

app = modal.App("example-http-server")


@app.cls(region=REGION)
@modal.experimental.http_server(port=PORT, proxy_regions=[PROXY_REGION])
class FileServer:
    @modal.enter()
    def start(self):
        import subprocess

        subprocess.Popen(["python", "-m", "http.server", f"{PORT}"])


# ## How to write a client and tests for a Modal HTTP Server

# We test the file server defined above by requesting file from it.
# This one will do nicely.

# We put the test in a `local_entrypoint` so that we can execute it from the command line:

# ```bash
# modal run http_server.py
# ```


@app.local_entrypoint()
def ping():
    from urllib.error import HTTPError
    from urllib.request import urlopen

    url = FileServer._experimental_get_flash_urls()[0]  # one URL per proxy region

    this = Path(__file__).name

    print(f"requesting {this} from Modal HTTP Server at {url}")

    while True:
        try:
            print(urlopen(url + f"/{this}").read().decode("utf-8"))
            break
        except HTTPError as e:
            if e.code == 503:
                continue
            else:
                raise e


# Notice the retry loop! Modal Clses and Functions are serverless and scale to zero by default.
# When a Modal HTTP Server has scaled to zero, clients will get a
# [503 Service Unavailable](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# error response from Modal. Those requests still trigger the underlying Modal Cls to scale up,
# and once a container is ready, the 503s will stop and clients will receive the server's responses.

# Modal HTTP Servers also support "sticky routing" for improved cache locality within client sessions.
# For details, see [this example](https://modal.com/docs/examples/http_server).
