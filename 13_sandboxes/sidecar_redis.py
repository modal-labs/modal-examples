# ---
# cmd: ["python", "13_sandboxes/sidecar_redis.py"]
# pytest: false
# ---

# # Test a web app with Redis in a Sidecar
#
# In this example, we use [Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars)
# to test a small web app that runs in a [Sandbox](https://modal.com/docs/guide/sandboxes)
# and depends on Redis. The containers use separate images and communicate over
# a private bridge on the same host.
#
# Run the example with:
#
# ```bash
# python 13_sandboxes/sidecar_redis.py
# ```
#

import modal

app = modal.App.lookup("example-sidecar-redis", create_if_missing=True)

MINUTES = 60  # seconds

# ## Define the web app
#
# This is a simple web app that tracks visits to the page.

WEB_APP = r"""
import os

import redis
import uvicorn
from fastapi import FastAPI

server = FastAPI()
store = redis.Redis(
    host=os.environ["REDIS_HOST"], socket_connect_timeout=1, socket_timeout=1
)


@server.post("/visit")
def visit():
    return {"count": store.incr("visits")}


@server.get("/count")
def count():
    return {"count": int(store.get("visits") or 0)}


uvicorn.run(server)
"""

# ## Define the test
#
# The following test runs in the main Sandbox and calls the app at `localhost:8000`,
# which talks to Redis in the Sidecar.

CHECK_APP = r"""
import json
import urllib.request


def request(path, method="GET"):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, method=method)
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


assert request("/count") == {"count": 0}
assert request("/visit", "POST") == {"count": 1}
assert request("/visit", "POST") == {"count": 2}
assert request("/count") == {"count": 2}
"""

# ## Start the app and its Redis server
#
# The driver starts the app and Redis using two
# [container images](https://modal.com/docs/guide/images). Note that Sidecars
# require a pre-built image, so it is built before the Sidecar starts.
#
# A [readiness probe](https://modal.com/docs/guide/sandboxes#readiness-probes) confirms
# that the app can read from Redis and the driver runs the test.


def main():
    app_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "fastapi[standard]==0.139.2", "redis~=5.2.1"
    )
    redis_image = modal.Image.from_registry("redis:7.4.2-bookworm")

    with modal.enable_output():
        redis_image = redis_image.build(app)
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "-c",
            WEB_APP,
            app=app,
            image=app_image,
            env={"REDIS_HOST": "redis"},
            timeout=10 * MINUTES,
            readiness_probe=modal.Probe.with_exec(
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/count', timeout=2)",
            ),
        )

    try:
        sandbox._experimental_sidecars.create(
            "redis-server",
            "--protected-mode",
            "no",
            "--save",
            "",
            name="redis",
            image=redis_image,
        )
        sandbox.wait_until_ready()
        process = sandbox.exec("python", "-c", CHECK_APP)
        print(process.stdout.read(), end="")
        if process.wait() != 0:
            raise RuntimeError(process.stderr.read())
    finally:
        sandbox.terminate(wait=True)


if __name__ == "__main__":
    main()
