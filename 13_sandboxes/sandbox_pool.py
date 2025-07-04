# ---
# cmd: ["python", "13_sandboxes/sandbox_pool.py"]
# pytest: false
# ---

# # Build a pool of warm Sandboxes that are healthy and ready to serve requests
#
# This example demonstrates how to build a pool of "warm"
# [Modal Sandboxes](https://modal.com/docs/guide/sandbox), and deploy a
# [Modal web endpoint](https://modal.com/docs/guide/webhook-urls) that let's you claim
# a Sandbox from the pool, getting a URL to the server running in the Sandbox.
#
# Maintaining a pool of warm Sandboxes is useful for example if your Sandboxes need
# to do significant work after being created, like downloading code, installing
# dependencies, or running tests, before they are ready to serve requests.
#
# It uses a [Modal Queue](https://modal.com/docs/guide/dicts-and-queues#modal-queues)
# to store references to the warm Sandboxes, and functionality to maintain the pool
# by adding and removing Sandboxes, checking the current size, etc.
#
# The pool keeps track of the time to live for each Sandbox, and will always return
# a Sandbox with at least 5 minutes left.
#
# ## Setting things up
#
# Start by deploying the Modal app:
#
# ```bash
# modal deploy 13_sandboxes/sandbox_pool.py
# ```
#
# This deploys the app with an empty pool. To fill the pool with 3 Sandboxes, run
#
# ```bash
# modal run 13_sandboxes/sandbox_pool.py::resize_pool --target 3
# ```
#
# You can check the current size of the pool by running:
#
# ```bash
# modal run 13_sandboxes/sandbox_pool.py::check_pool --verbose
# ```
#
# ## Claiming a Sandbox from the pool
#
# You can claim a Sandbox by sending a GET request to the web endpoint URL.


import time
from dataclasses import dataclass
from datetime import datetime

import modal

app = modal.App("sandbox-pool")

pool_queue = modal.Queue.from_name("sandbox-pool-buffer", create_if_missing=True)

server_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]~=0.115.14",
    "requests~=2.32.4",
)
sandbox_image = modal.Image.debian_slim(python_version="3.11")


SERVER_PORT = 8080
HEALTH_CHECK_TIMEOUT_SECONDS = 10
SANDBOX_TIMEOUT_SECONDS = 5 * 60
SANDBOX_USE_DURATION_SECONDS = 2 * 60


@dataclass
class SandboxReference:
    id: str
    url: str
    expires_at: int


# ## Health check
#
# In this example, we run a very simple health check that just ensures that the
# server is running and responding to requests.
def health_check_sandbox(url: str) -> None:
    import requests

    start_time = time.time()
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT_SECONDS:
        try:
            response = requests.get(url, timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
            response.raise_for_status()
            return
        except requests.RequestException:
            if time.time() - start_time >= HEALTH_CHECK_TIMEOUT_SECONDS:
                raise
            time.sleep(0.1)

    raise requests.RequestException("Health check failed")


# ## Adding a Sandbox to the pool
#
# This function is called by the `resize_pool` function to add a Sandbox to the pool,
# and also by the `get_sandbox` function to replace the Sandbox that was claimed.
#
# It creates a new Sandbox, runs the health check, and adds the Sandbox to the pool.
@app.function(image=server_image, retries=3)
@modal.concurrent(max_inputs=100)
def add_sandbox_to_queue() -> None:
    # This is done so we don't create Sandboxes in the ephemeral app
    deployed_app = modal.App.lookup("sandbox-pool", create_if_missing=True)

    server_command = ["python", "-m", "http.server", f"{SERVER_PORT}"]
    sb = modal.Sandbox.create(
        *server_command,
        app=deployed_app,
        image=sandbox_image,
        encrypted_ports=[SERVER_PORT],
        timeout=SANDBOX_TIMEOUT_SECONDS,
    )
    expires_at = int(time.time()) + SANDBOX_TIMEOUT_SECONDS
    url = sb.tunnels()[SERVER_PORT].url

    health_check_sandbox(url)

    pool_queue.put(SandboxReference(id=sb.object_id, url=url, expires_at=expires_at))


@app.function()
def terminate_sandboxes(sandbox_ids: list[str]) -> int:
    count = 0
    for id in sandbox_ids:
        sb = modal.Sandbox.from_id(id)
        sb.terminate()
        count += 1

    return count


# ## Claiming a Sandbox from the pool
#
# We expose two ways to claim a Sandbox from the pool:
#
# - a web endpoint, where GET requests claim a Sandbox and return the Sandbox URL.
# - a Function that can be called using the Modal SDK for [Python][1], [Go, or JS][2], etc.
#
# [1]: https://github.com/modal-labs/modal-client
# [2]: https://github.com/modal-labs/libmodal
#
# It checks the pool for a Sandbox that has enough time left, and returns the Sandbox URL.
#
# The web endpoint is deployed as a Modal web endpoint, and calls the `claim_sandbox`
# Function using `claim_sandbox.local()`, meaning that it's called in the same process
# as the web endpoint.
#
@app.function(image=server_image)
@modal.fastapi_endpoint()
@modal.concurrent(max_inputs=100)
def claim_sandbox_web_endpoint() -> str:
    return claim_sandbox.local()


@app.function()
def claim_sandbox() -> str:
    expiring_sandboxes: list[str] = []

    while True:
        # backfill + ensures the queue will have at least one Sandbox
        add_sandbox_to_queue.spawn()

        sr = pool_queue.get(timeout=None)
        if sr is None:
            continue

        if sr.expires_at < time.time() + SANDBOX_USE_DURATION_SECONDS:
            print(f"Sandbox {sr.id} does not have enough time left - removing it")
            expiring_sandboxes.append(sr.id)
            continue

        break

    if expiring_sandboxes:
        terminate_sandboxes.spawn(expiring_sandboxes)

    return sr.url


# ## Resizing the pool
#
# This function grows or shrinks the pool to the desired size.
#
# It can be called programmatically, or manually by running e.g.
# `modal run 13_sandboxes/sandbox_pool.py::resize_pool --target 3`
@app.local_entrypoint()
def resize_pool(target: int = 2):
    if target < 0:
        raise ValueError("Target pool size must be non-negative")

    current_size = pool_queue.len()
    diff = target - current_size
    actual_diff = 0

    if diff > 0:
        for _ in add_sandbox_to_queue.starmap(() for _ in range(diff)):
            actual_diff += 1
            pass
    elif diff < 0:
        actual_diff -= terminate_sandboxes.local(
            [sr.id for sr in pool_queue.get_many(n_values=-diff, timeout=0)]
        )

    print(
        f"Changed pool size by {actual_diff:+d}, now at {pool_queue.len()} sandboxes."
    )


# ## Checking the pool
#
# This function prints the current state of the pool.
#
# It can be called manually by running e.g.
# `modal run 13_sandboxes/sandbox_pool.py::check_pool --verbose`
@app.local_entrypoint()
def check_pool(verbose: bool = False):
    print(f"Number of Sandboxes in the pool: {pool_queue.len()}")
    if verbose:
        for sr in pool_queue.iterate():
            seconds_left = sr.expires_at - time.time()
            print(
                f"Sandbox '{sr.id}' is at {sr.url} and expires at "
                f"{datetime.fromtimestamp(sr.expires_at).isoformat()} "
                f"({int(seconds_left)} seconds left)"
            )


def demo():
    import urllib.parse
    import urllib.request

    app.deploy()

    print("\nSetting pool size to 3...")
    resize_pool(3)

    print("\nCurrent pool state:")
    check_pool(verbose=True)

    web_endpoint = modal.Function.from_name(
        "sandbox-pool", "claim_sandbox_web_endpoint"
    )
    web_endpoint_url = web_endpoint.get_web_url()
    print(f"\nWeb endpoint URL: {web_endpoint_url}")

    print("\nClaiming a Sandbox by sending a GET request to the web endpoint...")
    with urllib.request.urlopen(web_endpoint_url) as response:
        sandbox_server_url = response.read().decode("utf-8").strip(' "')
        print(f"URL to Sandbox server: {sandbox_server_url}")

    print("\nCall the server in the Sandbox...")
    with urllib.request.urlopen(sandbox_server_url) as response:
        result = response.read().decode("utf-8")
        print(f"Sandbox server response:\n{result}")

    time.sleep(5)  # wait for replacement Sandbox to be created

    print("\nDraining the pool back to zero...")
    resize_pool(0)

    print("\nDouble-checking that the pool is empty:")
    check_pool(verbose=True)


if __name__ == "__main__":
    demo()
