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
# a Sandbox with enough time left.
#
# It's structured into two Apps:
# - `sandbox-pool` is the main App that contains all the functionality for maintaining
#   the pool.
# - `sandbox-pool-sandboxes` houses all the actual Sandboxes, and nothing else.


import time
from dataclasses import dataclass
from datetime import datetime

import modal

app = modal.App("sandbox-pool")

server_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]~=0.115.14",
    "requests~=2.32.4",
)

## Configuration of the pool

# Here we define the image that will be used to run the server that runs in the
# Sandbox. In this simple example, we just run the built in Python HTTP server, that
# returns a directory listing.
sandbox_image = modal.Image.debian_slim(python_version="3.11")
SANDBOX_SERVER_PORT = 8080
HEALTH_CHECK_TIMEOUT_SECONDS = 10

# In this example, Sandboxes live for 5 minutes, and we assume that they are used for
# 2 minutes, meaning that if a Sandbox has less than 2 minutes left, it's considered
# to be expiring too soon, and will be terminated.
#
# You'll want to adjust these values depending on your use case.
SANDBOX_TIMEOUT_SECONDS = 5 * 60
SANDBOX_USE_DURATION_SECONDS = 2 * 60
SANDBOX_POOL_SIZE = 3


## Main implementation

# We keep track of all warm Sandboxes in a Modal Queue of SandboxReference objects.
pool_queue = modal.Queue.from_name("sandbox-pool-sandboxes", create_if_missing=True)


@dataclass
class SandboxReference:
    id: str
    url: str
    expires_at: int


# ## Health check
#
# In this example, we run a very simple health check that just ensures that the
# server is running and responding to requests.
def is_healthy(url: str, wait_for_container_start: bool) -> bool:
    import requests

    start_time = time.time()
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT_SECONDS:
        try:
            response = requests.get(url, timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
            response.raise_for_status()
            return True
        except requests.RequestException:
            if (
                not wait_for_container_start
                or time.time() - start_time >= HEALTH_CHECK_TIMEOUT_SECONDS
            ):
                return False
            time.sleep(0.1)

    return False


def is_still_good(sr: SandboxReference, check_health: bool) -> bool:
    """Check if a Sandbox is still good to use.

    It assumes that it's already been added to the pool, so we don't wait for the
    container to start.
    """
    if sr.expires_at < time.time() + SANDBOX_USE_DURATION_SECONDS:
        print(f"Sandbox '{sr.id}' does not have enough time left")
        return False

    if check_health and not is_healthy(sr.url, wait_for_container_start=False):
        print(f"Sandbox '{sr.id}' is not healthy")
        return False

    return True


# ## Adding a Sandbox to the pool
#
# This function creates and adds a new Sandbox to the pool.
#
# It runs a health check on the Sandbox before adding it.
#
# We deploy the Sandboxes in a separate Modal App called "sandbox-pool-sandboxes",
# so that we separate the control app (logs, etc.) from the Sandboxes.
@app.function(image=server_image, retries=3)
@modal.concurrent(max_inputs=100)
def add_sandbox_to_queue() -> None:
    deployed_app = modal.App.lookup("sandbox-pool-sandboxes", create_if_missing=True)

    sandbox_cmd = ["python", "-m", "http.server", "8080"]
    sb = modal.Sandbox.create(
        *sandbox_cmd,
        app=deployed_app,
        image=sandbox_image,
        encrypted_ports=[SANDBOX_SERVER_PORT],
        timeout=SANDBOX_TIMEOUT_SECONDS,
    )
    expires_at = int(time.time()) + SANDBOX_TIMEOUT_SECONDS
    url = sb.tunnels()[SANDBOX_SERVER_PORT].url

    if not is_healthy(url, wait_for_container_start=True):
        raise Exception("Health check failed")

    pool_queue.put(SandboxReference(id=sb.object_id, url=url, expires_at=expires_at))


# We also have a utility function that can be .spawn()'ed to terminate Sandboxes.
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
# - a web endpoint
# - a Function that can be called using the Modal SDK for [Python][1], [Go, or JS][2].
#
# [1]: https://github.com/modal-labs/modal-client
# [2]: https://github.com/modal-labs/libmodal
#
# The web endpoint is deployed as a [Modal web endpoint][3], and calls the
# `claim_sandbox` Function using `claim_sandbox.local()`, meaning that it's called in
# the same process as the web endpoint.
#
# The Function can be called using the Modal SDK for [Python][1], [Go, or JS][2].
#
# [1]: https://github.com/modal-labs/modal-client
# [2]: https://github.com/modal-labs/libmodal
# [3]: https://modal.com/docs/guide/webhook-urls
@app.function(image=server_image)
@modal.fastapi_endpoint()
@modal.concurrent(max_inputs=100)
def claim_sandbox_web_endpoint(check_health: bool = False) -> str:
    return claim_sandbox.local(check_health=check_health)


@app.function(image=server_image)
def claim_sandbox(check_health: bool = False) -> str:
    to_terminate: list[str] = []

    # Remove any expiring or unhealthy sandboxes, and return the first good one.
    while True:
        # Backfill and ensure the queue will have at least one Sandbox
        add_sandbox_to_queue.spawn()

        # timeout=None here means we block in case we need to wait for the backfill.
        sr = pool_queue.get(timeout=None)
        if sr is None:
            continue

        if not is_still_good(sr, check_health):
            to_terminate.append(sr.id)
            continue

        break

    if to_terminate:
        terminate_sandboxes.spawn(to_terminate)

    return sr.url


# ## Maintaining the pool
#
# This function grows or shrinks the pool to SANDBOX_POOL_SIZE. It first removes any
# expiring or unhealthy sandboxes, then adjusts the pool size to reach the target.
#
# It runs on a schedule to ensure the pool doesn't drift too far from the target size.
@app.function(schedule=modal.Period(minutes=2))
def maintain_pool():
    to_terminate: list[str] = []

    # First remove expiring and unhealthy sandboxes
    while True:
        sr = pool_queue.get(timeout=0)
        if sr is None:
            break

        if not is_still_good(sr, check_health=True):
            to_terminate.append(sr.id)
            continue

        # Found first good sandbox, but don't put it back in the queue to preserve
        # queue ordering.
        to_terminate.append(sr.id)
        break

    if to_terminate:
        print(f"Terminating {len(to_terminate)} sandboxes...")
        terminate_sandboxes.spawn(to_terminate)

    # Now resize to target
    diff = SANDBOX_POOL_SIZE - pool_queue.len()
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

    print(f"\nSetting pool size to {SANDBOX_POOL_SIZE}...")
    maintain_pool()

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
    # Note: resize_pool() now targets SANDBOX_POOL_SIZE, so we need to manually drain
    current_size = pool_queue.len()
    if current_size > 0:
        terminate_sandboxes.local(
            [sr.id for sr in pool_queue.get_many(n_values=current_size, timeout=0)]
        )

    print("\nDouble-checking that the pool is empty:")
    check_pool(verbose=True)


if __name__ == "__main__":
    demo()
