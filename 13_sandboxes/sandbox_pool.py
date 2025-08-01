# ---
# cmd: ["python", "13_sandboxes/sandbox_pool.py", "demo"]
# pytest: false
# ---

# # Maintain a pool of warm Sandboxes that are healthy and ready to serve requests
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
# - `example-sandbox-pool` is the main App that contains all the control logic for maintaining
#   the pool, exposing ways to claim Sandboxes, etc.
# - `example-sandbox-pool-sandboxes` houses all the actual Sandboxes, and nothing else.
#
# The implementation borrows from [pawalt](https://github.com/pawalt)'s [Sandbox pool
# example gist](https://gist.github.com/pawalt/7a505c38bba75cafae0780a5dd40e8b8). 🙏


import argparse
import time
from dataclasses import dataclass
from datetime import datetime

import modal

app = modal.App("example-sandbox-pool")

server_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]~=0.115.14",
    "requests~=2.32.4",
)

## Configuration of the pool

# Here we define the image that will be used to run the server that runs in the
# Sandbox. In this simple example, we just run the built in Python HTTP server that
# returns a directory listing.
sandbox_image = modal.Image.debian_slim(python_version="3.11")
SANDBOX_SERVER_PORT = 8080
HEALTH_CHECK_TIMEOUT_SECONDS = 10

# In this example Sandboxes live for 5 minutes, and we assume that they are used for
# 2 minutes, meaning that if a Sandbox has less than 2 minutes left it's considered
# to be expiring too soon and will be terminated.
#
# You'll want to adjust these values depending on your use case.
SANDBOX_TIMEOUT_SECONDS = 5 * 60
SANDBOX_USE_DURATION_SECONDS = 2 * 60
POOL_SIZE = 3
POOL_MAINTENANCE_SCHEDULE = modal.Period(minutes=2)


# ## Main implementation

# We keep track of all warm Sandboxes in a Modal Queue of `SandboxReference` objects.
pool_queue = modal.Queue.from_name(
    "example-sandbox-pool-sandboxes", create_if_missing=True
)


@dataclass
class SandboxReference:
    id: str
    url: str
    expires_at: int


# ### Health check
#
# We add a simple health check that just ensures that the server in the Sandbox is
# running and responding to requests.
#
# If you just want to ensure the sandbox is running you could for example check
# `sb.poll() is not None` instead.
def is_healthy(url: str, wait_for_container_start: bool) -> bool:
    """Check if a Sandbox is healthy.

    When the Sandbox is first created, the server may not imemediately accept
    connections, so if `wait_for_container_start` is True, we retry if we fail to
    connect to the server URL.
    """
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
        return False

    if check_health and not is_healthy(sr.url, wait_for_container_start=False):
        return False

    return True


# ### Adding a Sandbox to the pool
#
# This function creates and adds a new Sandbox to the pool. It runs a health check on
# the Sandbox before adding it.
#
# We deploy the Sandboxes in a separate Modal App called `example-sandbox-pool-sandboxes`,
# to separate the control app (logs, etc.) from the Sandboxes.
@app.function(image=server_image, retries=3)
@modal.concurrent(max_inputs=100)
def add_sandbox_to_queue() -> None:
    sandbox_app = modal.App.lookup(
        "example-sandbox-pool-sandboxes", create_if_missing=True
    )

    sandbox_cmd = ["python", "-m", "http.server", "8080"]
    sb = modal.Sandbox.create(
        *sandbox_cmd,
        app=sandbox_app,
        image=sandbox_image,
        encrypted_ports=[SANDBOX_SERVER_PORT],
        timeout=SANDBOX_TIMEOUT_SECONDS,
    )
    expires_at = int(time.time()) + SANDBOX_TIMEOUT_SECONDS
    url = sb.tunnels()[SANDBOX_SERVER_PORT].url

    if not is_healthy(url, wait_for_container_start=True):
        raise Exception("Health check failed")

    pool_queue.put(SandboxReference(id=sb.object_id, url=url, expires_at=expires_at))


# We also have a utility function that can be `.spawn()`ed to terminate Sandboxes.
@app.function()
def terminate_sandboxes(sandbox_ids: list[str]) -> int:
    num_terminated = 0
    for id in sandbox_ids:
        sb = modal.Sandbox.from_id(id)
        sb.terminate()
        num_terminated += 1

    print(f"Terminated {num_terminated} Sandboxes")
    return num_terminated


# ### Claiming a Sandbox from the pool
#
# We expose two ways to claim a Sandbox from the pool and get a URL to the server:
#
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
def claim_sandbox_web_endpoint(check_health: bool = True) -> str:
    return claim_sandbox.local(check_health=check_health)


@app.function(image=server_image)
def claim_sandbox(check_health: bool = True) -> str:
    to_terminate: list[str] = []

    # Remove any expiring or unhealthy sandboxes, and return the first good one:
    while True:
        print(
            "Adding a new Sandbox to the pool to backfill "
            "(and ensure we have at least one)..."
        )
        add_sandbox_to_queue.spawn()

        # timeout=None here means we block in case we need to wait for the backfill:
        sr = pool_queue.get(timeout=None)
        if sr is None:
            continue

        if not is_still_good(sr, check_health):
            print(f"Sandbox '{sr.id}' was not good - terminating and trying another...")
            to_terminate.append(sr.id)
            continue

        break

    if to_terminate:
        terminate_sandboxes.spawn(to_terminate)

    print(f"Claimed Sandbox '{sr.id}', with URL: {sr.url}")
    return sr.url


# ### Maintaining the pool
#
# This function grows or shrinks the pool to SANDBOX_POOL_SIZE. It first removes any
# expiring or unhealthy sandboxes, then adjusts the pool size to reach the target.
#
# It runs on a schedule to ensure the pool doesn't drift too far from the target size.
@app.function(
    image=server_image,
    schedule=POOL_MAINTENANCE_SCHEDULE,
)
def maintain_pool():
    to_terminate: list[str] = []

    # First remove expiring and unhealthy sandboxes
    while True:
        sr = pool_queue.get(block=False)

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
        print(f"Terminating {len(to_terminate)} expiring/unhealthy sandboxes...")
        terminate_sandboxes.spawn(to_terminate)

    # Now resize to target
    diff = POOL_SIZE - pool_queue.len()

    if diff > 0:
        for _ in add_sandbox_to_queue.starmap(() for _ in range(diff)):
            pass
    elif diff < 0:
        terminate_sandboxes.spawn(
            [sr.id for sr in pool_queue.get_many(n_values=-diff, timeout=0)]
        )

    print(f"Pool size after maintenance: {pool_queue.len()}")


# ## Local commands for interacting with the pool
#
# ### Deploy the app
#
# This also runs the `maintain_pool` function to ensure the pool is at the correct size
# without having to wait for the first scheduled maintenance run.
#
# Run it with `python 13_sandboxes/sandbox_pool.py deploy`.
def deploy():
    print("Deploying the app...")
    app.deploy()
    print("Done.")

    print("\nRunning initial pool maintenance...")
    maintain_pool.remote()
    print("Done.")


# ### Check the current state of the pool
#
# Run it with `python 13_sandboxes/sandbox_pool.py check`.
def check():
    print(f"Number of Sandboxes in the pool: {pool_queue.len()}")

    for sr in pool_queue.iterate():
        seconds_left = sr.expires_at - time.time()
        print(
            f"- Sandbox '{sr.id}' is at {sr.url} and expires at "
            f"{datetime.fromtimestamp(sr.expires_at).isoformat()} "
            f"({int(seconds_left)} seconds left)"
        )


# ### Claiming a Sandbox from the pool and print its URL
#
# This is implemented as if you wanted to call the Function from a Python backend
# application using the Modal SDK, i.e. using `.from_name()` to get the Function, etc.
#
# Run it with `python 13_sandboxes/sandbox_pool.py claim`.
def claim() -> None:
    deployed_claim_sandbox = modal.Function.from_name(
        "example-sandbox-pool", "claim_sandbox"
    )
    print(deployed_claim_sandbox.remote())


# ### Run a demo of the Sandbox pool.
#
# This is implemented as if you wanted to call the Function from a Python backend
# application using the Modal SDK, i.e. using `.from_name()` to get the Function, etc.
#
# Run it with `python 13_sandboxes/sandbox_pool.py demo`.
def demo():
    import urllib.request

    deploy()

    check()

    print("\nClaiming a Sandbox using the `claim_sandbox` Function...")
    deployed_claim_sandbox = modal.Function.from_name(
        "example-sandbox-pool", "claim_sandbox"
    )
    sandbox_url = deployed_claim_sandbox.remote()
    print(f"Claimed Sandbox URL: {sandbox_url}")

    print("\nCall the server in the Sandbox...")
    with urllib.request.urlopen(sandbox_url) as response:
        result = response.read().decode("utf-8")
        print(f"Sandbox server response:\n{result}")

    time.sleep(2)  # wait for the pool to be backfilled in the background
    check()

    deployed_web_endpoint = modal.Function.from_name(
        "example-sandbox-pool", "claim_sandbox_web_endpoint"
    )
    web_endpoint_url = deployed_web_endpoint.get_web_url()
    print(f"\nClaiming a Sandbox using the web endpoint at '{web_endpoint_url}'...")
    with urllib.request.urlopen(web_endpoint_url) as response:
        sandbox_url = response.read().decode("utf-8").strip(' "')
        print(f"Claimed Sandbox URL: {sandbox_url}")

    print("\nCall the server in the Sandbox...")
    with urllib.request.urlopen(sandbox_url) as response:
        result = response.read().decode("utf-8")
        print(f"Sandbox server response:\n{result}")

    time.sleep(2)
    check()


def main():
    parser = argparse.ArgumentParser(description="Manage Sandbox pool")
    parser.add_argument(
        "command",
        choices=["check", "deploy", "claim", "demo"],
        help="Command to execute",
    )
    args = parser.parse_args()

    if args.command == "check":
        check()
    elif args.command == "claim":
        claim()
    elif args.command == "deploy":
        deploy()
    elif args.command == "demo":
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
